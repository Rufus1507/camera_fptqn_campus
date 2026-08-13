"""
Native GStreamer/NVDEC RTSP frame reader for Jetson.

Why this exists instead of cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER):
this OpenCV build reports "GStreamer: NO" in cv2.getBuildInformation()
(verified on this box — it's the generic opencv-python PyPI wheel, which
never ships GStreamer support), so cv2 cannot open a GStreamer pipeline at
all. NVDEC hardware decode (the `nvv4l2decoder` element) and the Python
`gi`/`Gst`/`GstApp` bindings ARE available on this box, so this module
drives GStreamer directly via those bindings, bypassing cv2 for capture
entirely. Frames are handed back as plain BGR numpy arrays so the rest of
the pipeline (rtsp_worker.py and everything downstream) doesn't need to
change.

Two things that only showed up testing against a real camera on this box,
both handled below:
  1. `nvv4l2decoder` outputs frames in Jetson's NVMM (GPU) memory, which
     plain `videoconvert` can't read directly ("not negotiated" error) —
     `nvvidconv` bridges NVMM -> normal system memory first.
  2. Passing `rtsp://user:pass@host/...` as a single `location` string
     fails GStreamer's rtspsrc auth negotiation on this camera model
     ("No supported authentication protocol was found"), even though the
     exact same URL works fine with cv2/FFmpeg. Splitting credentials out
     into rtspsrc's `user-id`/`user-pw` properties (with a credential-free
     location) fixes it. rtspsrc is therefore built as a standalone element
     and linked to decodebin by hand via pad-added, instead of being
     embedded in one Gst.parse_launch() string.
"""

import threading
import time
from urllib.parse import urlparse, urlunparse

import numpy as np

_gst_init_lock = threading.Lock()
_gst_initialized = False
_gi_import_error = None

try:
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    from gi.repository import Gst, GstApp, GLib  # noqa: F401
except Exception as e:  # pragma: no cover - environment without gi/GStreamer
    Gst = None
    _gi_import_error = e

GST_RTSP_LOWER_TRANS_TCP = 4  # GstRTSPLowerTrans enum value for "tcp"


def _ensure_gst_init():
    global _gst_initialized
    if Gst is None:
        raise RuntimeError(f"GStreamer python bindings unavailable: {_gi_import_error}")
    with _gst_init_lock:
        if not _gst_initialized:
            Gst.init(None)
            _gst_initialized = True


def gst_nvdec_available() -> bool:
    """True if native GStreamer bindings + the NVDEC decoder element are usable here."""
    try:
        _ensure_gst_init()
        return Gst.ElementFactory.find("nvv4l2decoder") is not None
    except Exception:
        return False


def _split_credentials(url: str):
    """Return (url_without_userinfo, user, password)."""
    p = urlparse(url)
    if not p.username:
        return url, None, None
    netloc = p.hostname + (f":{p.port}" if p.port else "")
    url_no_auth = urlunparse((p.scheme, netloc, p.path, p.params, p.query, p.fragment))
    return url_no_auth, p.username, p.password


class GstRTSPReader:
    """Drop-in-ish replacement for the subset of cv2.VideoCapture this
    codebase uses: isOpened() / read() -> (ret, frame) / release().
    """

    def __init__(self, url: str, latency_ms: int = 300, open_timeout_s: float = 8.0,
                 read_timeout_s: float = 2.0):
        _ensure_gst_init()
        self.url = url
        self.latency_ms = latency_ms
        self.open_timeout_s = open_timeout_s
        self.read_timeout_s = read_timeout_s
        self._pipeline = None
        self._rtspsrc = None
        self._decodebin = None
        self._appsink = None
        self._bus = None
        self._opened = False
        self.decoder_name = None  # populated on open(), for logging/verification

    def open(self) -> bool:
        url_no_auth, user, pw = _split_credentials(self.url)

        try:
            pipeline = Gst.parse_launch(
                'decodebin name=dbin ! nvvidconv ! video/x-raw,format=BGRx '
                '! videoconvert ! video/x-raw,format=BGR '
                '! appsink name=sink emit-signals=false max-buffers=2 drop=true sync=false'
            )
            decodebin = pipeline.get_by_name("dbin")
            appsink = pipeline.get_by_name("sink")

            rtspsrc = Gst.ElementFactory.make("rtspsrc", None)
            if rtspsrc is None:
                return False
            rtspsrc.set_property("location", url_no_auth)
            if user:
                rtspsrc.set_property("user-id", user)
                rtspsrc.set_property("user-pw", pw)
            rtspsrc.set_property("latency", self.latency_ms)
            rtspsrc.set_property("protocols", GST_RTSP_LOWER_TRANS_TCP)
            pipeline.add(rtspsrc)

            def on_pad_added(_src, pad):
                struct = pad.query_caps(None).get_structure(0)
                if struct.get_value("media") == "video":
                    sinkpad = decodebin.get_static_pad("sink")
                    if not sinkpad.is_linked():
                        pad.link(sinkpad)

            rtspsrc.connect("pad-added", on_pad_added)
        except GLib.Error:
            return False
        except Exception:
            return False

        self._pipeline = pipeline
        self._rtspsrc = rtspsrc
        self._decodebin = decodebin
        self._appsink = appsink
        self._bus = pipeline.get_bus()

        try:
            self._pipeline.set_state(Gst.State.PLAYING)

            reached_playing = False
            t0 = time.time()
            while time.time() - t0 < self.open_timeout_s:
                msg = self._bus.timed_pop_filtered(
                    int(0.2 * Gst.SECOND),
                    Gst.MessageType.ERROR | Gst.MessageType.STATE_CHANGED | Gst.MessageType.EOS,
                )
                if msg is None:
                    continue
                if msg.type in (Gst.MessageType.ERROR, Gst.MessageType.EOS):
                    break
                if msg.type == Gst.MessageType.STATE_CHANGED and msg.src == self._pipeline:
                    _old, new, _pending = msg.parse_state_changed()
                    if new == Gst.State.PLAYING:
                        reached_playing = True
                        break
        except Exception:
            reached_playing = False

        if not reached_playing:
            self._cleanup()
            return False

        self.decoder_name = self._find_decoder_name()
        self._opened = True
        return True

    def _find_decoder_name(self):
        try:
            return self._recurse_find_decoder(self._pipeline)
        except Exception:
            return None

    def _recurse_find_decoder(self, bin_elem):
        it = bin_elem.iterate_elements()
        while True:
            res, elem = it.next()
            if res != Gst.IteratorResult.OK:
                break
            factory = elem.get_factory()
            fname = factory.get_name() if factory else ""
            if fname.lower() not in ("", "decodebin") and "dec" in fname.lower():
                return fname
            if isinstance(elem, Gst.Bin):
                found = self._recurse_find_decoder(elem)
                if found:
                    return found
        return None

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        """Mirrors cv2.VideoCapture.read() -> (ret: bool, frame: np.ndarray|None)."""
        if not self._opened or self._appsink is None:
            return False, None

        try:
            msg = self._bus.timed_pop_filtered(0, Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if msg is not None:
                self._opened = False
                return False, None

            sample = self._appsink.try_pull_sample(int(self.read_timeout_s * Gst.SECOND))
        except Exception:
            self._opened = False
            return False, None

        if sample is None:
            return False, None

        buf = sample.get_buffer()
        caps = sample.get_caps()
        struct = caps.get_structure(0)
        width = struct.get_value("width")
        height = struct.get_value("height")

        ok, map_info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return False, None
        try:
            arr = np.frombuffer(map_info.data, dtype=np.uint8)
            expected = height * width * 3
            if arr.size < expected:
                return False, None
            frame = arr[:expected].reshape((height, width, 3)).copy()
        finally:
            buf.unmap(map_info)

        return True, frame

    def _cleanup(self):
        if self._pipeline is not None:
            try:
                self._pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        self._pipeline = None
        self._rtspsrc = None
        self._decodebin = None
        self._appsink = None
        self._bus = None
        self._opened = False

    def release(self):
        self._cleanup()
