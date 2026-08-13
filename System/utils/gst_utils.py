import os
import glob
import ctypes
import cv2

def preload_libraries():
    """Preload libcusparseLt.so.0 for Jetson Orin (BEFORE importing torch)"""
    matches = glob.glob(os.path.expanduser(
        "~/.local/lib/python*/site-packages/nvidia/cusparselt/lib/libcusparseLt.so.0"
    ))
    if matches:
        ctypes.CDLL(matches[0])

def suppress_ffmpeg_output():
    """Suppress verbose FFmpeg output at C level"""
    try:
        import ctypes as ct
        import ctypes.util as util
        libav_path = util.find_library("avcodec") or "libavcodec.so.58"
        libav = ct.cdll.LoadLibrary(libav_path)
        libav.av_log_set_level.argtypes = [ct.c_int]
        libav.av_log_set_level(8)   # AV_LOG_FATAL=8
    except Exception:
        pass

def check_gstreamer_support() -> bool:
    gst_available = False
    try:
        build_info = cv2.getBuildInformation()
        for line in build_info.splitlines():
            if "GStreamer" in line and "YES" in line:
                gst_available = True
                break
    except Exception:
        pass
    return gst_available

def make_gst_pipeline(url: str) -> str:
    """
    GStreamer pipeline dùng decodebin (tự chọn decoder tốt nhất).
    """
    return (
        f"rtspsrc location={url} latency=300 protocols=tcp+udp "
        "! decodebin "
        "! videoconvert "
        "! video/x-raw,format=BGR "
        "! appsink drop=1 max-buffers=4 sync=false emit-signals=false"
    )
