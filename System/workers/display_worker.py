import cv2
import time
import numpy as np
import shared.state as state
from config.settings import DISPLAY_FPS, DISPLAY_CELL_W, DISPLAY_CELL_H, DISPLAY_COLS
from utils.crop_engine import CropConfig, build_roi_mask, _cell_roi_coverage
from utils.logger import get_system_logger
import os
import threading
from flask import Flask, Response, render_template_string

app = Flask(__name__)
latest_grid = None
grid_lock = threading.Lock()

def grid_update_loop():
    global latest_grid
    
    interval = 1.0 / DISPLAY_FPS

    CELL_W = DISPLAY_CELL_W
    CELL_H = DISPLAY_CELL_H
    COLS   = DISPLAY_COLS

    BOX_COLOR   = (0, 220, 0)
    TEXT_COLOR  = (255, 255, 255)
    SHADOW_COLOR= (0, 0, 0)
    COUNT_BG    = (0, 180, 0)

    while True:
        t0 = time.perf_counter()

        with state.state_lock:
            # Chỉ lấy các camera có quyền hiển thị
            cur_ids = [cid for cid in state.CAM_IDS if state.cam_display_enabled.get(cid, 0) == 1]

        n = len(cur_ids)
        if n == 0:
            blank = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
            cv2.putText(blank, "No cameras", (20, CELL_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT_COLOR, 2)
            with grid_lock:
                latest_grid = blank
            
            elapsed = time.perf_counter() - t0
            wait_s = interval - elapsed
            if wait_s > 0:
                time.sleep(wait_s)
            continue

        rows = (n + COLS - 1) // COLS
        grid = np.zeros((rows * CELL_H, COLS * CELL_W, 3), dtype=np.uint8)

        with state.state_lock:
            snap_state = {cid: dict(state.camera_state[cid]) for cid in cur_ids if cid in state.camera_state}
        with state.display_lock:
            snap_frames = dict(state.display_frames)

        for idx, cid in enumerate(sorted(cur_ids)):
            row_i = idx // COLS
            col_i = idx % COLS
            y_off = row_i * CELL_H
            x_off = col_i * CELL_W

            frame_src = snap_frames.get(cid)
            if frame_src is None:
                cell = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
                cv2.putText(cell, f"Cam {cid}: connecting...",
                            (10, CELL_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (100, 100, 100), 1)
            else:
                orig_h, orig_w = frame_src.shape[:2]
                cell = cv2.resize(frame_src, (CELL_W, CELL_H))
                scale_x = CELL_W / orig_w
                scale_y = CELL_H / orig_h

                s = snap_state.get(cid, {})
                boxes = s.get("boxes", [])
                for box in boxes:
                    bx1 = int(box[0] * scale_x)
                    by1 = int(box[1] * scale_y)
                    bx2 = int(box[2] * scale_x)
                    by2 = int(box[3] * scale_y)
                    tid      = int(box[4]) if len(box) > 4 else 0
                    conf_pct = int(box[5] * 100) if len(box) > 5 else int(box[4] * 100)
                    
                    hue = (tid * 47) % 180
                    hsv_color = np.uint8([[[hue, 220, 220]]])
                    bgr_color  = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0].tolist()
                    cv2.rectangle(cell, (bx1, by1), (bx2, by2), bgr_color, 2)
                    
                    label = f"G_{tid} {conf_pct}%"
                    lx, ly = bx1, max(by1 - 6, 14)
                    cv2.putText(cell, label, (lx + 1, ly + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, SHADOW_COLOR, 2)
                    cv2.putText(cell, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 1)

                # ──────────────────────────────────────────────────────────
                # Lớp 1, 2, 3: Mạng lưới cây phân vùng + ROI overlay
                # ──────────────────────────────────────────────────────────
                cfg: CropConfig = state.cam_crop_configs.get(cid) or CropConfig.default()

                if cfg.crop_mode == 1 and (cfg.grid_rows > 1 or cfg.grid_cols > 1):
                    # Build display-scale ROI mask (kích thước CELL_W × CELL_H)
                    display_roi_mask = build_roi_mask(CELL_H, CELL_W, cfg.roi_polygons)

                    # ─ Lớp 1: Dark overlay lên cell bị filter (coverage < 5%) ─
                    cell_rows = cfg.grid_rows
                    cell_cols = cfg.grid_cols
                    for gr in range(cell_rows):
                        for gc in range(cell_cols):
                            cx0_d = int(gc * CELL_W / cell_cols)
                            cx1_d = int((gc + 1) * CELL_W / cell_cols)
                            cy0_d = int(gr * CELL_H / cell_rows)
                            cy1_d = int((gr + 1) * CELL_H / cell_rows)
                            cov = _cell_roi_coverage(display_roi_mask, cy0_d, cy1_d, cx0_d, cx1_d)
                            if cov < 0.05:
                                # Vẽ overlay tối bán trong suốt
                                overlay_region = cell[cy0_d:cy1_d, cx0_d:cx1_d]
                                dark = (overlay_region * 0.45).astype(np.uint8)
                                cell[cy0_d:cy1_d, cx0_d:cx1_d] = dark

                    # ─ Lớp 2: Vẽ đường lưới cyan (chỉ cho cell ACTIVE) ─
                    for ci in range(1, cell_cols):
                        vx = int(ci * CELL_W / cell_cols)
                        cv2.line(cell, (vx, 0), (vx, CELL_H), (0, 255, 255), 1)
                    for ri in range(1, cell_rows):
                        hy = int(ri * CELL_H / cell_rows)
                        cv2.line(cell, (0, hy), (CELL_W, hy), (0, 255, 255), 1)

                    # ─ Lớp 3: Vẽ polygon ROI màu cam (orange) ─
                    if cfg.roi_polygons:
                        for polygon_norm in cfg.roi_polygons:
                            if not polygon_norm:
                                continue
                            pts = np.array(
                                [[int(x * CELL_W), int(y * CELL_H)] for x, y in polygon_norm],
                                dtype=np.int32,
                            )
                            cv2.polylines(cell, [pts], isClosed=True, color=(0, 140, 255), thickness=2)

                elif cfg.crop_mode == 1:  # 1×1 grid (không cắt) — không vẽ lưới
                    # Chỉ vẽ polygon ROI nếu có
                    if cfg.roi_polygons:
                        for polygon_norm in cfg.roi_polygons:
                            if not polygon_norm:
                                continue
                            pts = np.array(
                                [[int(x * CELL_W), int(y * CELL_H)] for x, y in polygon_norm],
                                dtype=np.int32,
                            )
                            cv2.polylines(cell, [pts], isClosed=True, color=(0, 140, 255), thickness=2)

                cam_ip  = state.cam_ip_map.get(cid, str(cid))
                fps_val = s.get("fps", 0.0)
                info_txt = f"Cam {cam_ip}  |  {fps_val:.1f} fps"
                cv2.putText(cell, info_txt, (11, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, SHADOW_COLOR, 3)
                cv2.putText(cell, info_txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1)

                n_people = len(s.get("person_ids", []))
                count_txt = f"{n_people} nguoi"
                (tw, th), _ = cv2.getTextSize(count_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cx = CELL_W - tw - 12
                cy = 32
                cv2.rectangle(cell, (cx - 6, cy - th - 6), (cx + tw + 6, cy + 4), COUNT_BG, -1)
                cv2.putText(cell, count_txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)

            grid[y_off:y_off + CELL_H, x_off:x_off + CELL_W] = cell

        with grid_lock:
            latest_grid = grid

        elapsed = time.perf_counter() - t0
        wait_s = interval - elapsed
        if wait_s > 0:
            time.sleep(wait_s)

def generate_web_frames():
    while True:
        with grid_lock:
            frame = latest_grid.copy() if latest_grid is not None else None
        
        if frame is None:
            time.sleep(0.1)
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            time.sleep(0.1)
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(1.0 / DISPLAY_FPS)

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Camera AI - Smart Tracker</title>
        <style>
            body { 
                background-color: #121212; 
                color: #e0e0e0; 
                display: flex; 
                flex-direction: column; 
                align-items: center; 
                margin: 0; 
                padding: 20px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            }
            h1 { 
                margin-bottom: 20px; 
                font-weight: 300;
                letter-spacing: 1px;
                color: #ffffff;
            }
            .stream-container {
                max-width: 95%;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 8px 30px rgba(0,0,0,0.5);
                background-color: #1e1e1e;
                border: 1px solid #333;
                padding: 5px;
            }
            img { 
                max-width: 100%; 
                height: auto; 
                display: block;
                border-radius: 8px;
            }
        </style>
    </head>
    <body>
        <h1>Live Monitoring Dashboard</h1>
        <div class="stream-container">
            <img src="{{ url_for('video_feed') }}" alt="Video Stream">
        </div>
    </body>
    </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_web_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def display_worker():
    get_system_logger().info("[Display] Starting background grid builder")
    update_thread = threading.Thread(target=grid_update_loop, daemon=True)
    update_thread.start()

    # Tắt log quá mức của werkzeug
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    get_system_logger().info("[Display] Starting web stream at http://0.0.0.0:5050 (replaces cv2.imshow)")
    # Flask's built-in dev server spawns one unbounded thread per HTTP
    # connection (no pool cap, no lifecycle mgmt) — not safe for a 24/7
    # stream. waitress is a production WSGI server with a bounded worker
    # thread pool instead.
    from waitress import serve
    serve(app, host='0.0.0.0', port=5050, threads=8)
