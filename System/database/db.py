import sqlite3
import requests
from typing import List, Dict, Any, Set
from utils.crop_engine import CropConfig
from utils.logger import get_system_logger

def load_cameras(db_path: str) -> List[Dict[str, Any]]:
    """Đọc danh sách camera đang bật từ DB."""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")   # đọc/ghi song song không block
        cur  = conn.cursor()
        cur.execute("""
            SELECT device_id, device_name, ip_address, mqtt_topic, status, enable_camera
            FROM cameras
            WHERE status = 'online'
            ORDER BY device_id
        """)
        rows = cur.fetchall()
    except Exception as e:
        get_system_logger().error(f"load_cameras error: {e}")
        return []
    finally:
        if conn is not None:
            conn.close()

    return [
        {
            "id":         row[0],
            "name":       row[1],
            "rtsp":       row[2],
            "mqtt_topic": row[3],
            "status":     row[4],
            "enable_cam": int(row[5]),
        }
        for row in rows
    ]


def sync_cameras_from_api(api_url: str, db_path: str, timeout: int = 10) -> None:
    """Đồng bộ cameras.db với dữ liệu từ API.

    - Thêm camera mới (status='online', enable_camera=0 mặc định)
    - Cập nhật name / ip_address / mqtt_topic cho camera đã có
    - Xóa camera không còn trong API
    - Nếu API lỗi → giữ nguyên DB, không thay đổi gì
    """
    # ── 1. Gọi API ────────────────────────────────────────────────────────────
    try:
        resp = requests.get(api_url, timeout=timeout)
        resp.raise_for_status()
        api_data: List[Dict] = resp.json()
    except Exception as e:
        get_system_logger().warning(f"[API sync] Could not reach API ({e}); DB left unchanged")
        return

    if not isinstance(api_data, list):
        get_system_logger().warning("[API sync] Invalid API response format (not a list); DB left unchanged")
        return

    # ── 2. Kết nối DB ────────────────────────────────────────────────────────
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")   # đọc/ghi song song không block
        cur  = conn.cursor()

        # ID đang có trong DB
        cur.execute("SELECT device_id FROM cameras")
        local_ids = {row[0] for row in cur.fetchall()}

        # ID từ API
        api_map = {cam["id"]: cam for cam in api_data if "id" in cam}
        api_ids = set(api_map.keys())

        to_add    = api_ids - local_ids
        to_update = api_ids & local_ids
        to_delete = local_ids - api_ids

        # ── Thêm mới ─────────────────────────────────────────────────────────
        for cam_id in to_add:
            cam = api_map[cam_id]
            cur.execute("""
                INSERT INTO cameras (device_id, device_name, ip_address, mqtt_topic, status, enable_camera)
                VALUES (?, ?, ?, ?, 'online', 0)
            """, (
                cam["id"],
                cam.get("name", ""),
                cam.get("rtsp_url", ""),
                cam.get("mqtt_topic", ""),
            ))
            get_system_logger().info(f"[API sync] Added camera id={cam_id} name={cam.get('name')!r}")

        # ── Cập nhật (giữ nguyên status & enable_camera) ─────────────────────
        actual_updated_count = 0
        if to_update:
            placeholders = ",".join("?" for _ in to_update)
            cur.execute(f"""
                SELECT device_id, device_name, ip_address, mqtt_topic
                FROM cameras
                WHERE device_id IN ({placeholders})
            """, tuple(to_update))
            db_cam_states = {row[0]: {"name": row[1], "rtsp": row[2], "topic": row[3]} for row in cur.fetchall()}

            for cam_id in to_update:
                cam = api_map[cam_id]
                new_name = cam.get("name", "")
                new_rtsp = cam.get("rtsp_url", "")
                new_topic = cam.get("mqtt_topic", "")
                
                old_state = db_cam_states.get(cam_id)
                if (
                    old_state is None
                    or old_state["name"] != new_name
                    or old_state["rtsp"] != new_rtsp
                    or old_state["topic"] != new_topic
                ):
                    cur.execute("""
                        UPDATE cameras
                        SET device_name = ?, ip_address = ?, mqtt_topic = ?
                        WHERE device_id = ?
                    """, (new_name, new_rtsp, new_topic, cam_id))
                    actual_updated_count += 1
                    get_system_logger().info(f"[API sync] Updated camera id={cam_id} name={new_name!r}")

        # ── Xóa camera không còn trong API ───────────────────────────────────
        for cam_id in to_delete:
            cur.execute("DELETE FROM cameras WHERE device_id = ?", (cam_id,))
            get_system_logger().info(f"[API sync] Removed camera id={cam_id}")

        conn.commit()

        if len(to_add) > 0 or len(to_delete) > 0 or actual_updated_count > 0:
            get_system_logger().info(
                f"[API sync] +{len(to_add)} added | ~{actual_updated_count} updated | -{len(to_delete)} removed"
            )

    except Exception as e:
        get_system_logger().error(f"[API sync] DB error: {e}")
    finally:
        if conn is not None:
            conn.close()

def load_zone_map(db_path: str) -> Dict[int, int]:
    """Trả về dict {camera_id: zone_id} đọc từ bảng zone_cameras."""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        cur  = conn.cursor()
        cur.execute("SELECT camera_id, zone_id FROM zone_cameras")
        rows = cur.fetchall()
        return {cam_id: zone_id for cam_id, zone_id in rows}
    except Exception as e:
        get_system_logger().error(f"load_zone_map error: {e}")
        return {}
    finally:
        if conn is not None:
            conn.close()

def load_adjacent_zones(db_path: str) -> Dict[int, Set[int]]:
    """Trả về dict {zone_id: set_of_adjacent_zone_ids} đọc từ bảng zone_relations."""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        cur  = conn.cursor()
        cur.execute("SELECT zone_a, zone_b FROM zone_relations")
        rows = cur.fetchall()
        adjacency: Dict[int, Set[int]] = {}
        for za, zb in rows:
            adjacency.setdefault(za, set()).add(zb)
            adjacency.setdefault(zb, set()).add(za)  # quan hệ 2 chiều
        return adjacency
    except Exception as e:
        get_system_logger().error(f"load_adjacent_zones error: {e}")
        return {}
    finally:
        if conn is not None:
            conn.close()

def load_crop_configs(db_path: str) -> Dict[int, CropConfig]:
    """Trả về {cam_id: CropConfig} từ bảng camera_crop_config (bao gồm roi_polygon).
    Camera không có row sẽ nhận CropConfig.default() tại runtime.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        cur  = conn.cursor()
        cur.execute("""
            SELECT cam_id, crop_mode, grid_rows, grid_cols, overlap_ratio, roi_polygon
            FROM camera_crop_config
        """)
        rows = cur.fetchall()
        return {
            row[0]: CropConfig.from_db_row(
                crop_mode=row[1],
                grid_rows=row[2],
                grid_cols=row[3],
                overlap_ratio=row[4],
                roi_polygon_json=row[5] or "[]",
            )
            for row in rows
        }
    except Exception as e:
        get_system_logger().error(f"load_crop_configs error: {e}")
        return {}
    finally:
        if conn is not None:
            conn.close()
