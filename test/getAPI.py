import requests
import sqlite3
import json

API_URL = "http://0.0.0.0:8001/api/cameras"

# DB mới, tên rõ ràng riêng biệt với DB cũ
DB_PATH = "cameras_sync.db"

# ── Kết nối DB ────────────────────────────────────────────────────────────────

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# ── Tạo bảng khớp hoàn toàn với dữ liệu API trả về ──────────────────────────
# API trả về: id, name, mqtt_topic, rtsp_url
cursor.execute("""
CREATE TABLE IF NOT EXISTS cameras (
    id          INTEGER PRIMARY KEY,
    name        TEXT,
    mqtt_topic  TEXT,
    rtsp_url    TEXT
)
""")
conn.commit()


# ── Gọi API ───────────────────────────────────────────────────────────────────

def fetch_cameras_from_api():
    try:
        response = requests.get(API_URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Lỗi khi gọi API: {e}")
        return None   # None = không thể kết nối, khác với [] = API trả về rỗng


# ── Các thao tác DB ───────────────────────────────────────────────────────────

def get_all_local_ids():
    cursor.execute("SELECT id FROM cameras")
    return {row["id"] for row in cursor.fetchall()}


def insert_camera(cam):
    cursor.execute(
        "INSERT INTO cameras (id, name, mqtt_topic, rtsp_url) VALUES (?, ?, ?, ?)",
        (cam["id"], cam.get("name"), cam.get("mqtt_topic"), cam.get("rtsp_url"))
    )
    print(f"  [+] Thêm mới  id={cam['id']}  name={cam.get('name')!r}")


def update_camera(cam):
    cursor.execute(
        "UPDATE cameras SET name = ?, mqtt_topic = ?, rtsp_url = ? WHERE id = ?",
        (cam.get("name"), cam.get("mqtt_topic"), cam.get("rtsp_url"), cam["id"])
    )
    print(f"  [~] Cập nhật  id={cam['id']}  name={cam.get('name')!r}")


def delete_camera(cam_id):
    cursor.execute("DELETE FROM cameras WHERE id = ?", (cam_id,))
    print(f"  [-] Xóa       id={cam_id}")


# ── Đồng bộ chính ─────────────────────────────────────────────────────────────

def sync_cameras():
    print(f"\n=== Bắt đầu đồng bộ từ {API_URL} ===\n")

    api_data = fetch_cameras_from_api()

    if api_data is None:
        print("Không thể kết nối API — giữ nguyên DB, không thay đổi gì.")
        return

    print("--- Dữ liệu nhận được từ API ---")
    print(json.dumps(api_data, indent=4, ensure_ascii=False))
    print("---------------------------------\n")

    # ID từ API
    api_cameras = {cam["id"]: cam for cam in api_data}
    api_ids     = set(api_cameras.keys())

    # ID đang có trong DB
    local_ids   = get_all_local_ids()

    to_add    = api_ids - local_ids        # có ở API, chưa có trong DB → thêm
    to_update = api_ids & local_ids        # có cả hai → cập nhật
    to_delete = local_ids - api_ids        # chỉ còn trong DB, API không còn → xóa

    for cam_id in to_add:
        insert_camera(api_cameras[cam_id])

    for cam_id in to_update:
        update_camera(api_cameras[cam_id])

    for cam_id in to_delete:
        delete_camera(cam_id)

    conn.commit()

    print(f"\nHoàn thành: +{len(to_add)} thêm | ~{len(to_update)} cập nhật | -{len(to_delete)} xóa")
    print(f"DB lưu tại: {DB_PATH}\n")


if __name__ == "__main__":
    sync_cameras()
    conn.close()