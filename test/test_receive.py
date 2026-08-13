import asyncio
from amqtt.client import MQTTClient
import json
import time
import sqlite3
import os
from datetime import datetime

# ================= CONFIG =================
BROKER_HOST = "localhost"     # broker đang chạy tại broker.py trên cùng máy
BROKER_PORT = 1883
BROKER_URL  = f"mqtt://{BROKER_HOST}:{BROKER_PORT}/"
CLIENT_ID   = "machine_b_receiver"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAM_DB_PATH   = os.path.join(_PROJECT_ROOT, "test", "cameras.db")
TOPIC_RELOAD_INTERVAL = 10   # giây reload topic mới từ DB

PEOPLE_THRESHOLD = 5
LIGHT_MIN        = 1

# Ánh xạ light_level (1-4) → nhãn đọc được (khớp với get_brightness trong main.py)
LIGHT_LABEL = {
    "0": "unknown",
    "1": "dark/IR",
    "2": "dim",
    "3": "medium",
    "4": "bright",
}

# ================= ĐỌC TOPICS TỪ DB =================
def load_camera_topics() -> list[str]:
    try:
        conn = sqlite3.connect(CAM_DB_PATH)
        cur  = conn.cursor()
        cur.execute("SELECT mqtt_topic FROM cameras WHERE status = 'online'")
        rows = cur.fetchall()
        conn.close()
        return [row[0] for row in rows if row[0]]
    except Exception as e:
        print(f"⚠️  Không đọc được cameras.db: {e}")
        return []

# ================= STATE =================
latest_people  = {}
last_counts    = {"z1": -1, "z2": -1}

# Tracking thời gian nhận message theo từng topic
first_recv_time: dict[str, float] = {}  # thời điểm nhận message đầu tiên của mỗi topic
last_recv_time:  dict[str, float] = {}  # thời điểm nhận message gần nhất của mỗi topic

# ================= XỬ LÝ MESSAGE =================
async def handle_message(client, topic: str, payload_str: str):
    global latest_people, last_counts, first_recv_time, last_recv_time
    try:
        data        = json.loads(payload_str)
        now         = datetime.now().strftime("%H:%M:%S")
        recv_ts     = time.time()

        # ── Tính khoảng thời gian giữa các lần nhận ────────────────────────
        if topic not in first_recv_time:
            # Lần nhận đầu tiên của topic này
            first_recv_time[topic] = recv_ts
            print(f"[{now}] 🟢 Topic mới nhận lần đầu: {topic}")
        else:
            delta = recv_ts - last_recv_time[topic]
            print(f"[{now}] ⏱  {topic}: +{delta:.1f}s kể từ lần nhận trước")
        last_recv_time[topic] = recv_ts

        # Payload mới: person_ids (list)
        person_ids  = data.get("person_ids", [])

        # Cập nhật danh sách người mới nhất của camera này
        latest_people[topic] = set(person_ids)
        
        # Gom nhóm theo khu vực (dùng gộp Set để đếm chính xác Global ID)
        zone1_topics = {"autolight/f2/corridor/cam", "autolight/f5/corridor/cam"}
        zone1_set = set()
        zone2_set = set()
        
        for t, p_ids in latest_people.items():
            if t in zone1_topics:
                zone1_set.update(p_ids)
            else:
                zone2_set.update(p_ids)
                
        z1_count = len(zone1_set)
        z2_count = len(zone2_set)

        # Chỉ in ra dòng tổng hợp nếu như số lượng thay đổi, không in lặp lại mỗi khi nhận message
        if z1_count != last_counts["z1"] or z2_count != last_counts["z2"]:
            print(f"[{now}] 📊 Tổng người — Khu vực 1: {z1_count} | Khu vực 2: {z2_count}")
            last_counts["z1"] = z1_count
            last_counts["z2"] = z2_count

    except Exception as e:
        print(f"❌ Lỗi xử lý [{topic}]: {e}")

# ================= RELOAD TOPICS NỀN =================
async def _reload_worker(client, subscribed: set):
    while True:
        await asyncio.sleep(TOPIC_RELOAD_INTERVAL)
        topics = load_camera_topics()
        new_topics = set(topics) - subscribed
        if new_topics:
            await client.subscribe([(t, 1) for t in new_topics])
            subscribed.update(new_topics)
            print(f"📡 Đã subscribe thêm {len(new_topics)} topic mới: {new_topics}")

# ================= MAIN LOOP =================
async def main():
    client = MQTTClient(client_id=CLIENT_ID, config={"reconnect_retries": -1, "reconnect_max_interval": 5})

    while True:
        reload_task = None
        try:
            await client.connect(BROKER_URL)
            print(f"✅ Đã kết nối broker: {BROKER_URL}")

            # Subscribe topics từ DB
            topics = load_camera_topics()
            subscribed: set = set()
            if topics:
                await client.subscribe([(t, 1) for t in topics])
                subscribed = set(topics)
                print(f"📡 Đang lắng nghe {len(topics)} topics:")
                for t in topics:
                    print(f"   • {t}")
            else:
                print("⚠️  Không có topic nào trong DB — subscribe wildcard '#'")
                await client.subscribe([("#", 1)])
                subscribed.add("#")

            # Task reload topic nền
            reload_task = asyncio.create_task(_reload_worker(client, subscribed))

            # Nhận message
            while True:
                message     = await client.deliver_message()
                packet      = message.publish_packet
                topic       = packet.variable_header.topic_name
                payload_str = packet.payload.data.decode("utf-8", errors="replace")
                await handle_message(client, topic, payload_str)

        except asyncio.CancelledError:
            print("🛑 Dừng receiver.")
            return
        except Exception as e:
            print(f"⚠️  Lỗi kết nối: {e} — thử lại sau 3s...")
            await asyncio.sleep(3)
        finally:
            if reload_task and not reload_task.done():
                reload_task.cancel()
                try:
                    await reload_task
                except asyncio.CancelledError:
                    pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Đã dừng receiver.")
