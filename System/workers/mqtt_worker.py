import asyncio
import time
import json
import datetime
from amqtt.client import MQTTClient
import shared.state as state
from config.settings import CLIENT_ID, MQTT_URI, MQTT_BROKER, API_SEND_INTERVAL, FPS_WINDOW, MQTT_TOPIC
from utils.logger import get_cam_logger, get_system_logger

async def _async_mqtt_sender():
    _cfg = {"reconnect_retries": 0, "reconnect_max_interval": 5}
    sys_log = get_system_logger()
    last_person_counts = {}

    while True:
        client = MQTTClient(client_id=CLIENT_ID, config=_cfg)
        try:
            await client.connect(MQTT_URI)
            sys_log.info(f"MQTT connected to broker: {MQTT_BROKER}")
        except Exception as e:
            sys_log.warning(f"MQTT connect failed: {e}, retry in 5s")
            await asyncio.sleep(5)
            continue

        try:
            while True:
                await asyncio.sleep(API_SEND_INTERVAL)
                now2    = time.time()
                cutoff2 = now2 - FPS_WINDOW
                with state.state_lock:
                    cur_ids     = list(state.CAM_IDS)
                    snap_state  = {cid: dict(state.camera_state[cid]) for cid in cur_ids if cid in state.camera_state}
                    snap_topics = dict(state.cam_topic_map)

                for cid in cur_ids:
                    if cid not in snap_state:
                        continue
                    topic = snap_topics.get(cid, MQTT_TOPIC)
                    s     = snap_state[cid]

                    person_ids    = s.get("person_ids", [])
                    current_count = len(person_ids)

                    # Log only on a real change (skip first-observation to avoid "0 -> 0")
                    old_count = last_person_counts.get(cid)
                    last_person_counts[cid] = current_count   # always track latest

                    if old_count is not None and old_count != current_count:
                        cam_ip = state.cam_ip_map.get(cid, str(cid))
                        get_cam_logger(cid).info(
                            f"Count changed: {old_count} -> {current_count} | ip={cam_ip}"
                        )

                    payload = {
                        "person_ids":  person_ids,
                        "light_level": int(s["is_night"]),
                    }
                    await client.publish(topic, json.dumps(payload).encode(), qos=0x01)

        except Exception as e:
            sys_log.error(f"MQTT send error: {e}, reconnect in 3s")
            await asyncio.sleep(3)
        finally:
            try:
                await client.disconnect()
            except Exception:
                pass


def mqtt_sender_worker():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_async_mqtt_sender())
