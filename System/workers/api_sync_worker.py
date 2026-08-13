import time
from config.settings import API_URL, API_SYNC_INTERVAL, DB_PATH
from database.db import sync_cameras_from_api
from utils.logger import get_system_logger


def api_sync_worker():
    """Background thread: sync cameras.db with API every API_SYNC_INTERVAL seconds.

    - HTTP timeout: 5s to avoid blocking thread for too long
    - Exponential backoff on repeated failures: wait = min(interval * 2^streak, 300s)
    """
    fail_streak = 0
    while True:
        wait = API_SYNC_INTERVAL if fail_streak == 0 else min(API_SYNC_INTERVAL * (2 ** fail_streak), 300)
        time.sleep(wait)

        try:
            result = sync_cameras_from_api(API_URL, DB_PATH, timeout=5)
            if fail_streak > 0:
                get_system_logger().info(f"API sync reconnected after {fail_streak} failure(s)")
            fail_streak = 0
        except Exception as e:
            fail_streak += 1
            next_wait = min(API_SYNC_INTERVAL * (2 ** fail_streak), 300)
            get_system_logger().warning(f"API sync failed (streak={fail_streak}): {e}, retry in {next_wait}s")
