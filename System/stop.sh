#!/bin/bash
# ============================================================
#  stop.sh — Script tắt toàn bộ AI System (watchdog + main.py)
# ============================================================

echo "🛑 Đang tìm và tắt toàn bộ tiến trình hệ thống AI..."

# 1. Tìm PID của main.py đang chạy
MAIN_PID=$(pgrep -f "main.py")

if [ -z "$MAIN_PID" ]; then
    echo "⚠️  Không tìm thấy tiến trình main.py nào đang chạy."
    
    # Thử tìm các tiến trình bash đang chạy run.sh (dự phòng)
    WATCHDOG_PIDS=$(pgrep -f "run.sh")
    if [ -n "$WATCHDOG_PIDS" ]; then
        echo "🔪 Đang kill vòng lặp watchdog (PID: $WATCHDOG_PIDS)..."
        kill -9 $WATCHDOG_PIDS 2>/dev/null
        echo "✅ Đã dọn dẹp watchdog ngầm."
    fi
    exit 0
fi

# 2. Tìm Process ID Cha (PPID) của main.py (chính là vòng lặp bash watchdog)
PARENT_PID=$(ps -o ppid= -p $MAIN_PID | tr -d ' ')

# 3. Kill tiến trình cha (watchdog) trước để ngăn nó tự restart
if [ -n "$PARENT_PID" ] && [ "$PARENT_PID" -ne 1 ]; then
    echo "🔪 Đang kill tiến trình cha Watchdog (PID: $PARENT_PID)..."
    kill -9 $PARENT_PID 2>/dev/null
fi

# 4. Kill tiến trình main.py
echo "🔪 Đang tắt main.py (PID: $MAIN_PID)..."
# Gửi tín hiệu 15 (SIGTERM) để main.py có cơ hội dọn dẹp
kill -15 $MAIN_PID 2>/dev/null

# 5. Đợi tối đa 5 giây xem main.py đã chết hẳn chưa
for i in {1..5}; do
    if ! kill -0 $MAIN_PID 2>/dev/null; then
        echo "✅ Hệ thống đã được tắt hoàn toàn."
        exit 0
    fi
    sleep 1
done

# 6. Nếu main.py bị treo cứng không chịu tắt, ép buộc kill (SIGKILL)
echo "⚠️  main.py mất quá nhiều thời gian để tắt. Đang ép buộc tắt (Kill -9)..."
kill -9 $MAIN_PID 2>/dev/null
echo "✅ Hệ thống đã được tắt hoàn toàn."
