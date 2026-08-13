#!/bin/bash
# ============================================================
#  run.sh — Watchdog ngoài cho AI System
#  Tự động restart khi process thoát (bất kể lý do gì).
#
#  Cách dùng:
#    chmod +x run.sh
#    ./run.sh            # chạy thẳng (giữ terminal)
#    nohup ./run.sh &    # chạy ngầm
#    # hoặc dùng systemd (khuyến nghị cho production)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MAIN_PY="$SCRIPT_DIR/main.py"
LOG_DIR="$SCRIPT_DIR/logs"

# ============================================================
# CÀI ĐẶT TỰ CHẠY KHI BẬT MÁY
# Chạy lệnh: ./run.sh install
# ============================================================
if [ "$1" == "install" ]; then
    echo "🛠 Đang cài đặt service để tự chạy khi khởi động (Systemd)..."
    SERVICE_FILE="/etc/systemd/system/fpt-ai.service"
    
    cat <<EOF | sudo tee "$SERVICE_FILE" > /dev/null
[Unit]
Description=AI System Watchdog FPT
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=/bin/bash $SCRIPT_DIR/run.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    echo "🔄 Đang reload systemd..."
    sudo systemctl daemon-reload
    echo "🚀 Đang bật tự khởi động (enable service)..."
    sudo systemctl enable fpt-ai.service
    echo "▶️ Đang chạy service..."
    sudo systemctl start fpt-ai.service
    
    echo "✅ HOÀN TẤT! Hệ thống đã được cấu hình tự động chạy khi bật máy."
    echo "🔍 Lệnh xem trạng thái : sudo systemctl status fpt-ai.service"
    echo "🛑 Lệnh dừng process  : sudo systemctl stop fpt-ai.service"
    echo "🗑️ Lệnh xóa tự chạy   : sudo systemctl disable fpt-ai.service"
    exit 0
fi

mkdir -p "$LOG_DIR"

# Python logging module writes directly to logs/system/ and logs/cam_N/
# No exec redirect needed here.

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watchdog started"
echo "  Python : $PYTHON_BIN"
echo "  Main   : $MAIN_PY"
echo "  LogDir : $LOG_DIR"

RESTART_COUNT=0

while true; do
    RESTART_COUNT=$((RESTART_COUNT + 1))
    START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

    echo ""
    echo "[${START_TIME}] Starting process (run #${RESTART_COUNT})"

    # Run main.py (stderr suppressed: TRT/ONNX/torch noise goes to /dev/null)
    cd "$SCRIPT_DIR" || exit 1
    $PYTHON_BIN -u "$MAIN_PY" 2>/dev/null
    EXIT_CODE=$?

    STOP_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${STOP_TIME}] Process exited with code: $EXIT_CODE"

    # Clean exit (Ctrl+C / manual SIGTERM) -> stop watchdog
    if [ "$EXIT_CODE" -eq 130 ] || [ "$EXIT_CODE" -eq 143 ]; then
        echo "[${STOP_TIME}] Manual stop detected, watchdog exiting"
        break
    fi

    # All other cases (CUDA error, GPU die, YOLO hang, midnight reset) -> restart
    echo "[${STOP_TIME}] Restarting in 3 seconds..."
    sleep 3
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watchdog stopped."
