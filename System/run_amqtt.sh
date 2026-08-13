#!/bin/bash
# ============================================================
#  install_amqtt.sh — Cài đặt AMQTT Broker thành Systemd Service
# ============================================================

AMQTT_BIN="/home/fptqn/.local/bin/amqtt"
SERVICE_FILE="/etc/systemd/system/fpt-amqtt-broker.service"
PYTHON_SITE="/home/fptqn/.local/lib/python3.10/site-packages"

echo "🛠 Đang cài đặt service AMQTT Broker..."

# Tạo file service với PYTHONPATH hardcode đúng
sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=AMQTT Broker Service
After=network.target

[Service]
Type=simple
User=fptqn
Environment=HOME=/home/fptqn
Environment=PYTHONPATH=${PYTHON_SITE}
ExecStart=${AMQTT_BIN}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "🔄 Đang reload systemd..."
sudo systemctl daemon-reload

echo "🚀 Đang bật tự khởi động (enable service)..."
sudo systemctl enable fpt-amqtt-broker.service

echo "▶️ Đang khởi động lại service..."
sudo systemctl restart fpt-amqtt-broker.service

sleep 2
echo ""
sudo systemctl status fpt-amqtt-broker.service --no-pager

echo ""
echo "✅ HOÀN TẤT!"
echo "🔍 Xem trạng thái    : sudo systemctl status fpt-amqtt-broker.service"
echo "📋 Xem logs trực tiếp: journalctl -u fpt-amqtt-broker.service -f"
echo "🛑 Dừng service      : sudo systemctl stop fpt-amqtt-broker.service"
echo "▶️ Khởi động lại      : sudo systemctl restart fpt-amqtt-broker.service"
echo "🗑️ Xóa tự chạy        : sudo systemctl disable fpt-amqtt-broker.service"
