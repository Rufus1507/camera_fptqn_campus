# Hướng dẫn cài đặt và quản lý Hệ thống Service AI & AMQTT Broker

Tài liệu này hướng dẫn cách cài đặt, cấu hình và quản lý các dịch vụ systemd cho hệ thống AI và AMQTT Broker.

---

## 1. Dịch vụ AI System (`fpt-ai.service`)

Dịch vụ này chạy script watchdog `run.sh` để giám sát và tự động khởi chạy lại chương trình AI chính (`main.py`) khi gặp sự cố.

### Cài đặt
Chạy lệnh sau để cài đặt và kích hoạt dịch vụ tự khởi động cùng hệ thống:
```bash
cd /home/fptqn/workspace/AI_model/System
chmod +x run.sh
./run.sh install
```

### Các lệnh quản lý dịch vụ AI
* **Xem trạng thái hoạt động**:
  ```bash
  sudo systemctl status fpt-ai.service
  ```
* **Khởi động dịch vụ**:
  ```bash
  sudo systemctl start fpt-ai.service
  ```
* **Dừng dịch vụ**:
  ```bash
  sudo systemctl stop fpt-ai.service
  ```
* **Khởi động lại dịch vụ**:
  ```bash
  sudo systemctl restart fpt-ai.service
  ```
* **Hủy kích hoạt tự khởi động**:
  ```bash
  sudo systemctl disable fpt-ai.service
  ```
* **Xem log của dịch vụ trực tiếp**:
  ```bash
  journalctl -u fpt-ai.service -f
  ```

---

## 2. Dịch vụ AMQTT Broker (`fpt-amqtt-broker.service`)

Dịch vụ này khởi chạy trình môi giới tin nhắn AMQTT Broker để phục vụ truyền thông tin trong hệ thống.

### Cài đặt
Chạy lệnh sau để cài đặt dịch vụ AMQTT:
```bash
cd /home/fptqn/workspace/AI_model/System
chmod +x run_amqtt.sh
./run_amqtt.sh
```

### Các lệnh quản lý dịch vụ AMQTT Broker
* **Xem trạng thái hoạt động**:
  ```bash
  sudo systemctl status fpt-amqtt-broker.service
  ```
* **Khởi động dịch vụ**:
  ```bash
  sudo systemctl start fpt-amqtt-broker.service
  ```
* **Dừng dịch vụ**:
  ```bash
  sudo systemctl stop fpt-amqtt-broker.service
  ```
* **Khởi động lại dịch vụ**:
  ```bash
  sudo systemctl restart fpt-amqtt-broker.service
  ```
* **Hủy kích hoạt tự khởi động**:
  ```bash
  sudo systemctl disable fpt-amqtt-broker.service
  ```
* **Xem log của dịch vụ trực tiếp**:
  ```bash
  journalctl -u fpt-amqtt-broker.service -f
  ```

---

## 3. Gỡ bỏ các Service cũ (Nếu có)

Nếu hệ thống đang cài đặt các service cũ (`aisystem_fpt.service` hoặc `amqtt.service`), chạy các lệnh sau để dọn dẹp sạch sẽ:

```bash
# Dừng và vô hiệu hóa service cũ
sudo systemctl stop aisystem_fpt.service amqtt.service
sudo systemctl disable aisystem_fpt.service amqtt.service

# Xóa các file cấu hình cũ
sudo rm -f /etc/systemd/system/aisystem_fpt.service
sudo rm -f /etc/systemd/system/amqtt.service

# Tải lại cấu hình systemd
sudo systemctl daemon-reload
```