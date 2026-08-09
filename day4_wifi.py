# 1. Import modules
from servos import Servo
import csi
import json
import network
import socket
import time

print("[1] Imports OK")


# 2. Wi-Fi settings and connection
WIFI_SSID = "YOUR_WIFI_NAME"
WIFI_PASSWORD = "YOUR_WIFI_PASSWORD"

print("[2] Wi-Fi: connecting...")
wlan = network.WLAN(network.STA_IF)
# wlan.active(True)
# wlan.connect(WIFI_SSID, WIFI_PASSWORD)

# wifi_wait_count = 0
# while not wlan.isconnected():
#     wifi_wait_count += 1
#     if wifi_wait_count % 10 == 0:
#         print("[2] Wi-Fi: still connecting...")
#     time.sleep_ms(200)
#     wlan = network.WLAN(network.STA_IF)

while True:
    wlan.active(True)
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)

    # Wait up to 10 seconds
    for _ in range(3):
        if wlan.isconnected():
            break
        time.sleep(1)

    if wlan.isconnected():
        break

    # Reset Wi-Fi if connection gets stuck
    print("[2] Wi-Fi: retrying...")
    wlan.deinit()
    wlan = network.WLAN(network.STA_IF)

print("[2] Wi-Fi OK, IP:", wlan.ifconfig()[0])


# 3. Camera and servo settings
HTTP_PORT = 80
DRIVE_SPEED = 0.3
JPEG_QUALITY = 35
PAN_STEP = 5

print("[3] Camera/servo: initializing...")
camera = csi.CSI()
camera.reset()
camera.pixformat(csi.RGB565)
camera.framesize(csi.QQVGA)
camera.snapshot(time=2000)

servo = Servo()
servo.set_speed(0, 0)
servo.set_angle(0)
print("[3] Camera/servo OK")


# 4. API server
def json_response(command):
    body = json.dumps({"ok": True, "command": command}).encode()
    return (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/json\r\n"
        b"Access-Control-Allow-Origin: *\r\n"
        b"Connection: close\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
    )


def handle_api(path):
    if path == "/api/forward":
        servo.set_speed(DRIVE_SPEED, DRIVE_SPEED)
        return "forward"
    if path == "/api/backward":
        servo.set_speed(-DRIVE_SPEED, -DRIVE_SPEED)
        return "backward"
    if path == "/api/left":
        servo.set_speed(-DRIVE_SPEED, DRIVE_SPEED)
        return "left"
    if path == "/api/right":
        servo.set_speed(DRIVE_SPEED, -DRIVE_SPEED)
        return "right"
    if path == "/api/stop":
        servo.set_speed(0, 0)
        return "stop"
    if path == "/api/pan_left":
        servo.set_angle(max(servo.pan_pos - PAN_STEP, -90))
        return "pan_left"
    if path == "/api/pan_right":
        servo.set_angle(min(servo.pan_pos + PAN_STEP, 90))
        return "pan_right"
    if path == "/api/pan_center":
        servo.set_angle(0)
        return "pan_center"
    return None


print("[4] API server: starting...")
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", HTTP_PORT))
server.listen(2)
server.setblocking(False)
print("[4] API server ready, port:", HTTP_PORT)


# 5. Main loop and MJPEG stream
stream_client = None
print("[5] Main loop running, waiting for requests...")

while True:
    try:
        client, address = server.accept()
        # The listening socket is non-blocking, but wait for the HTTP request
        # on the accepted client so the request is not lost.
        client.setblocking(True)
        request = client.recv(512).decode()
        path = request.split(" ")[1].split("?")[0]
        print("[5] Request:", address, path)

        if path == "/stream":
            if stream_client:
                stream_client.close()
            stream_client = client
            stream_client.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
                b"Cache-Control: no-cache\r\n\r\n"
            )
            print("[5] Stream connected OK")
        else:
            command = handle_api(path)
            if command:
                client.sendall(json_response(command))
                print("[5] API command OK:", command)
            else:
                client.sendall(
                    b"HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n"
                )
                print("[5] Unknown path:", path)
            client.close()
    except OSError:
        # Non-blocking socket normally raises OSError when there is no client.
        pass

    if stream_client:
        try:
            image = camera.snapshot()
            jpeg = image.to_jpeg(quality=JPEG_QUALITY, copy=True)
            data = bytes(jpeg.bytearray())
            stream_client.sendall(
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                + data + b"\r\n"
            )
        except OSError:
            stream_client.close()
            stream_client = None

    time.sleep_ms(10)
