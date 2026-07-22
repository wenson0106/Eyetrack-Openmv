import network
import time
import csi
import machine
from microdot import Microdot, Response
import asyncio
from servos import Servo

SSID = 'wenson'
KEY = 'RTAC1500G'
STATIC_IP = '192.168.50.200'

led = machine.LED("LED_BLUE")
latest_jpeg = None
frame_version = 0
app = Microdot()

# Initialize servo instance
servo = Servo()
csi0 = None

@app.route('/api/led/on', methods=['POST'])
async def led_on(request):
    led.on()
    return {'status': 'ok'}

@app.route('/api/led/off', methods=['POST'])
async def led_off(request):
    led.off()
    return {'status': 'ok'}

@app.route('/api/move', methods=['POST'])
async def car_move(request):
    try:
        data = request.json
        left = float(data.get('left', 0.0))
        right = float(data.get('right', 0.0))
        servo.set_speed(left, right)
        return {'status': 'ok', 'left': left, 'right': right}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 400

@app.route('/api/pan', methods=['POST'])
async def camera_pan(request):
    try:
        data = request.json
        angle = float(data.get('angle', 0.0))
        actual_angle = servo.set_angle(angle)
        return {'status': 'ok', 'angle': actual_angle}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 400

@app.route('/api/exposure_roi', methods=['POST'])
async def exposure_roi(request):
    global csi0
    try:
        data = request.json
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))
        w = int(data.get('w', 16))
        h = int(data.get('h', 12))
        
        # QQVGA is 160x120. Make sure coordinates are in bounds.
        x = max(min(x, 159), 0)
        y = max(min(y, 119), 0)
        w = max(min(w, 160 - x), 1)
        h = max(min(h, 120 - y), 1)
        
        if csi0 is not None:
            if hasattr(csi0, 'set_auto_exposure'):
                csi0.set_auto_exposure(True, roi=(x, y, w, h))
                print("Exposure ROI set to:", (x, y, w, h))
                return {'status': 'ok', 'roi': [x, y, w, h]}
        return {'status': 'unsupported_or_inactive'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 400

@app.route('/capture')
async def capture(request):
    if latest_jpeg is None:
        return '', 503
    return Response(body=latest_jpeg, headers={'Content-Type': 'image/jpeg'})

class FrameStream:
    def __aiter__(self):
        self._last_ver = -1
        return self
    async def __anext__(self):
        global latest_jpeg, frame_version
        while latest_jpeg is None or frame_version == self._last_ver:
            await asyncio.sleep_ms(10)
        self._last_ver = frame_version
        return (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                latest_jpeg + b'\r\n')

@app.route('/video_feed')
async def video_feed(request):
    return Response(body=FrameStream(),
        headers={'Content-Type': 'multipart/x-mixed-replace; boundary=frame'})

async def capture_loop():
    global latest_jpeg, frame_version, csi0
    csi0 = csi.CSI()
    csi0.reset()
    csi0.pixformat(csi.RGB565)
    csi0.framesize(csi.QQVGA)
    csi0.snapshot(time=2000)
    while True:
        img = csi0.snapshot(blocking=False)
        if img is not None:
            jpeg = img.to_jpeg(quality=50, copy=True)
            latest_jpeg = bytes(jpeg.bytearray())
            frame_version += 1
        await asyncio.sleep_ms(0)

def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.ifconfig((STATIC_IP, '255.255.255.0', '192.168.50.1', '8.8.8.8'))
    wlan.connect(SSID, KEY)
    for i in range(30):
        if wlan.isconnected():
            print('WiFi:', wlan.ifconfig()[0])
            return wlan
        time.sleep_ms(1000)
    return None

async def main():
    asyncio.create_task(capture_loop())
    await app.start_server(host='0.0.0.0', port=80)

wlan = connect_wifi()
if wlan:
    asyncio.run(main())
