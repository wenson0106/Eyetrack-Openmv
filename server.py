import time
import threading
import json
from urllib.request import urlopen, Request
from flask import Flask, Response, render_template, jsonify

app = Flask(__name__)
OPENMV = 'http://192.168.50.200'

latest_frame = bytearray()
frame_lock = threading.Lock()

MJPEG_BOUNDARY = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
MJPEG_END = b'\r\n--frame'

def openmv_post(path):
    r = urlopen(Request(f'{OPENMV}{path}', method='POST'), timeout=3)
    return json.loads(r.read())

def stream_reader():
    global latest_frame
    while True:
        try:
            r = urlopen(f'{OPENMV}/video_feed', timeout=None)
            buf = b''
            while True:
                chunk = r.read(8192)
                if not chunk:
                    break
                buf += chunk
                while True:
                    start = buf.find(MJPEG_BOUNDARY)
                    if start < 0:
                        break
                    jpeg_start = start + len(MJPEG_BOUNDARY)
                    end = buf.find(MJPEG_END, jpeg_start)
                    if end < 0:
                        break
                    jpeg = buf[jpeg_start:end]
                    with frame_lock:
                        latest_frame = bytearray(jpeg)
                    buf = buf[end + len(MJPEG_END):]
        except Exception:
            time.sleep(2)

threading.Thread(target=stream_reader, daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/led/on', methods=['POST'])
def led_on():
    try:
        result = openmv_post('/api/led/on')
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/led/off', methods=['POST'])
def led_off():
    try:
        result = openmv_post('/api/led/off')
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if len(latest_frame) > 0:
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                    yield bytes(latest_frame)
            time.sleep(0.03)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print('Open http://127.0.0.1:5000 in your browser')
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
