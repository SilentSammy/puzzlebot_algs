import math
import socket
import threading
import time
import websocket
import json
import cv2
import numpy as np

class Puzzlebot:
    _DEFAULT_K = np.array([[791.28626825,   0.,         627.46998432],
                              [  0.,         790.76386751, 376.53014032],
                              [  0.,           0.,           1.        ]], dtype=np.float32)
    _DEFAULT_D = np.array([-3.48434917e-01,  1.53734767e-01, -1.36492904e-04,
                            -1.64288390e-04, -3.65786767e-02], dtype=np.float32)

    def __init__(self, host='192.168.137.208', port=9090, stream_port=8080, topic='/cmd_vel_safe', pose_topic='/estimated_pose', K=None, D=None, img_size=(1280, 720)):
        self.K = K if K is not None else self._DEFAULT_K
        self.D = D if D is not None else self._DEFAULT_D
        self.img_size = img_size
        self._host = host
        self._stream_port = stream_port
        self._frame = None
        self._stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self._stream_thread.start()
        self._topic = topic
        self._pose_topic = pose_topic
        self._pose = None
        try:
            self._ws = websocket.create_connection(
                f'ws://{host}:{port}',
                socket_options=((socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),)
            )
            self._ws.send(json.dumps({
                'op': 'advertise',
                'topic': topic,
                'type': 'geometry_msgs/Twist'
            }))
            self._ws.send(json.dumps({
                'op': 'subscribe',
                'topic': pose_topic,
                'type': 'geometry_msgs/PoseStamped'
            }))
        except Exception as e:
            print(f"[Puzzlebot] WebSocket unavailable, running stream-only: {e}")
            self._ws = None
        self.nominalLinearVelocity  = 0.5
        self.nominalAngularVelocity = math.pi

        # self.nominalLinearVelocity  = 0.15    # nominal linear speed (m/s)
        # self.nominalAngularVelocity = math.pi/4    # nominal angular speed (rad/s)
        self._linear_speed  = 0.0
        self._angular_speed = 0.0
        self._ws_recv_thread = threading.Thread(target=self._ws_recv_worker, daemon=True)
        self._ws_recv_thread.start()

    def _ws_recv_worker(self):
        if self._ws is None:
            return
        try:
            while True:
                raw = self._ws.recv()
                data = json.loads(raw)
                if data.get('op') == 'publish' and data.get('topic') == self._pose_topic:
                    self._pose = data.get('msg')
        except Exception:
            self._pose = None

    def _stream_worker(self):
        while True:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self._host, self._stream_port))
                sock.sendall(f'GET /?action=stream HTTP/1.0\r\nHost: {self._host}:{self._stream_port}\r\n\r\n'.encode())
                buf = b""
                while True:
                    data = sock.recv(65536)
                    if not data:
                        break
                    buf += data
                    # Skip directly to the latest complete JPEG to avoid backlog latency
                    end = buf.rfind(b'\xff\xd9')
                    if end == -1:
                        continue
                    start = buf.rfind(b'\xff\xd8', 0, end)
                    if start == -1:
                        continue
                    frame = cv2.imdecode(np.frombuffer(buf[start:end + 2], dtype=np.uint8), cv2.IMREAD_COLOR)
                    buf = buf[end + 2:]
                    if frame is not None:
                        self._frame = frame
            except Exception:
                pass
            finally:
                self._frame = None
                sock.close()
            time.sleep(1)  # wait before retrying

    def get_image(self):
        return self._frame is not None, self._frame

    def _publish(self):
        if self._ws is None:
            return
        self._ws.send(json.dumps({
            'op': 'publish',
            'topic': self._topic,
            'msg': {
                'linear':  {'x': float(self._linear_speed),  'y': 0.0, 'z': 0.0},
                'angular': {'x': 0.0, 'y': 0.0, 'z': float(self._angular_speed)}
            }
        }))

    @property
    def lin_vel(self):
        return self._linear_speed

    @lin_vel.setter
    def lin_vel(self, value):
        self._linear_speed = value

    @property
    def ang_vel(self):
        return self._angular_speed

    @ang_vel.setter
    def ang_vel(self, value):
        self._angular_speed = value

    @property
    def estimated_pose(self):
        """Latest estimated pose as a 4x4 homogeneous transform matrix, or None."""
        msg = self._pose
        if msg is None:
            return None
        p = msg['pose']['position']
        q = msg['pose']['orientation']
        x, y, z, w = q['x'], q['y'], q['z'], q['w']
        # Quaternion -> rotation matrix
        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
            [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
            [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3,  3] = [p['x'], p['y'], p['z']]
        return T


if __name__ == "__main__":
    import user_input as inp
    from ctrl_helpers import init_window, get_diff_drive_input

    car = Puzzlebot()
    WINDOW = 'Puzzlebot'
    show_camera = False

    try:
        while True:
            if inp.rising_edge('r'):
                show_camera = not show_camera
                if show_camera:
                    init_window(WINDOW, img_size=car.img_size, height=360)
                    print("Camera feed ON")
                else:
                    cv2.destroyWindow(WINDOW)
                    print("Camera feed OFF")

            cmd = get_diff_drive_input()
            car.lin_vel  = cmd['x'] * car.nominalLinearVelocity
            car.ang_vel  = cmd['w'] * car.nominalAngularVelocity
            car._publish()

            pose = car.estimated_pose
            if pose is not None:
                print("pose\n", np.array2string(pose, precision=3, suppress_small=True))

            if show_camera:
                ret, frame = car.get_image()
                if ret:
                    cv2.imshow(WINDOW, frame)
            cv2.waitKey(1)
    finally:
        car.lin_vel  = 0.0
        car.ang_vel  = 0.0
        car._publish()
        cv2.destroyAllWindows()

