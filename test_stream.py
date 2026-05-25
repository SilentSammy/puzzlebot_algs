import socket
import cv2
import numpy as np

HOST = '192.168.137.208'
PORT = 8080

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))
sock.sendall(b'GET /?action=stream HTTP/1.0\r\nHost: 192.168.137.208:8080\r\n\r\n')
print("Connected.")

cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)

buf = b""
while True:
    data = sock.recv(65536)
    if not data:
        print("Stream closed by server.")
        break
    buf += data

    while True:
        start = buf.find(b'\xff\xd8')   # JPEG SOI
        end   = buf.find(b'\xff\xd9')   # JPEG EOI
        if start == -1 or end == -1 or end < start:
            break
        frame = cv2.imdecode(np.frombuffer(buf[start:end + 2], dtype=np.uint8), cv2.IMREAD_COLOR)
        buf = buf[end + 2:]
        if frame is not None:
            cv2.imshow("Stream", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

sock.close()
cv2.destroyAllWindows()


