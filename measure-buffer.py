import time
import cv2

CAPTURE_DEVICE_ID = 0
BUFFER_SIZE = 1  # set to None to test the driver default

cap = cv2.VideoCapture(CAPTURE_DEVICE_ID)
if BUFFER_SIZE is not None:
    ok = cap.set(cv2.CAP_PROP_BUFFERSIZE, BUFFER_SIZE)
    print(f"Set buffer size to {BUFFER_SIZE}: {'ok' if ok else 'FAILED'}")
print(f"Reported buffer size: {cap.get(cv2.CAP_PROP_BUFFERSIZE)}")
print(f"Reported FPS: {cap.get(cv2.CAP_PROP_FPS)}")

if not cap.isOpened():
    exit("Fatal: Could not open video device.")

for _ in range(30):
    cap.grab()

print("Sleeping 2 seconds to let frames accumulate...")
time.sleep(2.0)

times = []
for _ in range(30):
    t0 = time.monotonic()
    cap.grab()
    elapsed_ms = (time.monotonic() - t0) * 1000
    times.append(elapsed_ms)

buffered = 0
for i, t in enumerate(times):
    is_buffered = t < 5.0
    if is_buffered:
        buffered += 1
    label = "BUFFERED" if is_buffered else "live"
    print(f"grab {i:2d}: {t:7.2f} ms  {label}")

print(f"\nEstimated actual buffer size: {buffered}")
cap.release()
