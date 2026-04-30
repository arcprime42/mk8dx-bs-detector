import atexit
import cv2
import numpy as np
import pathlib
import queue
import simpleaudio
import threading
import time

# configuration parameters
CAPTURE_DEVICE_ID = 0 # set this yourself or add your own device detection
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_FPS = 60
# try low-decode formats first; fallback to MJPG if the device/backend rejects them
CAPTURE_FOURCC_CANDIDATES = ("YUY2", "UYVY", "NV12", "MJPG")
GRAB_RATE_REPORT_SECONDS = 1.0
MATCH_QUALITY = 0.975 # a high threshold is needed to prevent false positives
COOLDOWN_SECONDS = 30 # pause scanning for 30 seconds after alert to save cpu
SCAN_EVERY_N_FRAMES = 4 # scan every 4th frame (every ~67 ms at 60fps) to save cpu
SCAN_DRAW_RATIO = 2 # draw every 2nd scan (every ~133 ms at 60fps) to save cpu
DRAW_AFTER_COOLDOWN = False # set True for more visual feedback, set False to save cpu
CV2_NUM_THREADS = 1 # 1 is ideal given this program's threading, set None to default

# mini-map location (percentage of frame) for MK8DX
ROI_LEFT_PCT = 0.724375
ROI_TOP_PCT = 0.324074
ROI_RIGHT_PCT = 0.979166
ROI_BOTTOM_PCT = 0.782407

# init video capture device
if CV2_NUM_THREADS is not None: cv2.setNumThreads(CV2_NUM_THREADS)
cap = cv2.VideoCapture(CAPTURE_DEVICE_ID)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize buffer for fresher frames
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
for fourcc in CAPTURE_FOURCC_CANDIDATES:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
if not cap.isOpened(): exit("Fatal: Could not open video device.")
ok, frame = cap.read()
if not ok: exit("Fatal: Failed to capture image.")

applied_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
applied_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
applied_fps = cap.get(cv2.CAP_PROP_FPS)
applied_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
applied_fourcc_str = "".join(chr((applied_fourcc >> (8 * i)) & 0xFF) for i in range(4))
print(f"Capture mode requested: {CAPTURE_WIDTH}x{CAPTURE_HEIGHT} @ {CAPTURE_FPS}fps, FOURCC candidates={CAPTURE_FOURCC_CANDIDATES}")
print(f"Capture mode applied: {applied_width}x{applied_height} @ {applied_fps:.2f}fps, FOURCC={applied_fourcc_str!r}")

# set the coordinates of the mini map
frame_h, frame_w, _ = frame.shape
roi_tl = (int(frame_w * ROI_LEFT_PCT), int(frame_h * ROI_TOP_PCT))
roi_br = (int(frame_w * ROI_RIGHT_PCT), int(frame_h * ROI_BOTTOM_PCT))
print(f"Video frame size: {frame_w} x {frame_h}")
print(f"Mini map top left: {roi_tl}")
print(f"Mini map bottom right: {roi_br}")

# resolve file paths relative to script location
SCRIPT_DIR = pathlib.Path(__file__).parent
TEMPLATE_PATH = SCRIPT_DIR / "bs-template.png"
SOUND_PATH = SCRIPT_DIR / "bs-alarm.wav"
for path in [TEMPLATE_PATH, SOUND_PATH]:
    if not path.exists():
        exit(f"Fatal: File not found at {path}")

# load the template
template = cv2.imread(str(TEMPLATE_PATH), cv2.IMREAD_UNCHANGED)
if template is None: exit(f"Fatal: Could not read template at {TEMPLATE_PATH}")
template_mask = template[:, :, 3]
template = cv2.cvtColor(template[:, :, :3], cv2.COLOR_BGR2GRAY)
template_h, template_w = template.shape
sound_wave = simpleaudio.WaveObject.from_wave_file(str(SOUND_PATH))

# text to display in the top center of the screen
text = "Working... press q to quit"
text_font = cv2.FONT_HERSHEY_SIMPLEX
text_scale = 0.8
text_thickness = 2
text_size = cv2.getTextSize(text, text_font, text_scale, text_thickness)[0]
text_x = (frame_w - text_size[0]) // 2
text_y = (frame_h + text_size[1]) // 10
text_org = (text_x, text_y)
text_color = (0, 255, 0)
text_linetype = cv2.LINE_AA

# threading and program state
stop_event = threading.Event()
cooldown_event = threading.Event()
scan_queue = queue.Queue(maxsize=1)
draw_queue = queue.Queue(maxsize=1)
capture_thread = None
scan_thread = None
last_alert_time = None
gray_frame = np.empty(frame.shape[:2], dtype=frame.dtype)

def capture_loop():
    try:
        _capture_loop()
    except Exception as e:
        print(f"Fatal: capture thread crashed: {e}", flush=True)
        stop_event.set()

# capture loop is rate limited to 60fps by cap.grab() blocking
# capture loop is signaled to stop by stop_event.set()
def _capture_loop(): 
    grab_errors = retrieve_errors = frame_count = scan_count = 0
    grab_report_start = time.monotonic()
    grabs_in_window = 0
    while True:
        if stop_event.is_set():
            return

        ok = cap.grab()
        if not ok:
            grab_errors += 1
            if grab_errors >= 10:
                raise RuntimeError("Too many consecutive grab errors.")
            time.sleep(0.01)
            continue
        grab_errors = 0
        grabs_in_window += 1
        now = time.monotonic()
        window_elapsed = now - grab_report_start
        if window_elapsed >= GRAB_RATE_REPORT_SECONDS:
            effective_grab_rate = grabs_in_window / window_elapsed
            print(f"effective_grab_rate: {effective_grab_rate:.2f} fps", flush=True)
            grab_report_start = now
            grabs_in_window = 0

        if cooldown_event.is_set():
            if (time.monotonic() - last_alert_time) > COOLDOWN_SECONDS:
                cooldown_event.clear()
            continue
        frame_count += 1
        needs_scan = frame_count % SCAN_EVERY_N_FRAMES == 0
        if not needs_scan:
            continue
        scan_count += 1
        needs_draw = (scan_count % SCAN_DRAW_RATIO == 0 and 
                     (last_alert_time is None or DRAW_AFTER_COOLDOWN))

        ok, frame = cap.retrieve()
        if not ok:
            retrieve_errors += 1
            if retrieve_errors >= 10:
                raise RuntimeError("Too many consecutive retrieve errors.")
            time.sleep(0.01)
            continue
        retrieve_errors = 0

        capture_ts = time.monotonic()
        try: scan_queue.get_nowait()
        except queue.Empty: pass
        scan_queue.put((frame, needs_draw, capture_ts))

def scan_loop():
    try:
        _scan_loop()
    except Exception as e:
        print(f"Fatal: scan thread crashed: {e}", flush=True)
        stop_event.set()

# scan loop is rate limited by scan_queue.get() blocking
# scan loop is signaled to stop by putting a None sentinel in the queue
def _scan_loop():
    global last_alert_time
    while True:
        item = scan_queue.get()
        if item is None:
            return
        if cooldown_event.is_set():
            continue
        frame, needs_draw, capture_ts = item
        match_found = False
        shell_tl = None
        shell_br = None

        roi_bgr = frame[roi_tl[1]:roi_br[1], roi_tl[0]:roi_br[0]]
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(roi_gray, template, cv2.TM_CCORR_NORMED, mask=template_mask)
        np.nan_to_num(result, copy=False, posinf=0.0, neginf=0.0)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= MATCH_QUALITY:
            age_at_match = time.monotonic() - capture_ts
            sound_wave.play()
            age_at_play_call = time.monotonic() - capture_ts
            print(f"Match quality: {max_val}", flush=True)
            print(f"age_at_match: {age_at_match:.3f}s, age_at_play_call: {age_at_play_call:.3f}s", flush=True)
            needs_draw = True
            match_found = True
            shell_tl = (roi_tl[0] + max_loc[0], roi_tl[1] + max_loc[1])
            shell_br = (shell_tl[0] + template_w, shell_tl[1] + template_h)
            last_alert_time = time.monotonic()
            cooldown_event.set()

        if needs_draw:
            try: draw_queue.get_nowait()
            except queue.Empty: pass
            draw_queue.put((frame, match_found, shell_tl, shell_br))

# start worker threads
capture_thread = threading.Thread(target=capture_loop, daemon=True)
scan_thread = threading.Thread(target=scan_loop, daemon=True)
capture_thread.start()
scan_thread.start()

# give the worker threads a chance to finish before the rug is pulled out with cap.release(),
# which if called mid-grab could cause an error or crash depending on the driver.
def cleanup():
    print("Shutting down...")

    # signal the capture thread to stop
    stop_event.set()
    if capture_thread is not None and capture_thread.is_alive():
        capture_thread.join(timeout=1)

    # signal the scan thread to stop
    try: scan_queue.get_nowait()
    except queue.Empty: pass
    scan_queue.put(None)
    if scan_thread is not None and scan_thread.is_alive():
        scan_thread.join(timeout=1)

    cap.release()
    cv2.destroyAllWindows()

# register cleanup to run on any exit: normal, exception, or Ctrl+C
atexit.register(cleanup)

# display loop is rate limited by cv2.waitKey()
# display loop is signaled to stop by pressing the 'q' key
while True:
    if cv2.waitKey(1) == ord("q"):
        break
    if stop_event.is_set():
        break

    try:
        frame, match_found, shell_tl, shell_br = draw_queue.get_nowait()
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=gray_frame)
        cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR, dst=frame)
        cv2.putText(frame, text, text_org, text_font, text_scale, text_color, text_thickness, text_linetype)
        cv2.rectangle(frame, roi_tl, roi_br, text_color, 2)
        if match_found: cv2.rectangle(frame, shell_tl, shell_br, text_color, 3)
        cv2.imshow("Blue Shell Detector", frame) # imshow requires the main thread on macOS
    except queue.Empty:
        pass
