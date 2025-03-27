import cv2
import time
import threading
import playsound
import numpy


# configuration parameters
CAPTURE_DEVICE_ID = 0 # set this yourself or add your own device detection
MATCH_QUALITY = 0.975 # a high threshold is suitable when using a video capture device 
ALERT_INTERVAL_MS = 30000 # pause scanning for 30 sec after alert to save cpu
SCAN_INTERVAL_MS = 67 # scan every 4 frames or about 67 ms to save cpu
DRAW_INTERVAL_MS = 100 # only draw once per 100 ms interval to save cpu
DRAW_BETWEEN_ALERTS = False # do not update the screen between alerts to save cpu


# load the template
template = cv2.imread("bs-template.png", cv2.IMREAD_UNCHANGED)
template, template_alpha = template[:, :, :3], template[:, :, 3]
template_mask = template_alpha > 0
template_mask = template_mask.astype(numpy.uint8) * 255
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template_w = template.shape[1]
template_h = template.shape[0]


# init sound
def play_sound():
    playsound.playsound("bs-alarm.mp3")


# init video capture device
cap = cv2.VideoCapture(CAPTURE_DEVICE_ID)
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()
ret, frame = cap.read()
if not ret:
    print("Error: Failed to capture image.")
    exit()


# set the coordinates of the mini map, adjust here if needed
frame_h, frame_w, _ = frame.shape
top_left = (int(frame_w * 0.724375), int(frame_h * 0.324074))
bottom_right = (int(frame_w * 0.979166), int(frame_h * 0.782407))
print(f"Video frame size: {frame_w} x {frame_h}")
print(f"Mini map top left: {top_left}")
print(f"Mini map bottom right: {bottom_right}")


text = "Working... press q to quit"
text_font = cv2.FONT_HERSHEY_SIMPLEX
text_scale = 0.8
text_thickness = 2
text_size = cv2.getTextSize(text, text_font, text_scale, text_thickness)[0]
text_x = (frame_w - text_size[0]) // 2
text_y = (frame_h + text_size[1]) // 10
text_color = (0, 255, 0)

last_alert_time_ms = 0
last_draw_time_ms = 0
count_errors = 0

while True:
    current_time_ms = int(time.time() * 1000)
    keypress = None

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        count_errors += 1
        if count_errors >= 10:
            print("Fatal: Too many read errors.")
            break
        continue
    count_errors = 0

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    result = cv2.matchTemplate(roi, template, cv2.TM_CCORR_NORMED, mask=template_mask)
    result[numpy.isinf(result)] = 0 # eliminate any erroneous infinite values
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val > MATCH_QUALITY:
        last_alert_time_ms = current_time_ms

        thread = threading.Thread(target=play_sound, daemon=True)
        thread.start()
        print(f"Match quality: {max_val}", flush=True)

        # convert match coordinates in the roi to full frame size
        top_left_roi = max_loc
        bottom_right_roi = (top_left_roi[0] + template_w, top_left_roi[1] + template_h)
        top_left_full = (top_left[0] + top_left_roi[0], top_left[1] + top_left_roi[1])
        bottom_right_full = (top_left[0] + bottom_right_roi[0], top_left[1] + bottom_right_roi[1])

        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(frame, top_left_full, bottom_right_full, (0, 255, 0), 3)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(frame, text, (text_x, text_y), text_font, text_scale, text_color, text_thickness, cv2.LINE_AA)
        cv2.imshow("Blue Shell Detector", frame)
        keypress = cv2.waitKey(ALERT_INTERVAL_MS)

    elif (last_alert_time_ms == 0 or DRAW_BETWEEN_ALERTS) and (current_time_ms - last_draw_time_ms >= DRAW_INTERVAL_MS):
        last_draw_time_ms = current_time_ms

        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(frame, text, (text_x, text_y), text_font, text_scale, text_color, text_thickness, cv2.LINE_AA)
        cv2.imshow("Blue Shell Detector", frame)
        elapsed_time = int(time.time() * 1000) - current_time_ms
        keypress = cv2.waitKey(max(1, SCAN_INTERVAL_MS - elapsed_time))

    else:
        elapsed_time = int(time.time() * 1000) - current_time_ms
        keypress = cv2.waitKey(max(1, SCAN_INTERVAL_MS - elapsed_time))

    if keypress == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

