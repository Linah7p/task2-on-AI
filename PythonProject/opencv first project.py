import cv2
import numpy as np
import time
import json
import os
from collections import deque

CONFIG_FILE = "color_ranges.json"

DEFAULT_COLORS = {
    "Red": {"ranges": [[[0, 120, 70], [10, 255, 255]],
                       [[170, 120, 70], [180, 255, 255]]], "bgr": [0, 0, 255]},
    "Green": {"ranges": [[[40, 50, 50], [80, 255, 255]]], "bgr": [0, 255, 0]},
    "Blue": {"ranges": [[[100, 150, 50], [140, 255, 255]]], "bgr": [255, 0, 0]},
    "Yellow": {"ranges": [[[20, 100, 100], [35, 255, 255]]], "bgr": [0, 255, 255]},
    "Orange": {"ranges": [[[10, 150, 150], [20, 255, 255]]], "bgr": [0, 165, 255]},
    "Purple": {"ranges": [[[130, 50, 50], [160, 255, 255]]], "bgr": [255, 0, 255]},
}


class ColorTracker:

    def __init__(self, name, bgr, max_trail=32):
        self.name = name
        self.bgr = tuple(bgr)
        self.trail = deque(maxlen=max_trail)


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {k: v for k, v in DEFAULT_COLORS.items()}


def save_config(colors_cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(colors_cfg, f, indent=2)


def build_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def draw_hud(frame, fps, counts, min_area, recording):
    h, w = frame.shape[:2]
    panel_h = 45 + 22 * len(counts)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (230, panel_h), (20, 20, 20), -1)
    frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Min area: {min_area}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    y = 62
    for name, c in counts.items():
        cv2.putText(frame, f"{name}: {c}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 22

    if recording:
        cv2.circle(frame, (w - 20, 20), 8, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 65, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


def nothing(_):
    pass


def setup_calibration_window():
    cv2.namedWindow("Calibration")
    cv2.createTrackbar("H Min", "Calibration", 0, 180, nothing)
    cv2.createTrackbar("H Max", "Calibration", 180, 180, nothing)
    cv2.createTrackbar("S Min", "Calibration", 0, 255, nothing)
    cv2.createTrackbar("S Max", "Calibration", 255, 255, nothing)
    cv2.createTrackbar("V Min", "Calibration", 0, 255, nothing)
    cv2.createTrackbar("V Max", "Calibration", 255, 255, nothing)


def read_calibration_range():
    h_min = cv2.getTrackbarPos("H Min", "Calibration")
    h_max = cv2.getTrackbarPos("H Max", "Calibration")
    s_min = cv2.getTrackbarPos("S Min", "Calibration")
    s_max = cv2.getTrackbarPos("S Max", "Calibration")
    v_min = cv2.getTrackbarPos("V Min", "Calibration")
    v_max = cv2.getTrackbarPos("V Max", "Calibration")
    return [h_min, s_min, v_min], [h_max, s_max, v_max]


def main():
    colors_cfg = load_config()
    trackers = {name: ColorTracker(name, cfg["bgr"]) for name, cfg in colors_cfg.items()}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    min_area = 700
    calib_mode = False
    show_mask = False
    recording = False
    writer = None
    prev_time = time.time()

    clicked_hsv_point = None

    def on_mouse(event, x, y, flags, param):
        nonlocal clicked_hsv_point
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_hsv_point = (x, y)

    cv2.namedWindow("Color Recognition")
    cv2.setMouseCallback("Color Recognition", on_mouse)

    print(__doc__)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if clicked_hsv_point:
            px, py = clicked_hsv_point
            if 0 <= py < hsv.shape[0] and 0 <= px < hsv.shape[1]:
                h, s, v = hsv[py, px]
                cv2.circle(frame, (px, py), 5, (255, 255, 255), 1)
                cv2.putText(frame, f"HSV:({h},{s},{v})", (px + 8, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        counts = {}
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for name, cfg in colors_cfg.items():
            mask = build_mask(hsv, cfg["ranges"])
            combined_mask |= mask
            tracker = trackers[name]

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            count = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    count += 1
                    x, y, w, h = cv2.boundingRect(contour)
                    cx, cy = x + w // 2, y + h // 2
                    tracker.trail.appendleft((cx, cy))

                    cv2.rectangle(frame, (x, y), (x + w, y + h), tracker.bgr, 2)
                    cv2.circle(frame, (cx, cy), 4, tracker.bgr, -1)
                    cv2.putText(frame, f"{name} {int(area)}px", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, tracker.bgr, 2)

            counts[name] = count

            pts = list(tracker.trail)
            for i in range(1, len(pts)):
                thickness = max(1, int(np.sqrt(len(pts) / float(i + 1)) * 2))
                cv2.line(frame, pts[i - 1], pts[i], tracker.bgr, thickness)

        now = time.time()
        fps = 1 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now

        draw_hud(frame, fps, counts, min_area, recording)

        if calib_mode:
            lo, hi = read_calibration_range()
            calib_mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
            cv2.imshow("Calibration Mask", calib_mask)
            cv2.putText(frame, "CALIBRATION MODE (press 'a' to save color)",
                        (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        if recording and writer is not None:
            writer.write(frame)

        cv2.imshow("Color Recognition", frame)
        if show_mask:
            cv2.imshow("Combined Mask", combined_mask)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('c'):
            calib_mode = not calib_mode
            if calib_mode:
                setup_calibration_window()
            else:
                cv2.destroyWindow("Calibration")
                if cv2.getWindowProperty("Calibration Mask", 0) >= 0:
                    cv2.destroyWindow("Calibration Mask")

        elif key == ord('a') and calib_mode:
            lo, hi = read_calibration_range()
            name = input("Enter a name for this new color: ").strip() or f"Color{len(colors_cfg) + 1}"
            bgr = [int(np.random.randint(50, 255)) for _ in range(3)]
            colors_cfg[name] = {"ranges": [[lo, hi]], "bgr": bgr}
            trackers[name] = ColorTracker(name, bgr)
            save_config(colors_cfg)
            print(f"Saved new color '{name}' -> {CONFIG_FILE}")

        elif key == ord('m'):
            show_mask = not show_mask
            if not show_mask and cv2.getWindowProperty("Combined Mask", 0) >= 0:
                cv2.destroyWindow("Combined Mask")

        elif key == ord('s'):
            fname = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(fname, frame)
            print(f"Saved {fname}")

        elif key == ord('r'):
            recording = not recording
            if recording:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(f"recording_{int(time.time())}.mp4", fourcc, 20.0,
                                         (frame.shape[1], frame.shape[0]))
                print("Recording started")
            else:
                if writer:
                    writer.release()
                writer = None
                print("Recording stopped")

        elif key == ord('+'):
            min_area += 100
        elif key == ord('-'):
            min_area = max(100, min_area - 100)

    if writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()