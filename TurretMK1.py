import time

import cv2
from ultralytics import YOLO

# --- settings you can change ---#
CAMERA_INDEX = 0  # camera index (0 for default camera)
CONFIDENCE = 0.45
CHEST_Y = 0.45
STICKY_FRAMES = 20
LOST_FRAMES_TO_STOP = 10
MODEL_NAME = "yolov8n.pt"


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def chest_point(box):
    x1, y1, x2, y2 = box
    h = y2 - y1
    ratio = 0.34 if h > 380 else 0.50
    cx = int((x1 + x2) / 2)
    cy = int(y1 + h * ratio)
    return cx, cy


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0

def is_red_hat(frame, box):
    x1, y1, x2, y2 = box
    h = max(1, y2 - y1)
    top = frame[max(0, y1): y1 + h // 3, max(0, x1): x2]
    if top.size == 0:
        return False
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 125, 125), (10, 255, 255))
    m2 = cv2.inRange(hsv, (172, 125, 125), (180, 255, 255))
    red = cv2.bitwise_or(m1, m2)
    return (red > 0).mean() > 0.08

def main():
    print("loading YOLOv8n (first run downloads the model)...")
    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Could not open camera. Try CAMERA_INDEX = 1")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


    target = None
    people = []
    armed = False
    sticky_box = None
    sticky_count = 0
    lost_count = 0
    last_print = ""

    print("camera on. press A to arm, Q to quit")

    while True:
        ok, frame = cap.read()
        frame = cv2.flip(frame, 1)  # mirror image
        if not ok:
            print("lost camera frame")
            break

        results = model.predict(
            frame, classes=[0], conf=CONFIDENCE, verbose=False  # only person class
        )[0]

        people = []
        if results.boxes is not None:
            for b in results.boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                people.append((x1, y1, x2, y2))
                people =[p for p in people if not is_red_hat(frame, p)]  # filter out red hats
        target = None
        if people:
            closest = max(people, key=box_area)

            if sticky_box is not None:
                best_match = max(people, key=lambda p: iou(p, sticky_box))
                if iou(best_match, sticky_box) > 0.25:
                    sticky_count += 1
                    if sticky_count >= STICKY_FRAMES:
                        target = best_match
                    else:
                        if box_area(closest) > box_area(best_match):
                            target = closest
                            sticky_count = 0
                        else:
                            target = best_match

                else:
                    target = closest
                    sticky_count = 0
            else:
                target = closest
                sticky_count = 0
            sticky_box = target
            if target is not None and is_red_hat(frame, target):
                others = [p for p in people if p != target and not is_red_hat(frame, p)]
                if others:
                    target = max(others, key=box_area)
                else:
                    target = None           
            
            lost_count = 0
        else:
            lost_count += 1
            if lost_count >= LOST_FRAMES_TO_STOP:
                sticky_box = None
                sticky_count = 0
                target = None

        for box in people:
            x1, y1, x2, y2 = box
        color = (0, 0, 255) if target is not None and box == target else (180, 180, 180)
        thickness = 3 if color == (0, 0, 255) else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        status = "IDLE"
        if target is not None:
            ax, ay = chest_point(target)
            cv2.drawMarker(frame, (ax, ay), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.circle(frame, (ax, ay), 8, (0, 0, 255), 2)
            cv2.putText(
                frame,
                "AIM CHEST",
                (ax + 12, ay - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            status = "FIRE" if armed else "TRACK (disarmed)"


        else:
            status = "STOP" if not people else "IDLE"

        # Default color logic
        color_status = (0, 0, 255) if "FIRE" in status else (0, 255, 255)

        # Override for STOP / IDLE
        if status in ("STOP", "IDLE"):
            color_status = (200, 200, 200)


        cv2.putText(
            frame,
            f"{status}    armed={armed}    people={len(people)}",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color_status,
            2,
        )
        cv2.putText(
            frame,
            "A=arm/disarm, Q=quit red box=closest   crosschair=chest",
            (16, frame.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
        )

        if status != last_print:
                print(time.strftime("%H:%M:%S"), status, f"people={len(people)}")
                last_print = status

        cv2.imshow("ProtoSmith room turret - body track", frame)
        key = cv2.waitKey(1) & 0xFF

            # Quit
        if key in (ord("q"), ord("Q"), 27):
                break

        # Toggle armed
        if key in (ord("a"), ord("A")):
            armed = not armed
            print("ARMED" if armed else "DISARMED")

    cap.release()
cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
