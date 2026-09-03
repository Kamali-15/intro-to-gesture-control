import cv2, mediapipe as mp, numpy as np
import subprocess

# --- macOS-specific functions ---
def set_volume(level):  # 0–100
    subprocess.run(["osascript", "-e", f"set volume output volume {int(level)}"])

def set_brightness(level):  # 0–100 mapped to 0–1.0
    subprocess.run(["brightness", str(level/100.0)])

# --- Mediapipe setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

TH, IX = mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP

# --- Camera setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not Found")
    exit()

win = "Hand gesture control"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)

while True:
    ok, img = cap.read()
    if not ok: break
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(imgRGB)

    if res.multi_hand_landmarks and res.multi_handedness:
        for idx, hand in enumerate(res.multi_hand_landmarks):
            hand_label = res.multi_handedness[idx].classification[0].label
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

            thumb = hand.landmark[TH]
            index = hand.landmark[IX]

            tx, ty = int(thumb.x * w), int(thumb.y * h)
            ix, iy = int(index.x * w), int(index.y * h)

            cv2.circle(img, (tx, ty), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (ix, iy), 10, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (tx, ty), (ix, iy), (255, 0, 0), 3)

            length = np.hypot(ix - tx, iy - ty)

            if hand_label == "Right":  # Right hand controls volume
                vol = np.interp(length, [20, 200], [0, 100])
                set_volume(vol)

                vol_bar = np.interp(length, [20, 200], [400, 150])
                cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
                cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, "Volume", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            else:  # Left hand controls brightness
                bright = np.interp(length, [20, 200], [0, 100])
                try:
                    set_brightness(bright)
                except Exception:
                    pass

                bright_bar = np.interp(length, [20, 200], [400, 150])
                cv2.rectangle(img, (w - 120, 150), (w - 85, 400), (255, 0, 0), 3)
                cv2.rectangle(img, (w - 120, int(bright_bar)), (w - 85, 400), (255, 0, 0), cv2.FILLED)
                cv2.putText(img, "Brightness", (w - 150, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow(win, img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
