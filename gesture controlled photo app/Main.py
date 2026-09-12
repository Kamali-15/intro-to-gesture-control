from cv2 import cuda
import cv2, time, numpy as np
# pyrefly: ignore [missing-import]
import mediapipe as mp

mp_hands= mp.solutions.hands
hands=mp_hands.Hands(min_detection_confidence=0.7,min_tracking_confidence=0.7)
mp_draw=mp.solutions.drawing_utils

filters={
    "ORIGINAL": lambda img: img,
    "NEGATIVE":lambda img: cv2.bitwise_not(img),
    "GLITCH":lambda img:cv2.merge([
        np.roll(img[:,:,0],-20,1),
        img[:,:,1],
        np.roll(img[:,:,2], 40, 1)
    ]) 
}

pairs={"middle":["NEGATIVE", "ORIGINAL"],"ring":["GLITCH", "ORIGINAL"],"pinky":["GLITCH", "NEGATIVE"]}
cur_filter,states="ORIGINAL",  {k:0 for k in pairs}
DEB,CAP,TP=0.6, 1.2, 20
last_action,last_capture,pinch_on=0,0,False

cap=cv2.VideoCapture(0)
if not cap.isOpened():
    exit("Error: Web cam not found")

while True:
    ok,img=cap.read()
    if not ok:
        break
    img=cv2.flip(img,1)
    h,w=img.shape[:2]
    res=hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    now=time.time()
    capture=False
    if res.multi_hand_landmarks:
        hand= res.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img,hand,mp_hands.HAND_CONNECTIONS)
        lm=hand.landmark
        tip_map = {"middle": mp_hands.HandLandmark.MIDDLE_FINGER_TIP, "ring": mp_hands.HandLandmark.RING_FINGER_TIP, "pinky": mp_hands.HandLandmark.PINKY_TIP}
        
        def get_dist(lm1, lm2):
            return ((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2)**0.5
            
        thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        if get_dist(thumb_tip, index_tip) < 0.06:
            if not pinch_on and now-last_capture>CAP:
                capture,pinch_on,last_capture=True,True,now
        else:
            pinch_on=False
            for f in pairs:
                if get_dist(thumb_tip, lm[tip_map[f]]) < 0.06 and now-last_action>DEB:
                    cur_filter=pairs[f][states[f]]
                    states[f]^=1
                    last_action=now
                    print("Filter: ",cur_filter)
    out=filters[cur_filter](img)
    if capture:
        name=f"picture_{int(now)}.jpg"
        cv2.imwrite(name,out)
        print(f"SAVED: {name}" )

    cv2.imshow("Gesture Photo App", out)
    if cv2.waitKey(1) & 0xFF == ord("q"): break
cap.release()
cv2.destroyAllWindows()