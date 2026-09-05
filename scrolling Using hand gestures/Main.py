import cv2,time,pyautogui
import mediapipe as mp

mp_hands=mp.solutions.hands
hands=mp_hands.Hands(max_num_hands=1,min_detection_confidence=0.7)
mp_drawing=mp.solutions.drawing_utils

SCROLL_SPEED=5
SCROLL_DELAY=5
CAM_WIDTH,CAM_HEIGHT=640,480

def detect_gestures(landmarks,handedness):
    fingers=[]
    tips=[mp_hands.HandLandmark.INDEX_FINGER_TIP,mp_hands.HandLandmark.MIDDLE_FINGER_TIP,mp_hands.HandLandmark.RING_FINGER_TIP,mp_hands.HandLandmark.PINKY_TIP]

    for tip in tips:
        if landmarks.landmark[tip].y<landmarks.landmark[tip-2].y:
            fingers.append(1)

    thumb_tip=landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip=landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]

    if(handedness=='Right' and thumb_tip.x>thumb_ip.x) or (handedness =='left' and thumb_tip.x<thumb_ip.x):
        fingers.append(1)

    return "scroll_up" if sum(fingers)==4 else "scroll_down" if sum(fingers)==0 else "None"

cap=cv2.VideoCapture(0)
cap.set(3,CAM_WIDTH)
cap.set(4,CAM_HEIGHT)
last_scroll=p_time=0
print("Open palms:Scroll Up\nFist: Scroll down\Press 'Q' to quit")

while cap.isOpened():
    success,img=cap.read()
    if not success:
        break
    img=cv2.flip(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),1)
    results=hands.process(img)
    gesture,handedness="none","Unknown"

    if results.multi_hand_landmarks:
        for hand,handedness_info in zip(results.multi_hand_landmarks,results.multi_handedness):
            mp_drawing.draw_landmarks(img,hand,mp_hands.HAND_CONNECTIONS)

            handedness=handedness_info.classification[0].label
            gesture=detect_gestures(hand,handedness)
            mp_drawing.draw_landmarks(img,hand,mp_hands.HAND_CONNECTIONS)

            if(time.time()- last_scroll)> SCROLL_DELAY:
                if gesture=="scroll_up":
                    pyautogui.scroll(SCROLL_SPEED)
                elif gesture=="scroll_down":
                    pyautogui.scroll(-SCROLL_SPEED)
                    last_scroll=time.time()

    fps=1/(time.time()-p_time) if (time.time()-p_time)!=0 else 0
    p_time=time.time()

    cv2.putText(img,f"FPS:{int(fps)} Hand: {handedness}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
    cv2.putText(img,f"Gesture: {gesture}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Scrolling",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()