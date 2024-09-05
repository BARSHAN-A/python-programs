import cv2
from deepface import DeepFace


cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    try:
        analyze = DeepFace.analyze(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), actions=['emotion'])
        emotion_dict = analyze[0]['emotion'] 
        dominant_emotion = max(emotion_dict, key=emotion_dict.get)

        cv2.putText(frame, dominant_emotion, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    except Exception as e:
        print(f"Error: {str(e)}")

    cv2.imshow('Emotion Analysis', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()