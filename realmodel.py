import cv2
import numpy as np
from pathlib import Path
from keras.models import model_from_json

project_dir = Path(__file__).resolve().parent
with (project_dir / "detection.json").open(encoding="utf-8") as json_file:
    model = model_from_json(json_file.read())
model.load_weights(str(project_dir / "detection.weights.h5"))

haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)
if face_cascade.empty():
    raise RuntimeError(f"Could not load the face detector: {haar_file}")

def extract_feature(image):
    feature = np.asarray(image).reshape(1, 32, 32, 1)
    return feature / 255.0

labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not webcam.isOpened():
    raise RuntimeError(
        "Could not open webcam 0. Check Windows camera permissions and close any "
        "other app that may be using the camera."
    )

try:
    while True:
        ok, frame = webcam.read()
        if not ok or frame is None:
            raise RuntimeError("The webcam opened but did not return a video frame.")

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            image = gray[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            img_gray = cv2.resize(image, (32, 32))
            prediction = model.predict(extract_feature(img_gray), verbose=0)
            label = labels[int(np.argmax(prediction))]
            cv2.putText(
                frame, label, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
            )

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    webcam.release()
    cv2.destroyAllWindows()