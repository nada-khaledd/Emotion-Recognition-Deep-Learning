import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model("emotion_model_advancedddd2.h5")
emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
IMG_SIZE = 96

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

st.set_page_config(page_title="Emotion Recognition", layout="centered")
st.title("😃 Emotion Recognition App")
st.write("Detect emotions using webcam or uploaded images")

tab1, tab2 = st.tabs(["📸 Webcam", "🖼 Upload Image"])

with tab1:
    run = st.checkbox("▶️ Start Webcam")
    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to open webcam")
                break

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_classifier.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5
            )

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]

                try:
                    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = face / 255.0
                    face = np.expand_dims(face, axis=0)

                    preds = model.predict(face, verbose=0)
                    emotion = emotion_labels[np.argmax(preds)]
                    confidence = np.max(preds)

                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                    cv2.putText(
                        frame,
                        f"{emotion} ({confidence*100:.1f}%)",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,0),
                        2
                    )
                except:
                    pass

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()

#  UPLOAD IMAGE TAB

with tab2:
    uploaded_file = st.file_uploader(
        "Upload a face image", type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces = face_classifier.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5
        )

        if len(faces) == 0:
            st.warning("No face detected 😕")
        else:
            for (x, y, w, h) in faces:
                face = img_array[y:y+h, x:x+w]

                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                face = face / 255.0
                face = np.expand_dims(face, axis=0)

                preds = model.predict(face, verbose=0)
                emotion = emotion_labels[np.argmax(preds)]
                confidence = np.max(preds)

                cv2.rectangle(img_array, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(
                    img_array,
                    f"{emotion} ({confidence*100:.1f}%)",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2
                )

            st.image(img_array, caption="Detected Emotion", use_column_width=True)
            st.success(f"Emotion: {emotion} ({confidence*100:.1f}%)")
