from flask import Flask, jsonify, request
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import base64

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins='*', logger=True, engineio_logger=True)

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

@socketio.on('video_feed_from_client')
def handle_video_feed(data):
    # Convert the data (dataURL) back to a NumPy array (OpenCV frame)
    image_data = np.frombuffer(base64.b64decode(data.split(',')[1]), np.uint8)
    frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    # Perform face detection and emotion recognition on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]

            # Emit the emotion label back to the client
            emit('emotion', label)

if __name__ == '__main__':
    socketio.run(app, debug=True)
