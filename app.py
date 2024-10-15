from flask import Flask, render_template, Response
import cv2
import dlib
import numpy as np
import logging
import threading
import time
import os

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Check for shape predictor file in the models folder
shape_predictor_file = os.path.join("models", "shape_predictor_68_face_landmarks.dat")
if not os.path.isfile(shape_predictor_file):
    print(f"Error: {shape_predictor_file} not found.")
    print("Please ensure the file is in the 'models' folder.")
    print("If not, download it from:")
    print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    print("Extract the file and place it in the 'models' folder.")
    exit(1)

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_file)
logging.info("Dlib models loaded successfully")

# Global variables for thread-safe frame sharing
global_frame = None
frame_lock = threading.Lock()

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Get facial landmarks
        shape = predictor(gray, face)
        for i in range(68):
            x, y = shape.part(i).x, shape.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
    
    return frame

def capture_frames():
    global global_frame
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    skip_frames = 2
    last_detection_time = time.time()
    fps = 0
    
    while True:
        success, frame = camera.read()
        if not success:
            logging.error("Failed to grab frame")
            time.sleep(0.1)
            continue
        
        frame_count += 1
        
        if frame_count % skip_frames == 0:
            frame = detect_faces(frame)
            last_detection_time = time.time()
        
        # Calculate and display FPS
        current_time = time.time()
        if current_time - last_detection_time >= 1.0:
            fps = frame_count / (current_time - last_detection_time)
            frame_count = 0
            last_detection_time = current_time
        
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        with frame_lock:
            global_frame = frame.copy()
        
        # Adaptive frame skipping
        processing_time = time.time() - last_detection_time
        if processing_time > 0.033:  # If processing takes more than 33ms (30 fps)
            skip_frames = min(skip_frames + 1, 5)  # Increase skip frames, max 5
        elif processing_time < 0.02:  # If processing takes less than 20ms
            skip_frames = max(skip_frames - 1, 1)  # Decrease skip frames, min 1
        
        time.sleep(0.001)  # Small delay to prevent CPU overuse

def generate_frames():
    global global_frame
    while True:
        with frame_lock:
            if global_frame is None:
                continue
            
            frame = global_frame.copy()
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    logging.info("Video feed requested")
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    threading.Thread(target=capture_frames, daemon=True).start()
    app.run(debug=True, threaded=True)