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
    exit(1)

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_file)
logging.info("Dlib models loaded successfully")

# Global variables
camera = None
output_file = None
out = None
is_camera_active = False
camera_lock = threading.Lock()

# Set a lower target FPS to ensure all frames are processed
TARGET_FPS = 10

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

def generate_frames():
    global camera, out, is_camera_active, output_file
    
    with camera_lock:
        if not is_camera_active:
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            
            output_folder = r"C:\Users\manoj\Videos\ai"
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f"ai_recording_{time.strftime('%Y%m%d_%H%M%S')}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_file, fourcc, TARGET_FPS, (640, 480))
            
            is_camera_active = True
            logging.info("Camera and recording started")
    
    try:
        start_time = time.time()
        frame_count = 0
        while True:
            frame_start = time.time()
            
            success, frame = camera.read()
            if not success:
                logging.error("Failed to grab frame")
                break
            
            frame = detect_faces(frame)
            
            # Calculate and display FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write frame to video file
            out.write(frame)
            
            # Encode frame for streaming
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                logging.error("Failed to encode frame")
                continue
            
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            # Calculate the time to sleep to maintain the target FPS
            processing_time = time.time() - frame_start
            sleep_time = max(1.0/TARGET_FPS - processing_time, 0)
            time.sleep(sleep_time)
            
            # Log actual frame processing time
            actual_frame_time = time.time() - frame_start
            logging.debug(f"Frame {frame_count}: Processing time = {processing_time:.4f}s, Sleep time = {sleep_time:.4f}s, Total frame time = {actual_frame_time:.4f}s")
            
            # Check if we're maintaining real-time
            if frame_count % TARGET_FPS == 0:
                expected_time = frame_count / TARGET_FPS
                actual_time = elapsed_time
                logging.info(f"After {frame_count} frames: Expected time = {expected_time:.2f}s, Actual time = {actual_time:.2f}s, Difference = {actual_time - expected_time:.2f}s")
    finally:
        with camera_lock:
            if is_camera_active:
                camera.release()
                out.release()
                is_camera_active = False
                logging.info(f"Camera stopped and recording saved to {output_file}")
                logging.info(f"Total frames: {frame_count}, Total time: {elapsed_time:.2f}s, Average FPS: {frame_count/elapsed_time:.2f}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    logging.info("Video feed requested")
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)