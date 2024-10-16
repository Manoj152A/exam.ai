import os
import cv2
import numpy as np
import logging
import threading
import time
import base64
import dlib
import face_recognition
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Initialize dlib's face detector
face_detector = dlib.get_frontal_face_detector()

# Global variables
camera = None
output_file = None
out = None
is_exam_active = False
camera_lock = threading.Lock()
alerts = []
reference_encoding = None
verification_interval = 5  # Verify every 5 seconds
candidate_name = ""
last_face_detection_time = time.time()
consecutive_different_person = 0
CONSECUTIVE_THRESHOLD = 3

TARGET_FPS = 10
FACE_DETECTION_TIMEOUT = 3  # Seconds before "No face detected" alert

def detect_faces(frame):
    global last_face_detection_time
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    
    if len(faces) == 0:
        if time.time() - last_face_detection_time > FACE_DETECTION_TIMEOUT:
            alerts.append("No face detected")
        return frame, 0
    
    if len(faces) > 1:
        alerts.append("Multiple faces detected")
    
    last_face_detection_time = time.time()
    
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return frame, len(faces)

def get_face_encoding(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    
    if not face_locations:
        return None
    
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    if face_encodings:
        return face_encodings[0]
    return None

def verify_face(current_encoding, reference_encoding, threshold=0.6):
    if reference_encoding is None or current_encoding is None:
        return False
    
    distance = face_recognition.face_distance([reference_encoding], current_encoding)[0]
    logging.debug(f"Face verification distance: {distance}")
    return distance < threshold

def generate_frames():
    global camera, out, is_exam_active, output_file, reference_encoding, consecutive_different_person
    
    with camera_lock:
        if not is_exam_active:
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            
            output_folder = r"C:\Users\manoj\Videos\ai_exam"
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f"exam_recording_{time.strftime('%Y%m%d_%H%M%S')}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_file, fourcc, TARGET_FPS, (640, 480))
            
            is_exam_active = True
            logging.info(f"Exam proctoring started. Recording to {output_file}")
    
    start_time = time.time()
    frame_count = 0
    last_verification_time = time.time()
    
    while is_exam_active:
        success, frame = camera.read()
        if not success:
            logging.error("Failed to read frame from camera")
            break
        
        frame, face_count = detect_faces(frame)
        
        current_time = time.time()
        if face_count == 1:
            if current_time - last_verification_time >= verification_interval:
                current_encoding = get_face_encoding(frame)
                if current_encoding is not None:
                    if not verify_face(current_encoding, reference_encoding):
                        consecutive_different_person += 1
                        if consecutive_different_person >= CONSECUTIVE_THRESHOLD:
                            alerts.append("Different person detected")
                            consecutive_different_person = 0
                    else:
                        consecutive_different_person = 0
                    last_verification_time = current_time
        
        out.write(frame)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        frame_count += 1
        time.sleep(max(1./TARGET_FPS - (time.time() - start_time), 0))
        start_time = time.time()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['GET', 'POST'])
def capture():
    global reference_encoding, candidate_name
    if request.method == 'POST':
        image_data = request.form['image']
        candidate_name = request.form['name']
        
        # Remove the data URL prefix
        image_data = image_data.split(',')[1]
        
        # Decode the base64 image
        image_array = np.frombuffer(base64.b64decode(image_data), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        # Save the captured image
        os.makedirs('captures', exist_ok=True)
        cv2.imwrite(f'captures/{candidate_name}.jpg', image)
        
        # Get the face encoding
        reference_encoding = get_face_encoding(image)
        
        if reference_encoding is not None:
            return redirect(url_for('exam'))
        else:
            return "Failed to capture face. Please try again.", 400
    
    return render_template('capture.html')

@app.route('/exam')
def exam():
    global candidate_name
    return render_template('exam.html', candidate_name=candidate_name)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_exam')
def start_exam():
    global is_exam_active
    is_exam_active = True
    alerts.clear()
    threading.Thread(target=generate_frames, daemon=True).start()
    return jsonify({"status": "Exam proctoring started"})

@app.route('/end_exam')
def end_exam():
    global is_exam_active, camera, out
    is_exam_active = False
    if camera:
        camera.release()
    if out:
        out.release()
    return jsonify({"status": "Exam ended", "alerts": alerts})

@app.route('/get_alerts')
def get_alerts():
    return jsonify({"alerts": alerts})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)