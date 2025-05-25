from flask import Flask, render_template, request, jsonify, Response
from face_auth import FaceAuth
from db_utils import MongoDBUtils
import cv2
import threading
import time
import base64
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Global variables for camera and authentication
camera = None
camera_lock = threading.Lock()
auth_in_progress = False
current_face_auth = None
auth_status = {
    'cue': None,
    'attempt': 1,
    'status': 'Ready',
    'active': False,
    'max_attempts': 3
}

# MongoDB connection URI
MONGODB_URI = os.getenv('MONGODB_URI')

class CameraManager:
    def __init__(self):
        self.camera = None
        self.is_active = False
        self.lock = threading.Lock()
        
    def start_camera(self):
        with self.lock:
            if not self.is_active:
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.is_active = True
                time.sleep(1)  # Give camera time to warm up
                
    def stop_camera(self):
        with self.lock:
            if self.is_active and self.camera:
                self.camera.release()
                self.camera = None
                self.is_active = False
                
    def get_frame(self):
        with self.lock:
            if self.is_active and self.camera:
                ret, frame = self.camera.read()
                if ret:
                    return frame
        return None

# Global camera manager
camera_manager = CameraManager()

def generate_frames():
    """Generate video frames for streaming"""
    while True:
        frame = camera_manager.get_frame()
        if frame is not None:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # If no frame available, wait a bit
            time.sleep(0.1)

class EnhancedFaceAuth(FaceAuth):
    """Enhanced FaceAuth class that works with the camera manager"""
    
    def __init__(self, db_utils, target_name, camera_manager):
        # Initialize parent class but don't load models yet
        self.db_utils = db_utils
        self.known_encoding = []
        self.known_names = []
        self.authorized_users = {}
        self.camera_manager = camera_manager
        
        # Load known faces
        self.load_known_faces(target_name)
        
        # Initialize models
        self.initialize_models()
        
        # Thresholds
        self.spoof_thresh = 0.4
        self.frame_history = 20
        self.spoof_confidence = 0.5
        
        # Status tracking
        self.current_cue = None
        self.current_attempt = 1
        
    def update_status(self, status, active=False, cue=None, attempt=None):
        """Update global status for UI"""
        global auth_status
        auth_status['status'] = status
        auth_status['active'] = active
        if cue:
            auth_status['cue'] = cue
            self.current_cue = cue
        if attempt:
            auth_status['attempt'] = attempt
            self.current_attempt = attempt
        
    def initialize_models(self):
        """Initialize the ML models"""
        try:
            from tensorflow.keras.models import load_model
            from tensorflow.keras.layers import DepthwiseConv2D
            
            anti_spoof_model = "models/antispoofing_full_model.h5"
            self.face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

            def custom_depthwise_conv2d(**kwargs):
                kwargs.pop('groups', None)
                return DepthwiseConv2D(**kwargs)
            
            self.anti_spoof_model = load_model(anti_spoof_model, custom_objects={
                'DepthwiseConv2D': custom_depthwise_conv2d})
            
            print("Models loaded successfully (haarcascade, spoofer)")
        except Exception as e:
            print(f"Error loading models: {e}")
            # Use fallback without anti-spoofing if models fail to load
            self.anti_spoof_model = None
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def live_auth_with_streaming(self, target_name, tolerance=0.45, max_attempts=1):
        """Modified live auth that works with the streaming camera with 3 attempts"""
        import face_recognition as fr
        from collections import deque
        import uuid
        
        self.update_status("Starting authentication...", True)
        
        # Initialize cue verification if available
        try:
            from cue_verification import CueVerification
            cue_verification = CueVerification()
            self.update_status("Cue verification loaded", True)
        except ImportError:
            print("CueVerification not available, skipping cue verification")
            cue_verification = None
            self.update_status("Cue verification not available", True)
        
        # Try authentication up to max_attempts times
        for attempt in range(1, max_attempts + 1):
            self.update_status(f"Attempt {attempt} of {max_attempts}", True, attempt=attempt)
            
            result = self._single_auth_attempt(target_name, cue_verification, tolerance, attempt)
            
            if result["success"]:
                return result
            elif result.get("spoof_detected"):
                self.update_status("Spoof detected - trying again", False, cue="Spoof detected! Please try again")
                time.sleep(2)
                continue
            else:
                if attempt < max_attempts:
                    self.update_status(f"Attempt {attempt} failed - trying again", False, cue="Authentication failed, please try again")
                    time.sleep(2)
                else:
                    self.update_status("All attempts failed", False)
        
        return {"success": False, "message": f"Authentication failed after {max_attempts} attempts"}
    
    def _single_auth_attempt(self, target_name, cue_verification, tolerance, attempt_num):
        """Single authentication attempt"""
        import face_recognition as fr
        from collections import deque
        import uuid
        
        # Auth rules for single attempt
        max_frames = 200  # Reduced frames per attempt
        frame_count = 0
        cue_completed = cue_verification is None
        spoof_history = deque(maxlen=self.frame_history)
        consecutive_matches = 0
        required_matches = 3  # Reduced for faster auth
        
        self.update_status(f"Attempt {attempt_num}: Looking for face...", True, cue="Position your face in the camera")
        
        while frame_count < max_frames:
            # Get frame from camera manager
            frame = self.camera_manager.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
                
            frame_count += 1
            
            # Convert frame to RGB for face recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = fr.face_locations(rgb_frame)
            
            if face_locations:
                self.update_status(f"Attempt {attempt_num}: Face detected", True, cue="Face detected - stay still")
                
                # 1. Check spoof if model is available
                if self.anti_spoof_model is not None:
                    is_real = self.check_spoof(frame)
                    if is_real is not None:
                        spoof_history.append(is_real)
                        
                        # Check spoof ratio if we have enough frames
                        if len(spoof_history) >= min(10, self.frame_history):
                            real_count = sum(spoof_history)
                            spoof_ratio = real_count / len(spoof_history)
                            
                            if spoof_ratio < self.spoof_confidence:
                                self.update_status(f"Attempt {attempt_num}: Spoof detected", False, cue="Spoof detected!")
                                return {"success": False, "spoof_detected": True, "message": "Spoof detection failed"}
                
                # 2. Cue Verification (if available)
                if not cue_completed and cue_verification:
                    try:
                        cue_in_progress, current_cue = cue_verification.run_verification(frame)
                        if current_cue and current_cue != self.current_cue:
                            self.update_status(f"Attempt {attempt_num}: Follow the cue", True, cue=current_cue)
                        
                        if not cue_in_progress:
                            cue_completed = True
                            self.update_status(f"Attempt {attempt_num}: Cue verification complete", True, cue="Cue verification complete")
                    except Exception as e:
                        print(f"Cue verification error: {e}")
                        cue_completed = True
                
                if cue_completed:
                    # 3. Face Recognition
                    self.update_status(f"Attempt {attempt_num}: Verifying identity...", True, cue="Verifying your identity...")
                    face_encodings = fr.face_encodings(rgb_frame, face_locations)
                    
                    for face_encoding in face_encodings:
                        if target_name in self.authorized_users:
                            matches = fr.compare_faces(
                                [self.authorized_users[target_name]['encoding']],
                                face_encoding,
                                tolerance=tolerance
                            )
                            
                            if matches[0]:
                                consecutive_matches += 1
                                self.update_status(f"Attempt {attempt_num}: Match found ({consecutive_matches}/{required_matches})", True, 
                                                 cue=f"Identity match! ({consecutive_matches}/{required_matches})")
                                
                                if consecutive_matches >= required_matches:
                                    # Save the frame as an image
                                    if not os.path.exists("static"):
                                        os.makedirs("static")
                                    
                                    image_filename = f"{uuid.uuid4().hex}.jpg"
                                    image_path = os.path.join("static", image_filename)
                                    cv2.imwrite(image_path, frame)
                                    
                                    self.update_status("Authentication successful!", True, cue="Access granted!")
                                    
                                    return {
                                        "success": True, 
                                        "message": f"Access granted for {target_name}!",
                                        "image_path": f"/static/{image_filename}"
                                    }
                            else:
                                consecutive_matches = max(0, consecutive_matches - 1)
                                if consecutive_matches == 0:
                                    self.update_status(f"Attempt {attempt_num}: Identity mismatch", True, cue="Identity not recognized")
            else:
                if frame_count % 20 == 0:  # Update every 20 frames to avoid spam
                    self.update_status(f"Attempt {attempt_num}: No face detected", True, cue="Please position your face in the camera")
            
            time.sleep(0.05)  # Small delay
        
        return {"success": False, "message": f"Attempt {attempt_num} timeout"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    camera_manager.start_camera()
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status():
    """Get current authentication status and cues"""
    global auth_status
    return jsonify(auth_status)

def reset_auth_state():
    """Reset authentication state variables"""
    global auth_in_progress, current_face_auth, auth_status
    auth_in_progress = False
    current_face_auth = None
    auth_status = {
        'cue': None,
        'attempt': 1,
        'status': 'Ready',
        'active': False,
        'max_attempts': 3
    }

@app.route('/authenticate', methods=['POST'])
def authenticate():
    global auth_in_progress, current_face_auth, auth_status
    
    if auth_in_progress:
        return jsonify({"success": False, "message": "Authentication already in progress"}), 400
    
    name = request.form.get('name')
    if not name:
        return jsonify({"success": False, "message": "Name is required!"}), 400

    try:
        # Reset status
        auth_status = {
            'cue': 'Starting authentication...',
            'attempt': 1,
            'status': 'Initializing...',
            'active': True,
            'max_attempts': 3
        }
        
        # Set authentication in progress
        auth_in_progress = True
        
        # Start camera if not already started
        camera_manager.start_camera()
        
        # Initialize MongoDBUtils and Enhanced FaceAuth
        db_utils = MongoDBUtils(MONGODB_URI)
        current_face_auth = EnhancedFaceAuth(db_utils, name, camera_manager)
        
        # Start live authentication with streaming (3 attempts)
        result = current_face_auth.live_auth_with_streaming(name)
        
        # Final status update
        if result["success"]:
            auth_status['status'] = 'Authentication successful!'
            auth_status['cue'] = 'Access granted!'
            auth_status['active'] = True
        else:
            auth_status['status'] = 'Authentication failed'
            auth_status['cue'] = result.get('message', 'Access denied')
            auth_status['active'] = False
        
        return jsonify(result)
        
    except Exception as e:
        auth_status['status'] = 'Authentication error'
        auth_status['cue'] = str(e)
        auth_status['active'] = False
        return jsonify({"success": False, "message": f"Authentication error: {str(e)}"}), 500
        
    finally:
        # Reset authentication state after delay
        def reset_auth():
            reset_auth_state()
            camera_manager.stop_camera()
        
        # Schedule the reset after 3 seconds to allow UI to display results
        threading.Timer(3.0, reset_auth).start()

@app.route('/cancel_auth', methods=['POST'])
def cancel_auth():
    """Cancel ongoing authentication"""
    global auth_in_progress, current_face_auth, auth_status
    
    # Reset all authentication state
    reset_auth_state()
    
    # Stop camera immediately
    camera_manager.stop_camera()
    
    # Update status for cancelled state
    auth_status['status'] = 'Authentication cancelled'
    auth_status['cue'] = 'Authentication cancelled by user'
    auth_status['active'] = False
    
    return jsonify({"success": True, "message": "Authentication cancelled"})

# Serve static files
@app.route('/static/<filename>')
def static_files(filename):
    return app.send_static_file(filename)

if __name__ == '__main__':
    # Ensure static directory exists
    if not os.path.exists("static"):
        os.makedirs("static")
    
    app.run(debug=True, threaded=True)