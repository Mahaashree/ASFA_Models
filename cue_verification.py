import cv2
import numpy as np
import dlib
import random
import time

# Load face detection and facial landmark models
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

class CueVerification:
    def __init__(self):
        self.cues = [
            {"command": "Please look right", "duration": 10, "verified": False, "attempts": 0},
            {"command": "Please look left", "duration": 10, "verified": False, "attempts": 0},
            {"command": "Please blink slowly", "duration": 10, "verified": False, "attempts": 0},
            {"command": "Please smile", "duration": 10, "verified": False, "attempts": 0}

        ]

        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        self.current_cue_index = 0
        self.cue_start_time = None
        self.in_progress = False
        self.verification_threshold = 2  # Number of successful detections needed
        self.success_counter = 0
        self.max_attempts = 5  # Maximum attempts per cue
        self.debug_mode = True  # Enable debug information
        #elf.verified_cues_count = 0 # Track number of verified cues

        random.shuffle(self.cues)
    
    def start_verification(self):
        self.in_progress = True
        self.current_cue_index = 0
        self.cue_start_time = time.time()
        self.success_counter = 0
        print("\n--- Starting Cue Verification Test ---")
        print(f"First Cue: {self.get_current_cue()['command']}")
    
    def get_current_cue(self):
        if self.current_cue_index < len(self.cues):
            return self.cues[self.current_cue_index]
        return None

    def _get_mouth_aspect_ratio(self, mouth):
        """Calculate mouth aspect ratio and smile indicators"""
        # Get corners of mouth
        left_corner = mouth[0]
        right_corner = mouth[6]
        
        # Get top and bottom points of inner mouth
        top_inner = np.mean(mouth[2:4], axis=0)
        bottom_inner = np.mean(mouth[8:10], axis=0)
        
        # Get outer points
        top_outer = np.mean(mouth[3:5], axis=0)
        bottom_outer = np.mean(mouth[9:11], axis=0)
        
        # Calculate vertical and horizontal distances
        horizontal_dist = np.linalg.norm(left_corner - right_corner)
        vertical_dist = np.linalg.norm(top_inner - bottom_inner)
        
        # Calculate mouth corner elevation (smile raises corners)
        left_elevation = left_corner[1] - (top_outer[1] + bottom_outer[1]) / 2
        right_elevation = right_corner[1] - (top_outer[1] + bottom_outer[1]) / 2
        
        # Calculate ratios
        aspect_ratio = vertical_dist / horizontal_dist if horizontal_dist > 0 else 0
        corner_ratio = (left_elevation + right_elevation) / 2
        
        return aspect_ratio, corner_ratio

    def _get_eye_aspect_ratio(self, eye):
        """Calculate eye aspect ratio"""
        vertical_dist = np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])
        horizontal_dist = np.linalg.norm(eye[0] - eye[3]) * 2
        return vertical_dist / horizontal_dist if horizontal_dist > 0 else 0

    def verify_action(self, landmarks):
        """Verify facial action based on landmarks"""
        if not landmarks:
            return False

        current_cue = self.get_current_cue()
        if not current_cue:
            return False

        points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])
        
        # Get key facial features
        left_eye = points[36:42]
        right_eye = points[42:48]
        mouth = points[48:68]
        
        # Calculate face center
        face_center = np.mean(points, axis=0)
        
        verified = False
        print(f"\nTesting Cue: {current_cue['command']}")
        
        if current_cue["command"] == "Please look left":
            nose_tip = points[33]
            # Check if nose is left of face center
            if nose_tip[0] < face_center[0] - 10:
                verified = True
                print("✅ Left look detected!")
                
        elif current_cue["command"] == "Please look right":
            nose_tip = points[33]
            # Check if nose is right of face center
            if nose_tip[0] > face_center[0] + 10:
                verified = True
                print("✅ Right look detected!")
                
        elif current_cue["command"] == "Please blink slowly":
            left_ear = self._get_eye_aspect_ratio(left_eye)
            right_ear = self._get_eye_aspect_ratio(right_eye)
            # Check for closed eyes
            if left_ear < 0.3 and right_ear < 0.3:
                verified = True
                print("✅ Slow blink detected!")
                
        elif current_cue["command"] == "Please smile":
            # Get both aspect ratio and corner elevation
            aspect_ratio, corner_ratio = self._get_mouth_aspect_ratio(mouth)
            
            # Calculate mouth width change
            mouth_width = np.linalg.norm(mouth[0] - mouth[6])
            neutral_width = np.linalg.norm(points[0] - points[16])
            width_ratio = mouth_width / neutral_width if neutral_width > 0 else 0
            
            # Check smile conditions
            if (corner_ratio < -3  # Mouth corners lowered
                and width_ratio > 0.35  # Mouth width increased
                and aspect_ratio < 0.35):  # Mouth flattened
                verified = True
                print("✅ Smile detected!")
            
            # Debug information for smile detection
            print(f"   Smile metrics:")
            print(f"   - Corner ratio: {corner_ratio:.2f}")
            print(f"   - Width ratio: {width_ratio:.2f}")
            print(f"   - Aspect ratio: {aspect_ratio:.2f}")

        if verified:
            self.success_counter += 1
            print(f"   Success counter: {self.success_counter}/{self.verification_threshold}")
        
        return verified

    def next_cue(self):
        """Move to next cue or end verification"""
        current_cue = self.get_current_cue()
        
        # Check if time has exceeded cue duration
        if time.time() - self.cue_start_time > current_cue["duration"]:
            print(f"❌ Time limit exceeded for cue: {current_cue['command']}")
            return False
    
        if self.success_counter >= self.verification_threshold:
            current_cue["verified"] = True
            self.success_counter = 0
            self.current_cue_index += 1
            
            if self.current_cue_index < len(self.cues):
                self.cue_start_time = time.time()
                print(f"\n--- Moving to next cue: {self.get_current_cue()['command']} ---")
                return True
            else:
                self.in_progress = False
                print("\n✨ All cues verified successfully! ✨")
                return False
        
        """current_cue["attempts"] += 1
        if current_cue["attempts"] >= self.max_attempts:
            print(f"❌ Failed to verify cue after {self.max_attempts} attempts")
            return False"""
            
        return True
    

    def run_verification(self, frame):
        #Runs cue ver on a single frame
        #return: bool(T if complete), str or None: current cue command in progress

        if not self.in_progress:
            self.start_verification()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector(gray)
        if not faces:
            return self.in_progress, self.get_current_cue()['command'] if self.get_current_cue() else None
        
        for face in faces:
            landmarks = self.shape_predictor(gray, face)

            if self.verify_action(landmarks):
                self.success_counter +=1

                if not self.next_cue():
                    return False, None
        
        return self.in_progress, self.get_current_cue()['command'] if self.get_current_cue() else None


"""
def test_cue_verification():
    # Initialize verification system
    verification = CueVerification()
    
    # Start the verification process
    verification.start_verification()
    
    # Capture video
    video = cv2.VideoCapture(0)
    
    if not video.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("\nInstructions:")
    print("1. Follow the prompts on screen")
    print("2. Perform the actions when asked")
    print("3. Press 'q' to quit\n")
    
    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Current cue details
        current_cue = verification.get_current_cue()
        if current_cue:
            # Show current cue and time left
            time_left = max(0, current_cue["duration"] - (time.time() - verification.cue_start_time))
            cv2.putText(frame, 
                        f"{current_cue['command']} ({int(time_left)}s)", 
                        (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 255), 
                        2)
        
        # Detect faces
        faces = detector(gray)
        for face in faces:
            # Detect landmarks
            landmarks = shape_predictor(gray, face)
            
            # Verify action
            if verification.verify_action(landmarks):
                if not verification.next_cue():
                    print("Verification complete!")
                    video.release()
                    cv2.destroyAllWindows()
                    return
        
        # Show the frame
        cv2.imshow('Cue Verification Test', frame)
        
        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    video.release()
    cv2.destroyAllWindows()

# Run the test
if __name__ == "__main__":
    test_cue_verification() """