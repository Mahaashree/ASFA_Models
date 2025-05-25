
#WELCOME TO FACE_AUTH
#INTEGRATING V6 AND SPOOFER - SUCCESSFUL


import cv2
import face_recognition as fr
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from collections import deque
from cue_verification import CueVerification  # Adjust the import path as needed

class FaceAuth:
    def __init__(self, db_utils, target_name):
        self.db_utils = db_utils
        self.known_encoding =[]
        self.known_names = []
        self.authorized_users = {}

        #Loading known faces
        self.load_known_faces(target_name)
        anti_spoof_model = "models/antispoofing_full_model.h5"

        #initialize Anti-spoof
        self.face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

        def custom_depthwise_conv2d(**kwargs):
            kwargs.pop('groups', None)
            return DepthwiseConv2D(**kwargs)
        
        #Loading anti spoof model
        self.anti_spoof_model = load_model(anti_spoof_model, custom_objects = {
            'DepthwiseConv2D': custom_depthwise_conv2d})
        
        print("Models loaded successfully (haarcascade, spoofer)")

        #THRESHOLDS
        self.spoof_thresh = 0.4 #below this is considered spoof
        self.frame_history = 20 #Number of frames we're considering
        self.spoof_confidence = 0.5 # allow max 80% of spoof frames


    def load_known_faces(self, target_name):

        user = self.db_utils.get_user_from_db(target_name)
        
        #name = user['name']

        if user:
            encoding = user['encoding']

            encoding_np = np.array(encoding)
            self.known_encoding.append(encoding_np)
            self.known_names.append(target_name)

            self.authorized_users[target_name] = {
                'encoding': encoding_np,
                'access_level': user['access_level']

            }
        
        else:
            print(f"{target_name} : User not found in database. Please register: ")
            img_path = input("enter img path: ")
            self.db_utils.register(target_name, img_path)
 

    def check_spoof (self, frame):
        #checks if frame has real or spoof 
        #returns true if real, false id spoof

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            print(f"No faces detected.")
            return 

        spoof_preds = deque(maxlen = self.frame_history)
        
        for (x, y, w, h) in faces:
            face = frame[y-5:y+h+5, x-5:x+w+5]
            resized_face = cv2.resize(face, (160, 160))
            resized_face = resized_face.astype("float") / 255.0
            resized_face = np.expand_dims(resized_face, axis=0)
            
            # Predict using the anti-spoofing model
            preds = self.anti_spoof_model.predict(resized_face)[0]
            bool_pred = preds < self.spoof_thresh #True if real image
            print(f"Full prediction: {preds} bool: {bool_pred}")
            #storing this pred
            spoof_preds.append(bool_pred)
            
        if not spoof_preds:
            return False
        
        #print(f"Spoof preds: {spoof_preds}")
        spoof_count = sum(spoof_preds)
        spoof_ratio = spoof_count / len(spoof_preds)

        print(f"Spoof ratio: {spoof_ratio}")
        

        return spoof_ratio > self.spoof_confidence
    
  
    import os
    import uuid

    def live_auth(self, target_name, tolerance=0.45):
        # Initialize cue verification
        cue_verification = CueVerification()

        # Opening camera
        cap = cv2.VideoCapture(0)
        cv2.waitKey(2)

        # Auth rules
        max_attempts = 50
        attempts = 0
        cue_completed = False
        spoof_history = deque(maxlen=self.frame_history)

        while attempts < max_attempts:
            # Capturing frame by frame
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB for face recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_loc = fr.face_locations(rgb_frame)

            if face_loc:
                # 1. Check spoof
                is_real = self.check_spoof(frame)
                spoof_history.append(is_real)

                # Checking spoof ratio if we have enough frames
                if len(spoof_history) == self.frame_history:
                    spoof_ratio = spoof_history.count(False) / len(spoof_history)
                    if spoof_ratio > self.spoof_confidence:
                        cap.release()
                        return False

                # 2. Cue Verification
                if not cue_completed:
                    cue_in_progress, current_cue = cue_verification.run_verification(frame)

                    # Check if cue verification is complete
                    if not cue_in_progress:
                        cue_completed = True

                if cue_completed:
                    # 3. Live Auth
                    face_encoding = fr.face_encodings(rgb_frame, face_loc)

                    for (top, right, bottom, left), face_encoding in zip(face_loc, face_encoding):
                        name = "unknown"
                        access_granted = False

                        if target_name in self.authorized_users:
                            matches = fr.compare_faces(
                                [self.authorized_users[target_name]['encoding']],
                                face_encoding,
                                tolerance=tolerance
                            )

                            if matches[0]:
                                name = target_name
                                access_granted = True

                                # Save the frame as an image
                                image_path = os.path.join("static", f"{uuid.uuid4().hex}.jpg")
                                cv2.imwrite(image_path, frame)

                                cap.release()
                                return {"success": True, "image_path": image_path}

            attempts += 1

        cap.release()
        return {"success": False}

        