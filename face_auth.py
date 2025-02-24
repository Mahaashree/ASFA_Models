
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
    
  
    def live_auth(self, target_name, tolerance=0.45):
    # Initialize cue verification
        cue_verification = CueVerification()

        # Opening camera
        cap = cv2.VideoCapture(0)
        cv2.waitKey(2)

        # Auth rules
        max_attempts = 70
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
                #1. check spoof

                is_real = self.check_spoof(frame)
                spoof_history.append(is_real)

                #checking spoof ratio if we have enough frames
                if len(spoof_history) == self.frame_history:
                    spoof_ratio = spoof_history.count(False) / len(spoof_history)
                    print(f"Spoof ratio in live_auth: {spoof_ratio*100:.2f}")
                    if spoof_ratio > self.spoof_confidence:
                        print(f"Spoofing detected!: {spoof_ratio *100:.2f}% frame of spoof")
                        cap.release()
                        cv2.destroyAllWindows()
                        return False

                # 2. Cue Verification 
                if not cue_completed:

                    cue_in_progress, current_cue = cue_verification.run_verification(frame)
                    
                    # Display current cue
                    if current_cue:
                        cv2.putText(frame, current_cue, (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    # Check if cue verification is complete
                    if not cue_in_progress:
                        cue_completed = True
                        print("Cue verification completed successfully")
              
                    
                
                if cue_completed:
                    # 3. Live Auth
                    # Finding face locations and encodings in frame
                    
                    face_encoding = fr.face_encodings(rgb_frame, face_loc)

                    # Processing each face
                    for (top, right, bottom, left), face_encoding in zip(face_loc, face_encoding):
                        name = "unknown"
                        access_granted = False

                        # Checking if target user is in known encoding
                        if target_name in self.authorized_users:
                            # Compare with specific user's encoding
                            matches = fr.compare_faces(
                                [self.authorized_users[target_name]['encoding']],
                                face_encoding,
                                tolerance=tolerance
                            )

                            print(f"Face match: {matches[0]}")

                            # Verifying identity
                            if matches[0]:
                                name = target_name
                                access_granted = True
                                print(f"Access granted to {target_name}")

                                # Adding visuals for access granted
                                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
                                cv2.putText(frame, f"Access Granted for {name}!",
                                            (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                            0.9, (0, 255, 0), 2)
                                
                                cap.release()
                                cv2.destroyAllWindows()
                                return True

                        # Draw rectangle and name for face
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255) if not access_granted else (0, 255, 0), 2)
                        cv2.putText(frame, name, (left+6, bottom-6),
                                    cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 1)

                        # If a cue is in progress, display the current cue
                        if current_cue:
                            cv2.putText(frame, current_cue, (50, 50), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Show the frame
            cv2.imshow('Face Auth', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            attempts += 1

        cap.release()
        cv2.destroyAllWindows()
        print(f"Access denied for {target_name}")
        return False



