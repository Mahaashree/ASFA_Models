import pymongo
from dotenv import load_dotenv
import face_recognition as fr

class MongoDBUtils:
    def __init__(self, mongodb_uri):

        load_dotenv()

        try:
            self.client = pymongo.MongoClient(mongodb_uri)
            self.db = self.client['face_authentication']
            self.users_collection = self.db['registered_users']
            print("MongoDB connection established successfully")

        except Exception as e:
            print(f"MongoDB connection error: {e}")
            raise

    def sync_users_to_db(self):
        #Syncing local known users to MongoDB

        for name, user_data in self.authorized_users.items():
            #checking if user already exists
            usr_exists = self.users_collection.find_one({'name': name})

            if not usr_exists:
                #inserting new user
                user_doc = {
                    'name': name,
                    'encoding' : user_data['encoding'],
                    'access_level': user_data['access_level']
                }
                self.users_collection.insert_one(user_doc)
                print(f"Added {name} to db")
            else:
                #update existing user
                self.users_collection.update_one(
                    {'name': name},
                    {'$set': {
                        'encoding': user_data['encoding'],
                        'access_level': user_data['access_level']
                    }}
                )

    def get_user_from_db(self, name):
        
        return self.users_collection.find_one({'name':name})
    

    def register(self, name, img_path):
        try:
            #loading img
            img = fr.load_image_file(img_path)
            face_loc = fr.face_locations(img)

            if not face_loc:
                print(f"No face detected in {img_path}")
                return False
            
            encodings = fr.face_encodings(img, face_loc)
            if encodings:
                encoding = encodings[0].tolist()

                #check if user already exists
                usr_exists = self.users_collection.find_one({'name': name})
                if usr_exists:
                    print(f"User {name} already exists")
                    return False
                
                user_doc ={
                    'name': name,
                    'encoding': encoding,
                    'access_level': 'standard'
                }

                self.users_collection.insert_one(user_doc)
                print(f"User {name} registered successfully")
                return True
            else:
                print(f"Failed to extract face encoding from {img_path}")
                return False
        except Exception as e:
            print(f"Error registering user: {name}: {e}")
            return False
                

    








