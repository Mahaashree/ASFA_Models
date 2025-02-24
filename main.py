import os
from face_auth import FaceAuth
from db_utils import MongoDBUtils

def main():
    mongodb_uri = os.getenv('MONGODB_URI')
    db_utils = MongoDBUtils(mongodb_uri)
    # User interactions, e.g., registration, authentication

    
    name = input("Enter your name: ")

    face_auth = FaceAuth(db_utils,name)
    
    result = face_auth.live_auth(name)
    if result:
        print("Access granted!")
    else:
        print("Access denied.")

if __name__ == '__main__':
    main()