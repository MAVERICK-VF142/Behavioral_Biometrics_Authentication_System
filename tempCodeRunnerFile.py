import os
import cv2
import csv
import pandas as pd
import joblib
import math
import logging
from datetime import datetime
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Set up logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Dummy database for demo purposes
users_db = {
    "user1": {"password": "password123", "login_count": 0}
}

# Define the directory to store user data
USER_DATA_DIR = 'data/'

# Ensure directory exists
if not os.path.exists(USER_DATA_DIR):
    os.makedirs(USER_DATA_DIR)
    logger.info(f"Created directory: {USER_DATA_DIR}")

# Store the mouse and keyboard data into CSV
def save_mouse_data(user_id, mouse_data):
    user_dir = os.path.join(USER_DATA_DIR, user_id)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
        logger.info(f"Created user directory: {user_dir}")

    mouse_file = os.path.join(user_dir, f'{user_id}_mouse_data.csv')
    header = ['movement_id', 'startX', 'startY', 'endX', 'endY', 'deltaX', 'deltaY', 'distance', 'duration', 'velocity']
    movement_id = 1
    previous_x, previous_y, previous_time = None, None, None

    with open(mouse_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        if os.stat(mouse_file).st_size == 0:
            writer.writeheader()

        for event in mouse_data:
            current_x = event.get('x', 0)
            current_y = event.get('y', 0)
            current_time = event.get('time', datetime.now().timestamp())
            
            if previous_x is not None and previous_y is not None:
                delta_x = current_x - previous_x
                delta_y = current_y - previous_y
                distance = math.sqrt(delta_x**2 + delta_y**2)
                duration = current_time - previous_time
                velocity = distance / duration if duration > 0 else 0

                writer.writerow({
                    'movement_id': movement_id,
                    'startX': previous_x,
                    'startY': previous_y,
                    'endX': current_x,
                    'endY': current_y,
                    'deltaX': delta_x,
                    'deltaY': delta_y,
                    'distance': round(distance, 3),
                    'duration': round(duration, 3),
                    'velocity': round(velocity, 3)
                })
                movement_id += 1
            previous_x, previous_y, previous_time = current_x, current_y, current_time
    logger.info(f"Mouse data saved for user {user_id} in {mouse_file}")

def save_keyboard_data(user_id, keyboard_data):
    user_dir = os.path.join(USER_DATA_DIR, user_id)
    keyboard_file = os.path.join(user_dir, f'{user_id}_keyboard_data.csv')
    header = ['key_id', 'key', 'operation', 'pX', 'pY', 'LR', 'state', 'delta', 'time_diff', 'time_since_beginning', 'press_to_press', 'release_to_press', 'hold_time']
    
    key_id = 1
    with open(keyboard_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        if os.stat(keyboard_file).st_size == 0:
            writer.writeheader()

        for event in keyboard_data:
            if 'holdTime' not in event or event['holdTime'] == 0:
                continue
            
            writer.writerow({
                'key_id': key_id,
                'key': event.get('key', ''),
                'operation': 1,  # Assuming 1 for key press
                'pX': 0,  # Placeholder
                'pY': 0,  # Placeholder
                'LR': 0,  # Placeholder
                'state': 1,  # Placeholder
                'delta': 0,  # Placeholder
                'time_diff': 0,  # Placeholder
                'time_since_beginning': 0,  # Placeholder
                'press_to_press': 0,  # Placeholder
                'release_to_press': 0,  # Placeholder
                'hold_time': event['holdTime']
            })
            key_id += 1
    logger.info(f"Keyboard data saved for user {user_id} in {keyboard_file}")

# Function to make predictions using only the newly received data
def predict_user(user_id, mouse_data, keyboard_data):
    user_dir = os.path.join(USER_DATA_DIR, user_id)
    model_file = os.path.join(user_dir, f'{user_id}_model.pkl')

    if not os.path.exists(model_file):
        logger.warning(f"Model file not found for user {user_id}.")
        return False  # Model not found

    clf = joblib.load(model_file)

    # Process the new mouse data for prediction
    mouse_features = []
    previous_x, previous_y, previous_time = None, None, None
    for event in mouse_data:
        current_x = event.get('x', 0)
        current_y = event.get('y', 0)
        current_time = event.get('time', datetime.now().timestamp())
        
        if previous_x is not None and previous_y is not None:
            delta_x = current_x - previous_x
            delta_y = current_y - previous_y
            distance = math.sqrt(delta_x**2 + delta_y**2)
            duration = current_time - previous_time
            velocity = distance / duration if duration > 0 else 0
            mouse_features.append([delta_x, delta_y, distance, duration, velocity])
        previous_x, previous_y, previous_time = current_x, current_y, current_time

    # Process the new keyboard data for prediction
    keyboard_features = [[event.get('holdTime', 0)] for event in keyboard_data if 'holdTime' in event]

    # Combine features for prediction
    if mouse_features and keyboard_features:
        combined_features = pd.concat([pd.DataFrame(mouse_features), pd.DataFrame(keyboard_features)], axis=1)
        prediction = clf.predict(combined_features)
        logger.info(f"Prediction result for user {user_id}: {prediction[0]}")
        return prediction[0] == 1  # Assuming "1" is the predicted class for correct user

    logger.warning(f"No valid mouse or keyboard data for prediction for user {user_id}.")
    return False

def verify_face(user_id):
    """ Capture and compare face image using OpenCV """
    # Load the reference image of user1 (for face comparison)
    reference_image_path = os.path.join(USER_DATA_DIR, user_id, f'{user_id}_face.jpg')
    if not os.path.exists(reference_image_path):
        logger.error(f"Reference face image not found for {user_id}.")
        return False
    
    # Initialize the OpenCV face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)  # Use the default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to capture image from the webcam.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # If faces are detected, check if it matches the reference face image
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            # Save the face image for comparison
            cv2.imwrite('captured_face.jpg', face_img)

            # Compare with the reference image (user1's face)
            reference_face = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
            captured_face = cv2.imread('captured_face.jpg', cv2.IMREAD_GRAYSCALE)

            # Compare the images (you can use any method, for now, simple comparison)
            # Using the OpenCV matchTemplate for face comparison (simplified)
            res = cv2.matchTemplate(captured_face, reference_face, cv2.TM_CCOEFF_NORMED)
            threshold = 0.7  # Set a threshold value for face match

            if res.max() >= threshold:
                cap.release()
                cv2.destroyAllWindows()
                logger.info(f"Face verification succeeded for {user_id}.")
                return True  # Face matched successfully
        
        # Show the webcam feed
        cv2.imshow('Face Verification', frame)

        # Exit condition for the camera window (press 'q' to quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.error(f"Face verification failed for {user_id}.")
    return False  # Face verification failed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user_id = data['userId']
    password = data['password']
    mouse_movements = data['mouseMovements']
    keyboard_behavior = data['keyboardBehavior']

    # Validate user and password
    if user_id in users_db and users_db[user_id]['password'] == password:
        logger.info(f"User {user_id} login attempt successful.")
        # Save data
        save_mouse_data(user_id, mouse_movements)
        save_keyboard_data(user_id, keyboard_behavior)
        
        # Increment login count and check if model should be retrained
        users_db[user_id]['login_count'] += 1

        # Train model after every 20 successful logins
        if users_db[user_id]['login_count'] >= 20:
            retrain_model(user_id)
            users_db[user_id]['login_count'] = 0  # Reset login count

        # Predict based on the newly received data
        is_correct_user = predict_user(user_id, mouse_movements, keyboard_behavior)
        
        if is_correct_user:
            # Perform face ID verification if the prediction is successful
            if user_id == "user1" and verify_face(user_id):
                return jsonify({"message": "Login successful with face verification!"})
            else:
                return jsonify({"message": "Login successful, but face verification failed."})
        else:
            return jsonify({"message": "Login failed: User behavior mismatch."})

    else:
        logger.warning(f"Failed login attempt for user {user_id}. Invalid credentials.")
        return jsonify({"message": "Invalid user ID or password!"})

def retrain_model(user_id):
    user_dir = os.path.join(USER_DATA_DIR, user_id)
    mouse_file = os.path.join(user_dir, f'{user_id}_mouse_data.csv')
    keyboard_file = os.path.join(user_dir, f'{user_id}_keyboard_data.csv')

    if not os.path.exists(mouse_file) or not os.path.exists(keyboard_file):
        logger.warning(f"Insufficient data to train model for user {user_id}.")
        return

    mouse_data = pd.read_csv(mouse_file)[['deltaX', 'deltaY', 'distance', 'duration', 'velocity']]
    keyboard_data = pd.read_csv(keyboard_file)[['hold_time']]
    combined_data = pd.concat([mouse_data, keyboard_data], axis=1)

    y = [1] * len(combined_data)  # Dummy labels for correct user behavior
    X = combined_data

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    clf.fit(X, y)

    model_file = os.path.join(user_dir, f'{user_id}_model.pkl')
    joblib.dump(clf, model_file)
    logger.info(f"Model trained and saved for {user_id}")

if __name__ == '__main__':
    app.run(debug=True)
