import os
import csv
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import math

app = Flask(__name__)

# Dummy database for demo purposes
users_db = {
    "user1": {"password": "password123"}
}

# Define the main directory to store user data
DATA_DIR = 'data/'

# Function to create directory for user if it doesn't exist
def create_user_directory(user_id):
    user_dir = os.path.join(DATA_DIR, user_id)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    return user_dir

def save_mouse_data(user_id, mouse_data):
    # Create the user's directory
    user_dir = create_user_directory(user_id)
    
    # Define the path to the mouse data file
    mouse_file = os.path.join(user_dir, f'{user_id}_mouse_data.csv')
    header = [
        'movement_id', 'startX', 'startY', 'endX', 'endY', 
        'deltaX', 'deltaY', 'distance', 'duration', 'velocity'
    ]
    file_exists = os.path.exists(mouse_file)

    # Initialize variables for tracking movements
    movement_id = 1
    previous_x, previous_y, previous_time = None, None, None

    with open(mouse_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        if not file_exists:
            writer.writeheader()  # Write header if file doesn't exist

        for event in mouse_data:
            # Extract current position and time
            current_x = event.get('x', 0)
            current_y = event.get('y', 0)
            current_time = event.get('time', datetime.now().timestamp())

            # Calculate metrics
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
            # Update previous position and time
            previous_x, previous_y, previous_time = current_x, current_y, current_time

def save_keyboard_data(user_id, keyboard_data):
    # Create the user's directory
    user_dir = create_user_directory(user_id)
    
    # Define the path to the keyboard data file
    keyboard_file = os.path.join(user_dir, f'{user_id}_keyboard_data.csv')
    header = [
        'key_id', 'key', 'operation', 'pX', 'pY', 'LR', 
        'state', 'delta', 'time_diff', 'time_since_beginning', 
        'press_to_press', 'release_to_press', 'hold_time'
    ]
    file_exists = os.path.exists(keyboard_file)
    key_id = 1

    # Read existing data to avoid duplicates
    existing_data = []
    if file_exists:
        with open(keyboard_file, mode='r') as file:
            reader = csv.DictReader(file)
            existing_data = [row for row in reader]

    with open(keyboard_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        if not file_exists:
            writer.writeheader()  # Write header if file doesn't exist

        for event in keyboard_data:
            # Check for duplicate
            duplicate = any(
                row['key'] == event['key'] and
                row['operation'] == '1' and
                row['time_diff'] == '0'
                for row in existing_data
            )
            if not duplicate:
                writer.writerow({
                    'key_id': key_id,
                    'key': event.get('key', ''),
                    'operation': 1,  # Assuming "1" for press
                    'pX': 0,  # Placeholder
                    'pY': 0,  # Placeholder
                    'LR': 0,  # Placeholder
                    'state': 1,  # Placeholder
                    'delta': 0,  # Placeholder
                    'time_diff': 0,  # Placeholder
                    'time_since_beginning': 0,  # Placeholder
                    'press_to_press': 0,  # Placeholder
                    'release_to_press': 0,  # Placeholder
                    'hold_time': event.get('holdTime', 0)
                })
                key_id += 1


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

    if user_id in users_db and users_db[user_id]['password'] == password:
        save_mouse_data(user_id, mouse_movements)
        save_keyboard_data(user_id, keyboard_behavior)
        return jsonify({"message": "Login successful and behavior recorded!"})
    else:
        return jsonify({"message": "Invalid user ID or password!"})

if __name__ == '__main__':
    app.run(debug=True)
