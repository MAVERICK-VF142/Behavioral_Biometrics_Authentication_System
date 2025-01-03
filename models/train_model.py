import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_user_model(user_id):
    user_dir = os.path.join('data', user_id)
    mouse_file = os.path.join(user_dir, f'{user_id}_mouse_data.csv')
    keyboard_file = os.path.join(user_dir, f'{user_id}_keyboard_data.csv')
    
    # Check if files exist
    if not os.path.exists(mouse_file) or not os.path.exists(keyboard_file):
        print(f"Data files for {user_id} are missing.")
        return False

    # Load data
    mouse_data = pd.read_csv(mouse_file)
    keyboard_data = pd.read_csv(keyboard_file)

    # Combine data for training
    # Use relevant features only; exclude IDs and timestamps
    features = ['deltaX', 'deltaY', 'distance', 'duration', 'velocity', 'hold_time']
    mouse_data = mouse_data[['deltaX', 'deltaY', 'distance', 'duration', 'velocity']]
    keyboard_data = keyboard_data[['hold_time']]
    
    combined_data = pd.concat([mouse_data, keyboard_data], axis=1).dropna()

    # Labels (1 = valid user)
    labels = [1] * len(combined_data)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(combined_data, labels, test_size=0.2, random_state=42)

    # Train a RandomForestClassifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Training complete for {user_id}. Accuracy: {accuracy}")

    # Save the model
    model_file = os.path.join(user_dir, f'{user_id}_model.pkl')
    joblib.dump(clf, model_file)

    return True
