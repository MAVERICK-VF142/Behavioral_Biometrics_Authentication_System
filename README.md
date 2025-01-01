# Behavioral_Biometrics_Authentication_System

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
            # Skip events with no holdTime or other critical data
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