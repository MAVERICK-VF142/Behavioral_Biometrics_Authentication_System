import cv2
import os

def capture_user_image(user_id):
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Press 'c' to capture your face or 'q' to quit")

    while True:
        ret, frame = cap.read()

        # Show the captured frame
        cv2.imshow('Capture Face', frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):  # 'c' key is pressed to capture the image
            # Create user directory if it doesn't exist
            user_dir = os.path.join('data', user_id)
            if not os.path.exists(user_dir):
                os.makedirs(user_dir)

            # Define the image path
            image_path = os.path.join(user_dir, f'{user_id}_face.jpg')

            # Save the captured image to the specified path
            cv2.imwrite(image_path, frame)
            print(f"Image saved as {image_path}")

            # Break the loop after capturing the image
            break

        elif key == ord('q'):  # 'q' key is pressed to quit without capturing
            print("Exiting without capturing.")
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

# Example usage
user_id = 'user1'  # Use the user ID you want to capture the image for
capture_user_image(user_id)
