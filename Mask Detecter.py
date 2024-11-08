import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from datetime import datetime

class CrowdMaskDetection:
    def __init__(self, face_cascade_path, model_path, data_folder):
        """Initialize the face detection model and mask detection model."""
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.model = load_model(model_path)
        self.video_capture = cv2.VideoCapture(0)  # Capture video from default camera
        self.data_folder = data_folder
        if not os.path.exists(self.data_folder):  # Ensure the folder exists
            os.makedirs(self.data_folder)
        self.screenshot_taken = False  # Flag to ensure only one screenshot is taken

    def detect_faces(self, frame):
        """Detect faces in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        return faces

    def preprocess_face(self, face_frame):
        """Preprocess the face frame for mask prediction."""
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        face_frame = cv2.resize(face_frame, (224, 224))  # Resize to 224x224
        face_frame = img_to_array(face_frame)  # Convert to numpy array
        face_frame = np.expand_dims(face_frame, axis=0)  # Add batch dimension
        face_frame = preprocess_input(face_frame)  # Preprocess for MobileNetV2
        return face_frame

    def predict_mask(self, face_frame):
        """Predict whether the person is wearing a mask or not."""
        face_frame = self.preprocess_face(face_frame)
        preds = self.model.predict(face_frame)
        return preds[0]  # Return mask vs no-mask prediction

    def draw_results(self, frame, faces, preds):
        """Draw rectangles around faces and display mask predictions."""
        mask_wearers = []  # List to store the positions of faces wearing masks
        for (x, y, w, h), pred in zip(faces, preds):
            (mask, withoutMask) = pred
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Display the label with confidence
            label_text = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # If the person is wearing a mask, add them to the mask_wearers list
            if mask > withoutMask:
                mask_wearers.append((x, y, w, h))  # Append face's coordinates

        return mask_wearers

    def save_mask_wearer_image(self, frame, face_coordinates):
        """Save the image of the first detected mask wearer."""
        (x, y, w, h) = face_coordinates
        face_frame = frame[y:y + h, x:x + w]  # Extract the face
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.data_folder, f"mask_wearer_{timestamp}.jpg")
        cv2.imwrite(file_path, face_frame)  # Save the face image
        print(f"Saved image of mask wearer: {file_path}")

    def run(self):
        """Run the mask detection system for crowd identification."""
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            faces = self.detect_faces(frame)  # Detect faces in the frame
            preds = [self.predict_mask(frame[y:y + h, x:x + w]) for (x, y, w, h) in faces]  # Predict masks
            mask_wearers = self.draw_results(frame, faces, preds)  # Draw results on the frame

            # If there are mask wearers and screenshot has not been taken yet
            if mask_wearers and not self.screenshot_taken:
                self.save_mask_wearer_image(frame, mask_wearers[0])  # Save the first mask-wearer's image
                self.screenshot_taken = True  # Set flag to True to avoid saving multiple screenshots

            # Display the resulting frame with detections and predictions
            cv2.imshow("Crowd Mask Detection", frame)

            # Print the list of mask wearers with their face positions (coordinates)
            if mask_wearers:
                print(f"People wearing masks: {mask_wearers}")

            # Exit the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the capture object and close all OpenCV windows
        self.video_capture.release()
        cv2.destroyAllWindows()

# Paths to the cascade classifier and mask detection model
face_cascade_path = r"C:\Users\Nelson V I\Documents\Python Learning\data\haarcascade_frontalface_alt2.xml"
model_path = r"C:\Users\Nelson V I\Documents\Python Learning\data\mask_recog1.h5"
data_folder = r"C:\Users\Nelson V I\Documents\Python Learning\data"  # Folder to save images

# Initialize the system and run
if __name__ == "__main__":
    mask_detector = CrowdMaskDetection(face_cascade_path, model_path, data_folder)
    mask_detector.run()
