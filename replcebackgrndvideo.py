import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation()

# File paths
video_path = "C:/Users/dell/Desktop/opencv/mediapipe/mediavideo.mp4"
background_image_path = 'C:/Users/dell/Desktop/opencv/mediapipe/download.jpg'

# Attempt to open the video file
cap = cv2.VideoCapture(video_path)

# Load the background image
background_image = cv2.imread(background_image_path)

# Check if the video file is opened successfully
if not cap.isOpened():
    print(f"Error: Unable to open video file: {video_path}")
    exit()

# Check if the background image is loaded successfully
if background_image is None:
    print(f"Error: Unable to load background image: {background_image_path}")
    exit()

# Get the frame size from the video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Resize the background image to match the video frame size
background_image = cv2.resize(background_image, (frame_width, frame_height))

while cap.isOpened():
    # Read frame
    success, frame = cap.read()

    # Check if frame is read successfully
    if not success:
        print("Finished processing video or unable to read frame.")
        break

    # Convert to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image to get the segmentation mask
    results = selfie_segmentation.process(image_rgb)

    # Extract the mask
    mask = results.segmentation_mask

    # Create the condition array based on the segmentation mask
    condition = mask > 0.5  # Foreground where the mask probability is greater than 0.5

    # Apply the condition to combine the foreground from the original frame and the new background
    output_frame = np.where(condition[:, :, None], frame, background_image)

    # Display the resulting frame
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Background Replaced Frame", output_frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
