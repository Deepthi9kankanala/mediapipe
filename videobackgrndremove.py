import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation()


video_path = "C:\\Users\\dell\\Desktop\\opencv\\mediapipe\\mediavideo.mp4"

# Attempt to open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

while cap.isOpened():
    # Read frame
    success, frame = cap.read()

    # Check if frame is read successfully
    if not success:
        print("Error: Unable to read frame.")
        break

    # Resize frame
    frame = cv2.resize(frame, (600, 400))

    # Convert to RGB
    image_rgb= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the image to get the segmentation mask
    results = selfie_segmentation.process(image_rgb)

    # Extract the mask
    mask = results.segmentation_mask

    # Create a background image (white)
    background_image = 255 * np.ones_like(frame)

    # Apply the mask to get the foreground
    condition = mask > 0.5
    foreground = np.where(condition[:, :, None], frame, background_image)


    # Display frame and its grayscale version
    cv2.imshow("original",frame)
    cv2.imshow("remove backgrnd",foreground)

    # Check for 'q' key press to exit ,cv2.waitKey(25)-returns ascii val
    # ord('q') -returns ascii val of q , 0xFF is 255 heaxadecimal -11111111
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
cv2.destroyAllWindows()
