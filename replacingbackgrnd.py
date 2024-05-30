import cv2
import mediapipe as mp
import numpy as np

mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

image = cv2.imread('C:/Users/dell/Desktop/opencv/mediapipe/images.jpg')

# Load the new background image

background_image = cv2.imread('C:/Users/dell/Desktop/opencv/mediapipe/download.jpg')



# Resize the background image to match the original image dimensions
background_image = cv2.resize(background_image, (image.shape[1], image.shape[0])) # height-0,width 1

# Convert the BGR image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image to get the segmentation mask
results = selfie_segmentation.process(image_rgb)

# Extract the mask
mask = results.segmentation_mask

# Create the condition array based on the segmentation mask
condition = mask > 0.5  # Foreground where the mask probability is greater than 0.5

# Apply the condition to combine the foreground from the original image and the new background
output= np.where(condition[:, :, None], image, background_image)

cv2.imshow('original',image)

# Display the resulting image
cv2.imshow('Output Image', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
