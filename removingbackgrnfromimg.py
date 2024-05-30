#steps:

"""1.  firstly intilize selfie segmenation model:

    1. general model : pass 0 as param , default 256*256*1(channeli/p) o/p 256*256*3
    2. Landscape Model:  pass 1 as param, 144*256*1 (i/p) 144*256*3"""

#initilizing segment
# setting segment
# reading img
#converting bgr to rgb 
#Process the input image to get the segmentation mask
#Extracts the segmentation mask



import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt


#step 1 

#soln-clss , selfie_segmenation : model

change_background = mp.solutions.selfie_segmentation # initializing model

change_bg_segment = change_background.SelfieSegmentation() #setting s egmenation fn

cap = cv2.imread('C:/Users/dell/Desktop/opencv/mediapipe/images.jpg')  
image_rgb = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB) 

results = change_bg_segment.process(image_rgb) 
mask = results.segmentation_mask

#Creates a white background image with the same dimensions as the input image.  
background_image = 255 * np.ones_like(cap) # 
# Apply the mask to get the foreground

#  2D array of the same height and width as the input image, containing values between 0 and 1 that represent the probability of each pixel belonging to the foreground., 
# condition returs boolean vals 

# expands the dimensions of the condition array from (height, width) to (height, width, 1)
#  select pixels if trur original img cap , false background
condition = mask > 0.5
foreground = np.where(condition[:, :, None], cap, background_image)

cv2.imshow('original',cap)

    # Display the resulting frame

cv2.imshow('Foreground', foreground)
cv2.waitKey(0) # wait fir gvn time to destroty window 

cap.release()
cv2.destroyAllWindows()