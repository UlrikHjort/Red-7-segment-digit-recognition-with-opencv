#####################################################################-
#             7 segment digit detection with opencv
# 
#           Copyright (C) 2021 By Ulrik Hørlyk Hjort
#
#  This Program is Free Software; You Can Redistribute It and/or
#  Modify It Under The Terms of The GNU General Public License
#  As Published By The Free Software Foundation; Either Version 2
#  of The License, or (at Your Option) Any Later Version.
#
#  This Program is Distributed in The Hope That It Will Be Useful,
#  But WITHOUT ANY WARRANTY; Without Even The Implied Warranty of
#  MERCHANTABILITY or FITNESS for A PARTICULAR PURPOSE.  See The
#  GNU General Public License for More Details.
#
# You Should Have Received A Copy of The GNU General Public License
# Along with This Program; if not, See <Http://Www.Gnu.Org/Licenses/>.
#######################################################################
import cv2
import numpy as np

# Enable camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 420)

# Define (roughly) crop window for digits in image
y=45
x=250
h=200
w=200

_, frame = cap.read()
while True:

    # Set contract and brightness    
    cv2.normalize(frame, frame, 0, 200, cv2.NORM_MINMAX) 

    crop = frame[y:y+h, x:x+w]

    # Extract and create mask for the red segments     
    _, mask = cv2.threshold(crop[:, :,2], 190, 255, cv2.THRESH_BINARY)
    masking = np.zeros_like(crop)
    masking[:, :, 0] = mask
    masking[:, :, 1] = mask
    masking[:, :, 2] = mask

    masked = cv2.bitwise_and(crop, masking)

    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        
    cv2.imshow('Digits', gray)    

    key = cv2.waitKey(0) & 0xFF

    # Extract digits/contours in image and save them. File named by contour x position  
    if key == ord('s'): 
        thresh = 255-cv2.threshold(gray, 100, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]

        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.imwrite('./tmp/' + str(x) + '.png',thresh[y:y+h, x:x+w])
        break
    # Quit
    elif key == ord('q'):
        break
    # Recapture frame
    elif key == ord('r'):
        _, frame = cap.read()            


cap.release()
cv2.destroyWindow('Digits')
