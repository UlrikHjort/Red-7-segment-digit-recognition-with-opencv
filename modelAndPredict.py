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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, InputLayer, Flatten
from tensorflow.keras.models import Sequential, Model
import os
import time

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#############################################################################
#
#
#
#############################################################################
class DigitsPredict :

    def __init__(self):    
        self.original_image=[]
        self.probability_acceptance = 0.6
        
    #############################################################################
    #
    # Add gaussian noise to image
    #
    #############################################################################            
    def noise(self,image, mean=0, var=0.1):
        row,col= image.shape
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)      
        return (image + gauss)


    #############################################################################
    #
    # Read data sets and generate training data sets for model creation
    #
    #############################################################################    
    def generateLists(self):
        no_of_digits = 10
        
        shape_list = []
        image_list = []
        label_list = []
        
        # Read all image dimensions and use largest for resizing of the images
        for i in range(no_of_digits):
            image = cv2.imread("./data/"+ str(i)+".png", cv2.IMREAD_UNCHANGED)
            #image = cv2.imread("./data/"+ str(i)+".png")            
            shape_list.append(image.shape)
            print(image.shape)
        self.dim = max(shape_list)[:2]    
        
        for i in range(no_of_digits):
            image = cv2.imread("./data/"+ str(i)+".png",cv2.IMREAD_UNCHANGED)
            self.original_image.append(image)
            image = cv2.resize(image, self.dim, interpolation = cv2.INTER_AREA)
            image =np.array(image)
            image = image.astype('float32')
            image = image * 1.0/127.5 - 1.0 # Normalize image 
            image_list.append(image)
            label_list.append(i)
                
        self.x_train=tf.cast(np.array(image_list), tf.float64)
        self.y_train=tf.cast( np.array(list(map(int,label_list))),tf.int32)


    #############################################################################
    #
    #  Create model and train it. Simple Keras model is used with additional
    #  dropout layer for better generalization and to avoid overfitting the
    #  training data. 
    #
    #############################################################################    
    def createModel(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(self.dim[1],self.dim[0])),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ])

        predictions = self.model(self.x_train[:1]).numpy()
        tf.nn.softmax(predictions).numpy()
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        loss_fn(self.y_train[:1], predictions).numpy()

        self.model.compile(optimizer='adam',
                           loss=loss_fn,
                           metrics=['accuracy'])

        self.model.fit(self.x_train, self.y_train, epochs=5)
        self.probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])


    #############################################################################
    #
    #
    #
    #############################################################################    
    def predict(self, image):
        predictions = self.probability_model.predict(tf.cast(np.array(image), tf.float64))
        print (predictions[0])
        predicted = np.argmax(predictions[0])

        # Evaluate highest probalility in predict list against acceptance critia
        if predictions[0][predicted] > self.probability_acceptance:
            return predicted # Ok
        else:
            return -1 # Fail


    #############################################################################
    #
    # Capture a digit with cam and predict it by the keras model 
    #
    #############################################################################            
    def captureAndPredict(self):

        # Define (roughly) crop window for digits in image
        cy=45
        cx=250
        ch=200
        cw=200

        # Enable camera
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 420)

        # Capture a frame
        _, frame = cap.read()

        # Set contract and brightness    
        cv2.normalize(frame, frame, 0, 200, cv2.NORM_MINMAX) 

        # Do a rough crop of the digit area
        crop = frame[cy:cy+ch, cx:cx+cw]
            
        # Extract and create mask for the red segments     
        _, mask = cv2.threshold(crop[:, :,2], 190, 255, cv2.THRESH_BINARY)
        masking = np.zeros_like(crop)
        masking[:, :, 0] = mask
        masking[:, :, 1] = mask
        masking[:, :, 2] = mask

        masked = cv2.bitwise_and(crop, masking)
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        
        # Extract digits/contours in image and save them. File named by contour x position  
        thresh = 255-cv2.threshold(gray, 100, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]

        # Define approximately height and width (+/- 10 pixels) for digit to be able to filter out "garbage" samples
        limit_h_low = self.dim[0] - 10
        limit_h_high = self.dim[0] + 10            
        limit_w_low = self.dim[1] - 10
        limit_w_high = self.dim[1] + 10

        # Iterate contours and detect digits
        predicted = -1
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)

            # Detect and skip 'garbage' from dimension
            if w < limit_w_low or h < limit_h_low or w > limit_w_high or h > limit_h_high:
                continue 

            # Process digit for prediction
            image = thresh[y:y+h, x:x+w]
            image = cv2.resize(image, self.dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite('./tmp/' + str(x) + '.png',image)
            cv2.imwrite('./tmp/gray.png',gray)                                
            image = np.array(image)
            image = image.astype('float32')
            image = image * 1.0/127.5 - 1.0 # Normalize image

            # Predict digit
            predicted =  self.predict([image])
            if predicted > 0:
                break
                        
        cap.release()
        return predicted

    
    #############################################################################
    #
    # 
    #
    #############################################################################            
    def test(self):
        image_index = 4
        image = [self.x_train[image_index]]
        predicted = self.predict(image)
        cv2.imshow(str(predicted), self.original_image[image_index])


#############################################################################
#
# Main 
#
#############################################################################                    
p = DigitsPredict()
p.generateLists()
p.createModel()

while True:
    digit = p.captureAndPredict()
    if digit > -1:
        print("Digit = ", digit)
    time.sleep(0.5)

