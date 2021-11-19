import cv2 #Used for computer's camera
import numpy as np #Performs complex mathematical/list operations
import pandas as pd #To treat the data as a dataframe
import seaborn as sns #To pretify the chart we draw with matplotlib
import matplotlib.pyplot as plt #Used to draw charts
from sklearn.datasets import fetch_openml #This function allows us to retrieve a dataset by name from OpenML, a public repository for machine learning data and experiments.
from sklearn.model_selection import train_test_split #Helps us to test and train the data
from sklearn.linear_model import LogisticRegression #Used to create a logistic regression classifier
from sklearn.metrics import accuracy_score #Used to find out the accuracy of our prediction model
from PIL import Image
import PIL.ImageOps
import os,ssl,time

##Setting an HTTPS Context to fetch data from OpenML
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and 
getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

X, Y = fetch_openml('mnist_784', version = 1, return_X_y = True)
print(pd.Series(Y).value_counts())

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
nclasses = len(classes)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 2500, train_size = 7500, random_state = 9)
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(X_train_scaled, Y_train)

Y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(Y_test, Y_pred)
print('Accuracy: ', accuracy)

cap = cv2.VideoCapture(0)

while (True):
    #Capture frame-by-frame
    try:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #For drawing a box in the centre of the video.
        height, width = gray.shape()
        upperLeft = (int(width/2-56), int(height/2-56))
        bottomRight = (int(width/2+56), int(height/2+56))
        cv2.rectangle(gray, upperLeft, bottomRight, (0, 255, 0), 2)
        
        #To only consider the area inside the box for detecting the digit 
        #roi = Region Of Interest
        roi = gray[upperLeft[1]: bottomRight[1], upperLeft[0], bottomRight[0]]

        #Converting cv2 image to PIL format
        image_PIL = Image.formarray(roi)
        # convert to grayscale image - 'L' format means each pixel is 
        # represented by a single value from 0 to 255
        image_bw = image_PIL.convert('L')
        #resize - Resizes the image.
        image_bw_resized = image_bw.resize((28, 28), Image.ANTIALIAS)
        #We need to invert the image since we get a mirrored image.
        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
        pixel_filter = 20
        #Percentile function converts the values in scalar quantity.
        min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
        #We have to make it scalar to get the minimum pixel and limit its value between 0 &255
        #and then getting the aximum pixel of the image.
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted - min_pixel, 0, 255)
        max_pixel = np.max(image_bw_resized_inverted)
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1, 784)
        test_pred = clf.predict(test_sample)
        print('Predicted class is ', test_pred)
        cv2.imshow('frame', gray)
        
        #Adding a key control to turn off the camera, using the q key.
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()