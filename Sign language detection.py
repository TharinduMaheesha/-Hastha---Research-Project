import numpy as np
#from  flask import Flask
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import numpy as np
import pickle
import cv2
import os
from matplotlib import pyplot as plt
import mediapipe as mp
from tensorflow.keras.models import Sequential#to build sequencial neural network
from tensorflow.keras.layers import LSTM, Dense #to perform action detection
from tensorflow.keras.callbacks import TensorBoard#to monitor and traise model

import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from rake_nltk import Rake 
import csv


import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os
from tensorflow import keras
import pickle 
import webbrowser
import cv2

from flask import Flask
from flask import request, url_for, redirect, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

     
@app.route('/months',methods=['POST'])
def months():


    poses=np.array(['January','February','March','April','May','June','July','August','September','October','November','December'])

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(poses.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights('months.h5')
    #res = model.predict(x_test)
        
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities

    def mediapipe_detection(image, model):#keypoint detection
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results

    def draw_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


    def draw_styled_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(40,22,60), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(40,44,121), thickness=2, circle_radius=2)
                                ) 
        # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                ) 
        # Draw right hand connections  
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                ) 

    def extract_keypoints(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        #face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lefthand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        righthand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose,lefthand,righthand])


    # 1. New detection variables
    sequence = []
    sentence = []
    threshold = 0.7

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
    #         sequence.insert(0,keypoints)
    #         sequence = sequence[:30]
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(poses[np.argmax(res)])
                
            
            #3. Viz logic
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if poses[np.argmax(res)] != sentence[-1]:
                            sentence.append(poses[np.argmax(res)])
                    else:
                        sentence.append(poses[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                #image = prob_viz(res, actions, image, colors)
                
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        with open('months_result.pkl','wb') as sn:
            pickle.dump(sentence[len(sentence)-1],sn)

        #print(sentence)

        with open('months_result.pkl','rb') as f:
            x = pickle.load(f)
            print(x)

            if x == 'January':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Months%2FJanuary.mp4?alt=media&token=32904393-07bd-45bc-83d4-d153b1db1582'
            elif x == 'February':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Months%2FFebruary.mp4?alt=media&token=fffc01c7-6295-487e-a218-36955874c7ff'
            elif x == 'March':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Months%2FMarch.mp4?alt=media&token=c942a175-6a68-4652-94b7-f35bf75326c4'
            elif x == 'April':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Months%2FApril.mp4?alt=media&token=852a7673-d26f-4f9e-83f5-8cf084b3bc62'
            elif x == 'May':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Months%2FMay.mp4?alt=media&token=da895e6f-ceb4-439f-b94b-2bd82b77273d'
            elif x == 'June':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Months%2FJune.mp4?alt=media&token=2839dab1-2d4c-4e54-a39f-40bd6fd718ff'
            elif x == 'July':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Months%2FJuly.mp4?alt=media&token=84fb007e-445f-4ff4-95d1-c484dfc99395'
            elif x == 'August':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Months%2FAugust.mp4?alt=media&token=73174ce7-afff-4b44-abf4-530b8917befa'
            elif x == 'September':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Months%2FSeptember.mp4?alt=media&token=5b6809cc-d233-45f6-943c-c20f8890fae0'
            elif x == 'October':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Months%2FOctober.mp4?alt=media&token=154c5ce7-d8b2-4667-ab42-2987d8fcc98d'
            elif x == 'November':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Months%2FNovember.mp4?alt=media&token=5c7def82-9329-414b-9e0e-8e85c852c5fd'  
            elif x == 'December':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Months%2FDecember.mp4?alt=media&token=6c593285-5d31-4d00-bc56-2aba35bf4daf'  
        

        return (render_template('index.html',value=x, mode=mode))

@app.route('/numbers',methods=['POST'])
def numberpredict():

    json_file = open("model-bw_numbers.json", "r")
    model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(model_json)
    # load weights into new model
    loaded_model.load_weights("model-bw_numbers.h5")
    print("Loaded model from disk")


    cap = cv2.VideoCapture(0)

    #categories = {A: 'A', B: 'B', C: 'C', D: 'D', E: 'E', F: 'FIVE', G: 'G', H: 'H', I: 'I', J: 'J', K: 'K', L: 'L',
    #            M: 'M', N: 'N', O: 'O', P: 'P', Q: 'Q', R: 'R', S: 'S', T: 'T', U: 'U', V: 'V', W: 'W', X: 'X',
    #           Y: 'Y', Z: 'Z'}

    while True:
        _, frame = cap.read()
        # Simulating mirror image
        frame = cv2.flip(frame, 1)
        
        # Got this from collect-data.py
        # Coordinates of the ROI
        x1 = int(0.5*frame.shape[1])
        y1 = 10
        x2 = frame.shape[1]-10
        y2 = int(0.5*frame.shape[1])
        # Drawing the ROI
        # The increment/decrement by 1 is to compensate for the bounding box
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
        # Extracting the ROI
        roi = frame[y1:y2, x1:x2]
        
        # Resizing the ROI so it can be fed to the model for prediction
        roi = cv2.resize(roi, (64, 64)) 
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
        cv2.imshow("test", test_image)
        # Batch of 1
        result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
        prediction = {
                    '0': result[0][0], 
                    '1': result[0][1], 
                    '2': result[0][2], 
                    '3': result[0][3], 
                    '4': result[0][4], 
                    '5': result[0][5], 
                    '6': result[0][6],
                    '7': result[0][7],
                    '8': result[0][8],
                    '9': result[0][9],
                    '10': result[0][10]
                    
                                }
        
        # Sorting based on top prediction
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        
        # Displaying the predictions
        
        cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 5, (0,255,255), 5)    
        cv2.imshow("Frame", frame)
        result = prediction[0][0]
        
        with open('result_number','wb') as f:
                pickle.dump(result,f)

        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == 27: # esc key
            break
            
    
    cap.release()
    cv2.destroyAllWindows()  
            

    with open('result_number','rb') as f:
        x = pickle.load(f)
        #print(x)

        if x == '0':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Numbers%2F0.mp4?alt=media&token=68689dd2-0bb9-4800-a0b8-d72d5344dda4'
        elif x == '1':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Numbers%2F1.mp4?alt=media&token=1d7fe796-2f46-4d64-ae00-4a3a97e7a1d8'
        elif x == '2':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Numbers%2F2.mp4?alt=media&token=d9d8b1fc-df0d-4b6b-ab4c-fad772c2d23e'
        elif x == '3':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Numbers%2F3.mp4?alt=media&token=52bfc797-9739-4aa3-b876-550ba8597d30'
        elif x == '4':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Numbers%2F4.mp4?alt=media&token=b7ee4eee-dcf7-492c-bd0b-ccd784f1ab01'
        elif x == '5':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Numbers%2F5.mp4?alt=media&token=b9906989-175e-4bd5-8364-a6461aef9a42'
        elif x == '6':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Numbers%2F6.mp4?alt=media&token=cda039cf-8766-438b-b092-5d966a3f0791'
        elif x == '7':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Numbers%2F7.mp4?alt=media&token=22e725c2-6a70-4226-9f28-c0f6c7e39668'
        elif x == '8':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Numbers%2F8.mp4?alt=media&token=567d1d8a-166f-4cd7-aa8a-9a33da12004e'
        elif x == '9':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Numbers%2F9.mp4?alt=media&token=1fe7cadb-d2d1-41da-b7bb-7fd391fc2ace'
        elif x == '10':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Numbers%2F10.mp4?alt=media&token=068f364d-e0cf-416c-88a6-48cb9bc80630'
        
    return (render_template('index.html',value=x, mode=mode))


@app.route('/days',methods=['POST'])
def days():


    poses=np.array(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(poses.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights('days-dw.h5')
    #res = model.predict(x_test)
        
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities

    def mediapipe_detection(image, model):#keypoint detection
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results

    def draw_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


    def draw_styled_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(40,22,60), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(40,44,121), thickness=2, circle_radius=2)
                                ) 
        # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                ) 
        # Draw right hand connections  
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                ) 

    def extract_keypoints(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        #face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lefthand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        righthand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose,lefthand,righthand])


    # 1. New detection variables
    sequence = []
    sentence = []
    threshold = 0.7

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
    #         sequence.insert(0,keypoints)
    #         sequence = sequence[:30]
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(poses[np.argmax(res)])
                
            
            #3. Viz logic
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if poses[np.argmax(res)] != sentence[-1]:
                            sentence.append(poses[np.argmax(res)])
                    else:
                        sentence.append(poses[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                #image = prob_viz(res, actions, image, colors)
                
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        with open('days_result.pkl','wb') as sn:
            pickle.dump(sentence[len(sentence)-1],sn)

        #print(sentence)

        with open('days_result.pkl','rb') as f:
            x = pickle.load(f)
            print(x)
            
            if x == 'Monday':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Days%2FMonday.mp4?alt=media&token=545e69cf-6b25-4931-bb29-a4b3b26e35ad'
            elif x == 'Tuesday':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Days%2FTuesday.mp4?alt=media&token=670cf793-7e33-450c-b8c6-0668ea5daba6'
            elif x == 'Wednesday':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Days%2FWednesday.mp4?alt=media&token=ae58952b-757b-4c76-94fd-ff1b073471ec'
            elif x == 'Thursday':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Days%2FThursday.mp4?alt=media&token=96709b09-04e8-47cf-98a5-6debc472d667'
            elif x == 'Friday':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Days%2FFriday.mp4?alt=media&token=28f424d2-316a-407c-a0c4-d809b2d0bd48'
            elif x == 'Saturday':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Days%2FSaturday.mp4?alt=media&token=a8ce9458-961a-4a2b-bb3a-ed785d85afad'
            elif x == 'Sunday':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Days%2FSunday.mp4?alt=media&token=0ea8ed08-b636-4a29-93e1-bbb46cb1f4c3'
                    

        return (render_template('index.html',value=x, mode=mode))



@app.route('/prediction',methods=['POST'])
def prediction():

    json_file = open("model-bw.json", "r")
    model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(model_json)
    # load weights into new model
    loaded_model.load_weights("model-bw.h5")
    print("Loaded model from disk")


    # In[40]:


    cap = cv2.VideoCapture(0)


    # Category dictionary

    # In[41]:


    #categories = {A: 'A', B: 'B', C: 'C', D: 'D', E: 'E', F: 'FIVE', G: 'G', H: 'H', I: 'I', J: 'J', K: 'K', L: 'L',
    #            M: 'M', N: 'N', O: 'O', P: 'P', Q: 'Q', R: 'R', S: 'S', T: 'T', U: 'U', V: 'V', W: 'W', X: 'X',
    #           Y: 'Y', Z: 'Z'}


    # In[ ]:


    while True:
        _, frame = cap.read()
        # Simulating mirror image
        frame = cv2.flip(frame, 1)
        
        # Got this from collect-data.py
        # Coordinates of the ROI
        x1 = int(0.5*frame.shape[1])
        y1 = 10
        x2 = frame.shape[1]-10
        y2 = int(0.5*frame.shape[1])
        # Drawing the ROI
        # The increment/decrement by 1 is to compensate for the bounding box
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
        # Extracting the ROI
        roi = frame[y1:y2, x1:x2]
        
        # Resizing the ROI so it can be fed to the model for prediction
        roi = cv2.resize(roi, (64, 64)) 
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
        cv2.imshow("test", test_image)
        # Batch of 1
        result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
        prediction = {'A': result[0][0], 
                    'B': result[0][1], 
                    'C': result[0][2],
                    'D': result[0][3],
                    'E': result[0][4],
                    'F': result[0][5],
                    'G': result[0][6],
                    'H': result[0][7],
                    'I': result[0][8],
                    'J': result[0][9],
                    'K': result[0][10],
                    'L': result[0][11],
                    'M': result[0][12],
                    'N': result[0][13],
                    'O': result[0][14],
                    'P': result[0][15],
                    'Q': result[0][16],
                    'R': result[0][17],
                    'S': result[0][18],
                    'T': result[0][19],
                    'U': result[0][20],
                    'V': result[0][21],
                    'W': result[0][22],
                    'X': result[0][23],
                    'Y': result[0][24],
                    'Z': result[0][25]          
                    
                                }
        
        # Sorting based on top prediction
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        
        # Displaying the predictions
        
        cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 5, (0,255,255), 5)    
        cv2.imshow("Frame", frame)
        result = prediction[0][0]
        #print(result)


        with open('alphabet_results','wb') as f:
            pickle.dump(result,f)
        
        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == 27: # esc key
            break

    cap.release()
    cv2.destroyAllWindows()  
        

    with open('alphabet_results','rb') as f:
        x = pickle.load(f)
        #print(x)
        
        if x == 'A':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Alphabet%2FA.mp4?alt=media&token=28622cbb-a3dd-4ba3-bb7e-ba943e14804c'   
        elif x == 'B':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Alphabet%2FB.mp4?alt=media&token=385fd2b7-f75c-4851-81bb-ee14de0fa510'   
        elif x == 'C':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Alphabet%2FC.mp4?alt=media&token=308e45b3-8c75-4705-b0e1-57cdeae3bcfd'
        elif x == 'D':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Alphabet%2FD.mp4?alt=media&token=9c7fdd91-4f51-4284-a667-be2c4074c836'
        elif x == 'E':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Alphabet%2FE.mp4?alt=media&token=2da95c86-ba6c-4d9d-8947-b4bad851571d'
        elif x == 'F':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Alphabet%2FF.mp4?alt=media&token=1b063427-c251-49fd-a3f8-392c27d0aa4b'
        elif x == 'G':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Alphabet%2FG.mp4?alt=media&token=6160f2c3-e4a8-4bc9-9d6d-22bfe87cd644'
        elif x == 'H':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Alphabet%2FH.mp4?alt=media&token=974bb991-eb0b-4a71-bbf7-7cc180ed4284'
        elif x == 'I':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Alphabet%2FI.mp4?alt=media&token=267bc9b1-664b-4d80-9f9c-d4c3eea19e23'
        elif x == 'J':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Alphabet%2FJ.mp4?alt=media&token=bd275a90-2837-4a6e-852c-d48064933611'
        elif x == 'K':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Alphabet%2FK.mp4?alt=media&token=0bf5e818-7c88-4465-b526-370119facddd'
        elif x == 'L':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Alphabet%2FL.mp4?alt=media&token=0eb70d67-9930-4321-863a-95770b97c464'
        elif x == 'M':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Alphabet%2FM.mp4?alt=media&token=4a058da4-6f41-4e33-8bf3-1957a851ad29'
        elif x == 'N':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Alphabet%2FN.mp4?alt=media&token=c62ac6c0-cc00-408e-b5f6-5b90f685bd75'
        elif x == 'O':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Alphabet%2FO.mp4?alt=media&token=307714d8-782c-4a28-88e3-204429723260'
        elif x == 'P':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Alphabet%2FP.mp4?alt=media&token=7b6f6ecc-8d0b-43dd-99f7-b1e147daeb47'
        elif x == 'Q':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Alphabet%2FQ.mp4?alt=media&token=e65eef50-b4f7-4f25-9080-6ba9552f01bc'
        elif x == 'R':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Alphabet%2FR.mp4?alt=media&token=5937af6a-91a8-428a-8d77-36bf81a8126c'
        elif x == 'S':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Alphabet%2FS.mp4?alt=media&token=d087d46d-75f1-42ab-b95d-9b61009825f6'
        elif x == 'T':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Alphabet%2FT.mp4?alt=media&token=dca2ed49-9587-40de-938e-63ca0992085b'
        elif x == 'U':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Alphabet%2FU.mp4?alt=media&token=0a66b3ac-3c92-41c9-987e-f05b0ad10e19'
        elif x == 'V':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Alphabet%2FV.mp4?alt=media&token=8d3d95a7-f39d-4ebd-a343-048afdfed2a4'
        elif x == 'W':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Alphabet%2FW.mp4?alt=media&token=338481e6-de63-4f11-ad3d-e855ffb41256'
        elif x == 'X':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Alphabet%2FX.mp4?alt=media&token=594392b2-1777-4587-9401-d6c0b3c10419'
        elif x == 'Y':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Alphabet%2FY.mp4?alt=media&token=08551fa7-10c3-4502-b687-284ac614fe0e'
        elif x == 'Z':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Alphabet%2FZ.mp4?alt=media&token=52f4d2b7-4ae0-41ec-a314-479cf812c2fc'
        
        
    
    return (render_template('index.html',value=x, mode =mode))

        
    
    return (render_template('index.html',value=x))

    

#res[np.argmax(res)] > threshold


@app.route('/colours',methods=['POST'])
def colours():


    poses=np.array(['Black','Blue','Green','Red','White','Yellow'])

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(poses.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights('colour-dw.h5')
    #res = model.predict(x_test)
        
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities

    def mediapipe_detection(image, model):#keypoint detection
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results

    def draw_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


    def draw_styled_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(40,22,60), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(40,44,121), thickness=2, circle_radius=2)
                                ) 
        # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                ) 
        # Draw right hand connections  
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                ) 

    def extract_keypoints(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        #face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lefthand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        righthand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose,lefthand,righthand])


    # 1. New detection variables
    sequence = []
    sentence = []
    threshold = 0.7

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
    #         sequence.insert(0,keypoints)
    #         sequence = sequence[:30]
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(poses[np.argmax(res)])
                
            
            #3. Viz logic
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if poses[np.argmax(res)] != sentence[-1]:
                            sentence.append(poses[np.argmax(res)])
                    else:
                        sentence.append(poses[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                #image = prob_viz(res, actions, image, colors)
                
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        with open('colour_result.pkl','wb') as sn:
            pickle.dump(sentence[len(sentence)-1],sn)

        #print(sentence)

        with open('colour_result.pkl','rb') as f:
            x = pickle.load(f)
            print(x)

            if x == 'Black':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Colours%2FBlack.mp4?alt=media&token=ee735d59-0cb9-476a-b655-7613e8032441'   
            elif x == 'Blue':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Colours%2FBlue.mp4?alt=media&token=82409e9b-c13b-4451-9b10-ee8a960b8eb0'   
            elif x == 'Green':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Colours%2FGreen.mp4?alt=media&token=a1435ac4-5a9c-40d6-973f-0b8dbb7c3e1c'
            elif x == 'Red':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Colours%2FRed.mp4?alt=media&token=3e4a852a-1c72-4bf6-bb59-90a5c2b8b48f'
            elif x == 'White':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Colours%2Fwhite.mp4?alt=media&token=40bb970d-dcb1-4dbd-850e-2cbe1b365253'
            elif x == 'Yellow':
                mode = 'https://firebasestorage.googleapis.com/v0/b/avatar-b7639.appspot.com/o/Colours%2FYellow.mp4?alt=media&token=bc031c6f-c274-4d00-9d63-7c569ed3ecf1'
            
        

        return (render_template('index.html',value=x, mode=mode))

        #$env:FLASK_APP = "Sign language detection.py"
    #flask run