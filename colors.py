from flask import Blueprint , render_template,request,url_for
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from matplotlib import pyplot as plt
import mediapipe as mp
from tensorflow.keras.models import Sequential#to build sequencial neural network
from tensorflow.keras.layers import LSTM, Dense #to perform action detection
from tensorflow.keras.callbacks import TensorBoard#to monitor and traise model
import cv2

colors = Blueprint("colors",__name__,template_folder="templates")

@colors.route("/")
def color():
    return (render_template('colors.html'))
   
@colors.route("/colorselect1")
def color_select1():
    return (render_template('color_ck_l1.html'))

@colors.route("/colorselect2")
def color_select2():
    return (render_template('colors_l2_selection.html'))

@colors.route("/learncolors")
def learn_colors():
    return (render_template('learn_colors/learn_colors.html'))

@colors.route("/learncolors_light")
def learn_color_light():
    return (render_template('learn_colors/learn_color_light.html'))

@colors.route("/learncolors_dark")
def learn_color_dark():
    return (render_template('learn_colors/learn_color_dark.html'))

@colors.route("/lightselection")
def light_selection():
    return (render_template('color_ck_l2_light.html'))

@colors.route("/darkselection")
def dark_selection():
    return (render_template('color_ck_l2_dark.html'))

@colors.route('/colorselect1/colorck',methods=['POST'])
def colors_result1():
    name=request.form['color']
    poses=np.array(['Black', 'Blue', 'Brown','Grey' ,'Green','Orange','Red','Yellow','White'])

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(poses.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights('Colors.h5')
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
        status='incorrect'
        score=0
        evaluation='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/bad.mp4?alt=media&token=055e9110-5dde-4a25-817c-ee1d05e71515'
        
        for i in sentence:
            if name==i:
                status='correct';
                score=100;
                evaluation='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Good.mp4?alt=media&token=b2b89255-e83e-4caa-a37f-da9285dde240'
                break;
        if name == 'Black':
            video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fblack.mp4?alt=media&token=27a70c2c-094c-4c12-a8cf-ac9d0c4f858c"
        elif name == 'Blue':
            video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fblue%20zoom.mp4?alt=media&token=012d73df-037d-4752-963a-9fb5dc5e7e70"
        elif name == 'Green':
            video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fgreen.mp4?alt=media&token=262ddd8a-f985-4008-9f79-dcbc528cb2c5"     
        elif'Brown':
            video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fbrown%20zoom.mp4?alt=media&token=d65a2608-4fcd-4a8c-9286-fa80bc6b0df0"
        elif name == 'Grey':
            video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fgrey%20zoom.mp4?alt=media&token=66dde5e7-1a43-44f3-af00-2e165d82b47c"
        elif name == 'Orange':
            video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Forange%20zoom.mp4?alt=media&token=531ad1ae-5b34-4630-a782-07ab1359b1c8"
        elif name == 'Red':
            video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fred%20zoom.mp4?alt=media&token=a04c47bd-a38a-4e08-9ec5-514c8f53b789"
        elif name == 'Yellow':
            video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fyellow%20zoom.mp4?alt=media&token=c7a2df16-102f-4cd3-adb8-0d90b32ce982"
        elif name == 'White':
            video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fwhite%20zoom.mp4?alt=media&token=faba627b-dd46-454c-95e6-6dc49dc52e77"
        return (render_template('color_results.html',value=status,score=score,color=name,video=video,evaluation=evaluation))

    
        
    
        

@colors.route('/lightselection/colorck',methods=['POST'])
def colors_result2_light():
    name=request.form['color']
    poses=np.array(['Light', 'Dark','Black', 'Blue', 'Brown','Grey' ,'Green','Orange','Red','Yellow','White'])

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(poses.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights('Dark_Ligt_Colors.h5')
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
        score=0
        templist=[]
        sentence = set(dict.fromkeys(sentence))#remove duplicates
        print(templist)
        for i in sentence:
            if i == 'Light'or i == name:
                if len(templist)==0:
                    templist.insert(0,i)
                else:
                    templist.insert(1,i)
        if len(templist)==0:#no elements in the list
            pose1='incorrect'
            pose2='incorrect'
        elif len(templist)==1:#one element in the list
            if templist[0]=='Light':
                pose1='correct'
                pose2='incorrect'
            elif templist[0]==name:
                pose2='correct'
                pose1='incorrect'
        else :#two elements in the list
            if templist[0]=='Light':
                pose1='correct'
            if templist[1]==name:
                pose2='correct'
            elif templist[0]==name:
                pose2='correct'
                pose1='incorrect'
        
        if pose1=='correct' and pose2 == 'correct':
            score=100
        elif pose1=='correct' or pose2 == 'correct':
            score=50
        else:
            score=0
        print(templist)
        print(pose1)
        print(pose2)
        if pose1 == 'incorrect':
            pose1Evaluation='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/bad.mp4?alt=media&token=055e9110-5dde-4a25-817c-ee1d05e71515'
        elif pose1=='correct':
            pose1Evaluation='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Good.mp4?alt=media&token=b2b89255-e83e-4caa-a37f-da9285dde240'
        if pose2 =='incorrect':
            pose2Evaluation='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/bad.mp4?alt=media&token=055e9110-5dde-4a25-817c-ee1d05e71515'
        elif pose2=='correct':
            pose2Evaluation='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Good.mp4?alt=media&token=b2b89255-e83e-4caa-a37f-da9285dde240'
        
        
        lightVideo="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Flight.mp4?alt=media&token=81c99252-c748-4a48-a5ed-87997c41dc1a"
        if name == 'Blue':
            video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fblue%20zoom.mp4?alt=media&token=012d73df-037d-4752-963a-9fb5dc5e7e70"
        elif name == 'Green':
            video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fgreen.mp4?alt=media&token=262ddd8a-f985-4008-9f79-dcbc528cb2c5"     
        elif'Brown':
            video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fbrown%20zoom.mp4?alt=media&token=d65a2608-4fcd-4a8c-9286-fa80bc6b0df0"
        elif name == 'Grey':
            video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fgrey%20zoom.mp4?alt=media&token=66dde5e7-1a43-44f3-af00-2e165d82b47c"
        elif name == 'Orange':
            video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Forange%20zoom.mp4?alt=media&token=531ad1ae-5b34-4630-a782-07ab1359b1c8"
        elif name == 'Red':
            video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fred%20zoom.mp4?alt=media&token=a04c47bd-a38a-4e08-9ec5-514c8f53b789"
        elif name == 'Yellow':
            video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fyellow%20zoom.mp4?alt=media&token=c7a2df16-102f-4cd3-adb8-0d90b32ce982"
        return (render_template('light_color_results.html',score=score,pose1=pose1,pose2=pose2,color=name,video=video,lightVideo=lightVideo,pose1Evaluation=pose1Evaluation,pose2Evaluation=pose2Evaluation))
    

@colors.route('/darkselection/colorck',methods=['POST'])
def colors_result2_dark():
    name=request.form['color']
    poses=np.array(['Light', 'Dark','Black', 'Blue', 'Brown','Grey' ,'Green','Orange','Red','Yellow','White'])

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(poses.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights('Dark_Ligt_Colors.h5')
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
        score=0
        templist=[]
        sentence = set(dict.fromkeys(sentence))#remove duplicates
        print(templist)
        for i in sentence:
            if i == 'Light'or i == name:
                if len(templist)==0:
                    templist.insert(0,i)
                else:
                    templist.insert(1,i)
        if len(templist)==0:#no elements in the list
            pose1='incorrect'
            pose2='incorrect'
        elif len(templist)==1:#one element in the list
            if templist[0]=='Dark':
                pose1='correct'
                pose2='incorrect'
            elif templist[0]==name:
                pose2='correct'
                pose1='incorrect'
        else :#two elements in the list
            if templist[0]=='Dark':
                pose1='correct'
            if templist[1]==name:
                pose2='correct'
            elif templist[0]==name:
                pose2='correct'
                pose1='incorrect'
        
        if pose1=='correct' and pose2 == 'correct':
            score=100
        elif pose1=='correct' or pose2 == 'correct':
            score=50
        else:
            score=0
        print(templist)
        print(pose1)
        print(pose2)
        
        if pose1 == 'incorrect':
            pose1Evaluation='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/bad.mp4?alt=media&token=055e9110-5dde-4a25-817c-ee1d05e71515'
        elif pose1=='correct':
            pose1Evaluation='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Good.mp4?alt=media&token=b2b89255-e83e-4caa-a37f-da9285dde240'
        if pose2 =='incorrect':
            pose2Evaluation='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/bad.mp4?alt=media&token=055e9110-5dde-4a25-817c-ee1d05e71515'
        elif pose2=='correct':
            pose2Evaluation='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Good.mp4?alt=media&token=b2b89255-e83e-4caa-a37f-da9285dde240'
        
        darkImg="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2FDark.png?alt=media&token=363286e6-bb22-4b83-b423-0946bcad86eb"
        if name == 'Blue':
            video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fblue%20zoom.mp4?alt=media&token=012d73df-037d-4752-963a-9fb5dc5e7e70"
        elif name == 'Green':
            video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fgreen.mp4?alt=media&token=262ddd8a-f985-4008-9f79-dcbc528cb2c5"     
        elif'Brown':
            video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fbrown%20zoom.mp4?alt=media&token=d65a2608-4fcd-4a8c-9286-fa80bc6b0df0"
        elif name == 'Grey':
            video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fgrey%20zoom.mp4?alt=media&token=66dde5e7-1a43-44f3-af00-2e165d82b47c"
        elif name == 'Orange':
            video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Forange%20zoom.mp4?alt=media&token=531ad1ae-5b34-4630-a782-07ab1359b1c8"
        elif name == 'Red':
            video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fred%20zoom.mp4?alt=media&token=a04c47bd-a38a-4e08-9ec5-514c8f53b789"
        elif name == 'Yellow':
            video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fyellow%20zoom.mp4?alt=media&token=c7a2df16-102f-4cd3-adb8-0d90b32ce982"
        return (render_template('dark_color_results.html',score=score,pose1=pose1,pose2=pose2,color=name,video=video,darkImg=darkImg,pose1Evaluation=pose1Evaluation,pose2Evaluation=pose2Evaluation))

@colors.route('/colorselect1/learncolors',methods=['POST'])
def colorlearn():
    name=request.form['color']
    if name == 'Black':
        video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fblack.mp4?alt=media&token=27a70c2c-094c-4c12-a8cf-ac9d0c4f858c"
        videozoom=''
    elif name == 'Blue':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fblue.mp4?alt=media&token=7d21a18a-8e2d-488a-a346-32014db9c2d3'
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fblue%20zoom.mp4?alt=media&token=012d73df-037d-4752-963a-9fb5dc5e7e70"
    elif name == 'Green':
        video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fgreen.mp4?alt=media&token=262ddd8a-f985-4008-9f79-dcbc528cb2c5"     
        videozoom='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fgreen%20zoom.mp4?alt=media&token=b6c5fbfe-25fa-4bf5-9863-2905b152dd6d'
    elif'Brown':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fbrown.mp4?alt=media&token=e9bd0a09-da8d-4e10-8d89-23b589996d14'
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fbrown%20zoom.mp4?alt=media&token=d65a2608-4fcd-4a8c-9286-fa80bc6b0df0"
    elif name == 'Grey':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fgrey.mp4?alt=media&token=284bf3d6-144f-48dc-b84d-fd202f7494e5'
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fgrey%20zoom.mp4?alt=media&token=66dde5e7-1a43-44f3-af00-2e165d82b47c"
    elif name == 'Orange':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Forange.mp4?alt=media&token=0c68a9d1-03cd-4e8f-b921-fd0073144c65'
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Forange%20zoom.mp4?alt=media&token=531ad1ae-5b34-4630-a782-07ab1359b1c8"
    elif name == 'Red':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fred.mp4?alt=media&token=8f143f69-b8ba-407b-96ae-60017ec24bc7'
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fred%20zoom.mp4?alt=media&token=a04c47bd-a38a-4e08-9ec5-514c8f53b789"
    elif name == 'Yellow':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fyellow.mp4?alt=media&token=8adf506c-080e-4da3-a127-9881459b9f24'
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fyellow%20zoom.mp4?alt=media&token=c7a2df16-102f-4cd3-adb8-0d90b32ce982"
    elif name == 'White':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fwhite.mp4?alt=media&token=aa46523a-7fd4-4e09-960f-68f7b6b630cf'
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fwhite%20zoom.mp4?alt=media&token=faba627b-dd46-454c-95e6-6dc49dc52e77"
    return (render_template('learn_colors/learn_color.html',color=name,video=video,videozoom=videozoom))
            
@colors.route('/darkselection/colorlearn',methods=['POST'])
def colorlearn_dark():
    name=request.form['color']
    darkImg="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2FDark.png?alt=media&token=363286e6-bb22-4b83-b423-0946bcad86eb"
    if name == 'Blue':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fblue.mp4?alt=media&token=7d21a18a-8e2d-488a-a346-32014db9c2d3'
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fblue%20zoom.mp4?alt=media&token=012d73df-037d-4752-963a-9fb5dc5e7e70"
    elif name == 'Green':
        video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fgreen.mp4?alt=media&token=262ddd8a-f985-4008-9f79-dcbc528cb2c5"     
        videozoom='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fgreen%20zoom.mp4?alt=media&token=b6c5fbfe-25fa-4bf5-9863-2905b152dd6d'
    elif'Brown':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fbrown.mp4?alt=media&token=e9bd0a09-da8d-4e10-8d89-23b589996d14'
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fbrown%20zoom.mp4?alt=media&token=d65a2608-4fcd-4a8c-9286-fa80bc6b0df0"
    elif name == 'Grey':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fgrey.mp4?alt=media&token=284bf3d6-144f-48dc-b84d-fd202f7494e5'
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fgrey%20zoom.mp4?alt=media&token=66dde5e7-1a43-44f3-af00-2e165d82b47c"
    elif name == 'Orange':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Forange.mp4?alt=media&token=0c68a9d1-03cd-4e8f-b921-fd0073144c65'
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Forange%20zoom.mp4?alt=media&token=531ad1ae-5b34-4630-a782-07ab1359b1c8"
    elif name == 'Red':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fred.mp4?alt=media&token=8f143f69-b8ba-407b-96ae-60017ec24bc7'
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fred%20zoom.mp4?alt=media&token=a04c47bd-a38a-4e08-9ec5-514c8f53b789"
    elif name == 'Yellow':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fyellow.mp4?alt=media&token=8adf506c-080e-4da3-a127-9881459b9f24'
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fyellow%20zoom.mp4?alt=media&token=c7a2df16-102f-4cd3-adb8-0d90b32ce982"
    return (render_template('learn_colors/learn_color1.html',color=name,video=video,videozoom=videozoom,darkImg=darkImg))

@colors.route('/lightselection/colorlearn',methods=['POST'])
def colorlearn_light():
    name=request.form['color']
    videoLight="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Flight.mp4?alt=media&token=81c99252-c748-4a48-a5ed-87997c41dc1a"
    if name == 'Blue':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fblue.mp4?alt=media&token=7d21a18a-8e2d-488a-a346-32014db9c2d3'
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fblue%20zoom.mp4?alt=media&token=012d73df-037d-4752-963a-9fb5dc5e7e70"
    elif name == 'Green':
        video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fgreen.mp4?alt=media&token=262ddd8a-f985-4008-9f79-dcbc528cb2c5"     
        videozoom='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fgreen%20zoom.mp4?alt=media&token=b6c5fbfe-25fa-4bf5-9863-2905b152dd6d'
    elif'Brown':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fbrown.mp4?alt=media&token=e9bd0a09-da8d-4e10-8d89-23b589996d14'
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fbrown%20zoom.mp4?alt=media&token=d65a2608-4fcd-4a8c-9286-fa80bc6b0df0"
    elif name == 'Grey':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fgrey.mp4?alt=media&token=284bf3d6-144f-48dc-b84d-fd202f7494e5'
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fgrey%20zoom.mp4?alt=media&token=66dde5e7-1a43-44f3-af00-2e165d82b47c"
    elif name == 'Orange':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Forange.mp4?alt=media&token=0c68a9d1-03cd-4e8f-b921-fd0073144c65'
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Forange%20zoom.mp4?alt=media&token=531ad1ae-5b34-4630-a782-07ab1359b1c8"
    elif name == 'Red':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fred.mp4?alt=media&token=8f143f69-b8ba-407b-96ae-60017ec24bc7'
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fred%20zoom.mp4?alt=media&token=a04c47bd-a38a-4e08-9ec5-514c8f53b789"
    elif name == 'Yellow':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fyellow.mp4?alt=media&token=8adf506c-080e-4da3-a127-9881459b9f24'
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Colors%2Fyellow%20zoom.mp4?alt=media&token=c7a2df16-102f-4cd3-adb8-0d90b32ce982"
    return (render_template('learn_colors/learn_color2.html',color=name,video=video,videoLight=videoLight,videozoom=videozoom))
    