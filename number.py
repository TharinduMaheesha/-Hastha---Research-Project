#from crypt import methods
from urllib import request
from flask import Blueprint , render_template,request
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from matplotlib import pyplot as plt
import mediapipe as mp
from tensorflow.keras.models import Sequential#to build sequencial neural network
from tensorflow.keras.layers import LSTM, Dense #to perform action detection
from tensorflow.keras.callbacks import TensorBoard#to monitor and traise model
import cv2

number = Blueprint("numbers",__name__,template_folder="templates")

@number.route("/")
def num():
    return (render_template('numbers.html'))

@number.route("/learnnumbers_1_10")
def learn_numbers1_10():
    return (render_template('learn_numbers/learn_numbers_1_10.html'))

@number.route("/learnnumbers_11_20")
def learn_numbers11_20():
    return (render_template('learn_numbers/learn_numbers_11_20.html'))

@number.route("/select_1_10")
def num_1_10():
    return (render_template('numbers_1_10.html'))
   
@number.route("/select_11_20")
def num_11_20():
    return (render_template('numbers_11_20.html'))

@number.route('/num_1_10/numberck',methods=['POST'])
def numbers_check_1_10():
    name=request.form['number']
    poses=np.array(['One', 'Two','Three', 'Four', 'Five','Six' ,'Seven','Eight','Nine','Ten'])

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(poses.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights('Numbers_1_10.h5')
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
        evaluation='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/bad.mp4?alt=media&token=055e9110-5dde-4a25-817c-ee1d05e71515'
        score=0
        for i in sentence:
            if name==i:
                status='correct';
                evaluation='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Good.mp4?alt=media&token=b2b89255-e83e-4caa-a37f-da9285dde240'
                score=100;
                break;
        
        if name == 'One':
            imgnum='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Fone.png?alt=media&token=2782c335-9bfc-40dc-9f55-3176b7b7a50c'
            return (render_template('number_results_2.html',value=status,score=score,number=name,imgnum=imgnum,evaluation=evaluation))
        elif name == 'Two':
            imgnum='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Ftwo.png?alt=media&token=5f1cb695-787c-4971-924d-244e499d394c'
            return (render_template('number_results_2.html',value=status,score=score,number=name,imgnum=imgnum,evaluation=evaluation))
        elif name == 'Three':
            imgnum='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Fthree.png?alt=media&token=690d1bc8-0e86-43a8-b0f2-acfc59a49453'
            return (render_template('number_results_2.html',value=status,score=score,number=name,imgnum=imgnum,evaluation=evaluation))
        elif name == 'Four':
            imgnum='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Ffour.png?alt=media&token=10f5c72d-cc5a-48c4-987b-d73d5b0af472'
            return (render_template('number_results_2.html',value=status,score=score,number=name,imgnum=imgnum,evaluation=evaluation))
        elif name == 'Five':
            imgnum='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Ffive.png?alt=media&token=700ee575-21d1-4042-a254-b6b8d7e923b1'
            return (render_template('number_results_2.html',value=status,score=score,number=name,imgnum=imgnum,evaluation=evaluation))
        elif name == 'Six':
            imgnum='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Fsix.png?alt=media&token=8ba04620-af3f-483d-b6f3-d788b1a1dbe4'
            return (render_template('number_results_2.html',value=status,score=score,number=name,imgnum=imgnum,evaluation=evaluation))
        elif name == 'Seven':
            imgnum='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Fseven.png?alt=media&token=0da63a9b-a6fd-4b57-a41f-0a4fb42dc15e'
            return (render_template('number_results_2.html',value=status,score=score,number=name,imgnum=imgnum,evaluation=evaluation))
        elif name == 'Eight':
            imgnum='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Feight.png?alt=media&token=ce589259-b98e-4786-9880-b437edac851c'
            return (render_template('number_results_2.html',value=status,score=score,number=name,imgnum=imgnum,evaluation=evaluation))
        elif name == 'Nine':
            imgnum='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Fnine.png?alt=media&token=0783e32f-e348-4640-8bb5-23b62295a62c' 
            return (render_template('number_results_2.html',value=status,score=score,number=name,imgnum=imgnum,evaluation=evaluation))
        elif name == 'Ten':
            video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Ften.mp4?alt=media&token=00633fb4-3ba4-421d-8fb0-058de122a595'
            return (render_template('number_results_1.html',value=status,score=score,number=name,video=video,evaluation=evaluation))     

@number.route('/num_11_20/numberck',methods=['POST'])
def numbers_check_11_20():
    name=request.form['number']
    poses=np.array(['Eleven', 'Twelve','Thirteen', 'Fourteen', 'Fifteen','Sixteen' ,'Seventeen','Eighteen','Nineteen','Twenty'])

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(poses.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights('Numbers_11_20.h5')
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
        evaluation='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/bad.mp4?alt=media&token=055e9110-5dde-4a25-817c-ee1d05e71515'
        score=0
        for i in sentence:
            if name==i:
                status='correct';
                evaluation='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Good.mp4?alt=media&token=b2b89255-e83e-4caa-a37f-da9285dde240'
                score=100;
                break;
        print(name)

        if name == 'Eleven':
            video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Felevan%20zoom.mp4?alt=media&token=3a2ca1b2-a2ab-4248-bba6-063ea4f73651'
            return (render_template('number_results_1.html',value=status,score=score,number=name,video=video,evaluation=evaluation))
        elif name == 'Twelve':
            video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Ftwelve.mp4?alt=media&token=b3a60848-0b41-4687-9a58-82d503820bf5'
            return (render_template('number_results_1.html',value=status,score=score,number=name,video=video,evaluation=evaluation))
        elif  name == 'Thirteen':
            video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Fthirteen%20zoom.mp4?alt=media&token=ed38d60c-3503-498b-8c3d-e466fcd732a5'
            return (render_template('number_results_1.html',value=status,score=score,number=name,video=video,evaluation=evaluation))
        elif name == 'Fourteen':
            video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Ffourteen.mp4?alt=media&token=6ee42c88-3f16-451a-b491-918abbf1ff55'
            return (render_template('number_results_1.html',value=status,score=score,number=name,video=video,evaluation=evaluation))
        elif name == 'Fifteen':
            video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Ffifteen%20zoom.mp4?alt=media&token=ed46ef0d-439d-4bbd-92da-daf12f3bd936'
            return (render_template('number_results_1.html',value=status,score=score,number=name,video=video,evaluation=evaluation))
        elif name == 'Sixteen':
            video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Fsixteen%20zoom.mp4?alt=media&token=05f3e1a6-29a0-4af7-97a7-c8c4ce63880a'
            return (render_template('number_results_1.html',value=status,score=score,number=name,video=video,evaluation=evaluation))
        elif name == 'Seventeen':
            video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Fseventeen%20zoom.mp4?alt=media&token=0dbbc53c-64d7-4284-bcb4-39acf991ba20'
            return (render_template('number_results_1.html',value=status,score=score,number=name,video=video,evaluation=evaluation))
        elif name == 'Eighteen':
            video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Feighteen%20zoom.mp4?alt=media&token=129365d1-866b-434c-9e58-b6a7b0225392'
            return (render_template('number_results_1.html',value=status,score=score,number=name,video=video,evaluation=evaluation))
        elif name == 'Nineteen':
            video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Fnineteen%20zoom.mp4?alt=media&token=65e5acf6-98ed-43d7-b8d2-a6451dd62965'
            return (render_template('number_results_1.html',value=status,score=score,number=name,video=video,evaluation=evaluation))
        elif name == 'Twenty':
            video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Ftwenty%20zoom.mp4?alt=media&token=59957940-6237-4d20-af7c-b847d88082a8'
            return (render_template('number_results_1.html',value=status,score=score,number=name,video=video,evaluation=evaluation))
        
            
@number.route('/numberlearn',methods=['POST'])
def number_learn():
    name=request.form['number']
    if name == 'One':
        imgnum='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Fone.png?alt=media&token=2782c335-9bfc-40dc-9f55-3176b7b7a50c'
        return (render_template('learn_numbers/learn_numbers_2.html',number=name,imgnum=imgnum))
    elif name == 'Two':
        imgnum='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Ftwo.png?alt=media&token=5f1cb695-787c-4971-924d-244e499d394c'
        return (render_template('learn_numbers/learn_numbers_2.html',number=name,imgnum=imgnum))
    elif name == 'Three':
        imgnum='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Fthree.png?alt=media&token=690d1bc8-0e86-43a8-b0f2-acfc59a49453'
        return (render_template('learn_numbers/learn_numbers_2.html',number=name,imgnum=imgnum))
    elif name == 'Four':
        imgnum='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Ffour.png?alt=media&token=10f5c72d-cc5a-48c4-987b-d73d5b0af472'
        return (render_template('learn_numbers/learn_numbers_2.html',number=name,imgnum=imgnum))
    elif name == 'Five':
        imgnum='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Ffive.png?alt=media&token=700ee575-21d1-4042-a254-b6b8d7e923b1'
        return (render_template('learn_numbers/learn_numbers_2.html',number=name,imgnum=imgnum))
    elif name == 'Six':
        imgnum='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Fsix.png?alt=media&token=8ba04620-af3f-483d-b6f3-d788b1a1dbe4'
        return (render_template('learn_numbers/learn_numbers_2.html',number=name,imgnum=imgnum))
    elif name == 'Seven':
        imgnum='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Fseven.png?alt=media&token=0da63a9b-a6fd-4b57-a41f-0a4fb42dc15e'
        return (render_template('learn_numbers/learn_numbers_2.html',number=name,imgnum=imgnum))
    elif name == 'Eight':
        imgnum='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Feight.png?alt=media&token=ce589259-b98e-4786-9880-b437edac851c'
        return (render_template('learn_numbers/learn_numbers_2.html',number=name,imgnum=imgnum))
    elif name == 'Nine':
        imgnum='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Fnine.png?alt=media&token=0783e32f-e348-4640-8bb5-23b62295a62c' 
        return (render_template('learn_numbers/learn_numbers_2.html',number=name,imgnum=imgnum))
    elif name == 'Ten':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Ften.mp4?alt=media&token=00633fb4-3ba4-421d-8fb0-058de122a595'
        return (render_template('learn_numbers/learn_numbers_3.html',number=name,video=video))     
    elif name == 'Eleven':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Felevan%20zoom.mp4?alt=media&token=3a2ca1b2-a2ab-4248-bba6-063ea4f73651'
        return (render_template('learn_numbers/learn_numbers_3.html',number=name,video=video))
    elif name == 'Twelve':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Ftwelve.mp4?alt=media&token=b3a60848-0b41-4687-9a58-82d503820bf5'
        return (render_template('learn_numbers/learn_numbers_3.html',number=name,video=video))
    elif  name == 'Thirteen':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Fthirteen%20zoom.mp4?alt=media&token=ed38d60c-3503-498b-8c3d-e466fcd732a5'
        return (render_template('learn_numbers/learn_numbers_3.html',number=name,video=video))
    elif name == 'Fourteen':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Ffourteen.mp4?alt=media&token=6ee42c88-3f16-451a-b491-918abbf1ff55'
        return (render_template('learn_numbers/learn_numbers_3.html',number=name,video=video))
    elif name == 'Fifteen':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Ffifteen%20zoom.mp4?alt=media&token=ed46ef0d-439d-4bbd-92da-daf12f3bd936'
        return (render_template('learn_numbers/learn_numbers_3.html',number=name,video=video))
    elif name == 'Sixteen':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Fsixteen%20zoom.mp4?alt=media&token=05f3e1a6-29a0-4af7-97a7-c8c4ce63880a'
        return (render_template('learn_numbers/learn_numbers_3.html',number=name,video=video))
    elif name == 'Seventeen':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Fseventeen%20zoom.mp4?alt=media&token=0dbbc53c-64d7-4284-bcb4-39acf991ba20'
        return (render_template('learn_numbers/learn_numbers_3.html',number=name,video=video))
    elif name == 'Eighteen':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Feighteen%20zoom.mp4?alt=media&token=129365d1-866b-434c-9e58-b6a7b0225392'
        return (render_template('learn_numbers/learn_numbers_3.html',number=name,video=video))
    elif name == 'Nineteen':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Fnineteen%20zoom.mp4?alt=media&token=65e5acf6-98ed-43d7-b8d2-a6451dd62965'
        return (render_template('learn_numbers/learn_numbers_3.html',number=name,video=video))
    elif name == 'Twenty':
        video='https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Numbers%2Ftwenty%20zoom.mp4?alt=media&token=59957940-6237-4d20-af7c-b847d88082a8'
        return (render_template('learn_numbers/learn_numbers_3.html',number=name,video=video))
        
    