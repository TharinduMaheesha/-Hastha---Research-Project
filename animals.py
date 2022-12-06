from flask import Blueprint , render_template,request
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from matplotlib import pyplot as plt
import mediapipe as mp
from tensorflow.keras.models import Sequential#to build sequencial neural network
from tensorflow.keras.layers import LSTM, Dense #to perform action detection
from tensorflow.keras.callbacks import TensorBoard#to monitor and traise model
import cv2

animals = Blueprint("animals",__name__,template_folder="templates")

@animals.route("/")
def animal():
    return (render_template('animal_selection.html'))

@animals.route("/animalselection")
def animal_selection():
    return (render_template('animals.html'))

@animals.route("/learnanimals")
def learn_animals():
    return (render_template('learn_animals/learn_animals.html'))

@animals.route('/animalck',methods=['POST'])
def animals_result():
    name=request.form['animal']
    poses = np.array(['Tusker', 'Tortise', 'crab','fox' ,'dog','spider'])

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(poses.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights('AnimalSignModel.h5')
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
    if name == 'fox':
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Animals%2Ffox%20zoom.mp4?alt=media&token=b86459fb-18a9-4bb5-ad4f-9a5aaa437da0"
    elif name == 'Tusker':
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Animals%2FTusker%20zoom.mp4?alt=media&token=b5322090-c1b3-4c57-91a5-f9839c5b19d4"
    elif name == 'Tortise':
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Animals%2FTortise%20zoom.mp4?alt=media&token=abfdd043-ba42-4bd8-b871-a10a928179b8"
    elif name == 'crab':
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Animals%2Fcrab%20zoom.mp4?alt=media&token=ae55ccd9-52a1-4317-848c-0332e1532816"
    elif name == 'dog':
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Animals%2Fdog%20zoom.mp4?alt=media&token=8b88fa68-52d1-4845-8756-aa70f45ac7ba"
    elif name == 'spider':
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Animals%2Fspider%20zoom.mp4?alt=media&token=f657479b-c24f-4140-9404-e7578cf2bfdf"
        
    return (render_template('animal_results_1.html',value=status,score=score,animal=name,videozoom=videozoom,evaluation=evaluation))
       

@animals.route('/animallearn',methods=['POST'])
def animals_learn():
    name=request.form['animal']
    if name == 'fox':
        video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Animals%2Ffox.mp4?alt=media&token=23a26a5a-2649-4186-811b-8fc893c12200"
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Animals%2Ffox%20zoom.mp4?alt=media&token=b86459fb-18a9-4bb5-ad4f-9a5aaa437da0"
    elif name == 'Tusker':
        video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Animals%2FTusker.mp4?alt=media&token=94b759e3-b1ff-4644-b821-75a3c2f56cce"
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Animals%2FTusker%20zoom.mp4?alt=media&token=b5322090-c1b3-4c57-91a5-f9839c5b19d4"
    elif name == 'Tortise':
        video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Animals%2FTortise.mp4?alt=media&token=f511a9b5-6aab-43a1-8def-1ae5b7ea1496"
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Animals%2FTortise%20zoom.mp4?alt=media&token=abfdd043-ba42-4bd8-b871-a10a928179b8"
    elif name == 'crab':
        video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Animals%2Fcrab.mp4?alt=media&token=cf1b2156-eff0-45f6-9376-db24ea46b182"
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Animals%2Fcrab%20zoom.mp4?alt=media&token=ae55ccd9-52a1-4317-848c-0332e1532816"
    elif name == 'dog':
        video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Animals%2Fdog.mp4?alt=media&token=8bc66e30-86d3-48a4-a828-ef3059f3c8d8"
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Animals%2Fdog%20zoom.mp4?alt=media&token=8b88fa68-52d1-4845-8756-aa70f45ac7ba"
    elif name == 'spider':
        video="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Animals%2Fspider.mp4?alt=media&token=fb88cdb6-a369-4c5e-96e5-a9dd36b86b15"
        videozoom="https://firebasestorage.googleapis.com/v0/b/hastha-50a83.appspot.com/o/Animals%2Fspider%20zoom.mp4?alt=media&token=f657479b-c24f-4140-9404-e7578cf2bfdf"
    return (render_template('learn_animals/learn_animals_1.html',animal=name,video=video,videozoom=videozoom))
    
