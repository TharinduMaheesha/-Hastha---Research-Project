from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import os

# Function to extract frames
def FrameCapture(path , i):

    cap = cv2.VideoCapture(path)
    count = 0

    paths = 'imagez/'+str(i)

    # Check whether the specified path exists or not
    isExist = os.path.exists(paths)

    if not isExist:
    
    # Create a new directory because it does not exist 
        os.makedirs(paths)

    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            # Convert into grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h),
                        (0, 0, 255), 2)
            
                faces = frame[y:y + h, x:x + w]
                cv2.imwrite('imagez/'+str(i)+'/frame{:d}.jpg'.format(count), faces)

            count += 30 # i.e. at 30 fps, this advances one second
            cap.set(1, count)
        else:
            cap.release()
            break

def trim():

    x = 0
    paths = 'imagez/'+str(i)

    # Check whether the specified path exists or not
    isExist = os.path.exists(paths)

    if not isExist:
    
    # Create a new directory because it does not exist 
        os.makedirs(paths)
    for i in range(0,1000):
        # Read the input image
        img = cv2.imread('imagez/frame'+str(x)+'.jpg')

        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h),
                        (0, 0, 255), 2)
            
            faces = img[y:y + h, x:x + w]
            cv2.imshow("face",faces)
            cv2.imwrite('faces/face'+str(x)+'.jpg', faces)

        cv2.imshow('img', img)
        cv2.waitKey()

def detect_emotion(img):
    img1 = cv2.imread(img)
    result = DeepFace.analyze(img1 , actions = ['emotion'] , enforce_detection=False)
    # happy = result["happy"]
    # sad = result["sad"]
    # neutral = result["neutral"]

    return result


