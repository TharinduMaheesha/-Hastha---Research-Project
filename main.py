import video_to_audio as audio
import audio_to_text as text
import text_emotion as text2
from moviepy.editor import VideoFileClip, concatenate_videoclips
import moviepy.editor as mp
import SER_prediction as SER
import video_emotion as video
import shutil , glob
import os
from pytube import YouTube
import pyrebase




def mainFunction(link):

    firebaseConfig = {
  "apiKey": "AIzaSyCefNol1iQM0zR4BofF_cNjtL0RhL_8_rs",
  "authDomain": "hastha-6495f.firebaseapp.com",
  "projectId": "hastha-6495f",
  "storageBucket": "hastha-6495f.appspot.com",
  "messagingSenderId": "19307565630",
  "appId": "1:19307565630:web:57710008ed02dc8c79b76f",
  "measurementId": "G-DY1RPXR7XE",
  "serviceAccount" : "serviceAccount.json",
  "databaseURL" : "https://hastha-6495f-default-rtdb.firebaseio.com/"}

    firebase = pyrebase.initialize_app(firebaseConfig)

    Storage = firebase.storage()
    existing = Storage.list_files()

            
    yt = YouTube(link)
    val = yt.title
    url = yt.embed_url

    output =0

    for file in existing:
        if file.name == str(val+".mp4"):
            print("yes")
            output=1
            break

    if output == 0:
        yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download("static")

        audio.convert_video_to_audio_ffmpeg(val)
        audio.convert_to_wav(val)
        audio.break_video(val+".mp4")
        audio_chunks = audio.break_chunks("static/path.wav")

        for i in range (1,len(audio_chunks)+1):
            frame = 0
            video.FrameCapture("video/"+str(i)+".mp4" , i)

        dictionary = dict()

        for i in range (1,len(audio_chunks)+1):
            happy = 0
            sad = 0
            neutral = 0
            count = 0
            for filename in os.listdir("imagez/"+str(i)):
                count += 1
                result = video.detect_emotion("imagez/"+str(i)+"/"+filename)
                happy+= result['emotion']['happy']
                sad += result['emotion']['sad']
                neutral += result['emotion']['neutral']

            if count == 0:
                happy = 0
                sad = 0
                neutral = 0
                positive = happy 
                negative = sad 
                neutral = neutral

            else:
                positive = happy / count
                negative = sad / count
                neutral = neutral / count
        
            dictionary[str(i)] = [positive , negative , neutral]

            print("Positive : " + str(positive) + " Negative : " + str(negative) + " Neutral : " + str(neutral))

        audio_emotions = SER.predict_emotion(audio_chunks)
        final_text , test= text.test(audio_chunks)

        text_emotions = list()

        for i in test:
            emo = text2.detect_emotion(i)
            text_emotions.append(emo)

        facial_emotions = dict()

        final_emotion = list()
        for i in range(1,len(audio_chunks)+1):
            index = i-1
            pos_aud = 0
            neg_aud = 0
            neu_aud = 0
            if audio_emotions[index] == "positive":
                pos_aud = 0.1
            elif audio_emotions[index] == "negative":
                neg_aud = 0.1
            else:
                neu_aud = 0.1

            positive = (dictionary[str(i)][0] * 0.2) + pos_aud + (text_emotions[index][0] * 70)
            negative = (dictionary[str(i)][1] * 0.2) + neg_aud + (text_emotions[index][1] * 70)
            neutral = (dictionary[str(i)][2] * 0.2) + neu_aud + (text_emotions[index][2] * 70)

            final_emotion.append([positive , negative , neutral])


        emotion_index = list()
        for i in final_emotion:
            max_index = i.index(max(i))
            if max_index == 0:
                emotion_index.append("positive")
            elif max_index == 1:
                emotion_index.append("negative")
            else:
                emotion_index.append("neutral")

        import text_to_SSL as SSL
        SSL_text = list()

        for i in test:
            parsed = SSL.parse_text(i)
            lemmatized = SSL.word_lemmatization(parsed)
            tokenized = SSL.word_tokenization(lemmatized)
            SSL_text.append(tokenized)
            print(tokenized)
            
        for i in range(0,len(emotion_index)):
            if "field" not in SSL_text[i] and "empty" not in SSL_text[i]:
                print("Emotion : " + emotion_index[i] + " --- text : " , SSL_text[i])
                print(" ")

        try:
            os.remove("my_concatenation.mp4")
        except:
            pass

        path = "videos"
        arg_array=[]

        for i in range(0,len(emotion_index)):
            if "field" not in SSL_text[i] and "empty" not in SSL_text[i]:
                for tex in SSL_text[i]:
                    if os.path.exists(path+"/"+emotion_index[i]+"/"+tex+".mp4"):
                        arg_array.append(VideoFileClip(path+"/"+emotion_index[i]+"/"+tex+".mp4"))
                        print(tex+".mp4")
                    else:
                        for t in range(0,len(tex)):
                            arg_array.append(VideoFileClip(path+"/"+emotion_index[i]+"/letters/"+tex[t]+".mp4"))
                            print(tex[t])

                    arg_array.append(VideoFileClip(path+"/whitespace2022.mp4"))


        print(arg_array[0])
        final_clip = concatenate_videoclips(arg_array , method='compose')
        final_clip.write_videofile("static/my_concatenation.mp4")
        Storage.child(val+".mp4").put("static/my_concatenation.mp4")
        finale = Storage.child(val+".mp4").get_url(val+".mp4")

        files_1 = glob.glob('imagez/*')
        for f in files_1:
            shutil.rmtree(f)

        files_2 = glob.glob('video/*')
        for f in files_2:
            os.remove(f)

        files_3 = glob.glob('audio-chunks/*')
        for f in files_3:
            os.remove(f)

    else:
        print("Found")
        finale = Storage.child(val+".mp4").get_url(val+".mp4")

    return finale , url