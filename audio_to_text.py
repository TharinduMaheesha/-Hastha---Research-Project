import speech_recognition as sr

# creating object for speech recognition
r = sr.Recognizer()

def test(chunks):
    whole_text = ""
    temp= list()

    for i in range(1,len(chunks)+1):
        with sr.AudioFile("audio-chunks/chunk"+str(i)+".wav") as source:
            audio_listened = r.record(source)
            try:
                # r.adjust_for_ambient_noise(source, duration=0.2)
                text = r.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                # print("Error:", str(e))
                temp.append("This Field is Empty")
            else:
                text = f"{text.capitalize()}. "
                print("chunk"+str(i)+".wav", ":", text)
                temp.append(text)
                whole_text += text

    return whole_text , temp


