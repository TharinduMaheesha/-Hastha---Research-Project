from __future__ import unicode_literals
import pickle
import soundfile
import librosa
import numpy as np

filename = 'modelForPrediction2.sav'
loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

def predict_emotion(audio_chunks):
    temp = list()
    for i in range(1,len(audio_chunks)+1):
        feature=extract_feature("audio-chunks/chunk"+str(i)+".wav", mfcc=True, chroma=True, mel=True)
        feature=feature.reshape(1,-1)
        prediction=loaded_model.predict(feature)
        temp.append(prediction[0])

    return temp


