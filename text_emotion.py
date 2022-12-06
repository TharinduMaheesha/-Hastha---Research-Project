from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

def cleantext(data):
    data = re.sub(r'@[A-Za-z0-9]+', '', data) # remove @mentions
    data = re.sub(r'#', '', data)# remove # tag
    data = re.sub(r'RT[\s]+', '', data) # remove the RT
    data = re.sub(r'https?:\/\/\S+', '', data) # remove links
    data = re.sub('(\\\\u([a-z]|[0-9])+)', ' ', data) # remove unicode characters
    data = re.sub(r'"', '', data)
    data = re.sub(r':', '', data)
    data = re.sub(r'=', '', data)
    data = re.sub(r'`', '', data)


    return data

def pre_process(text):
    text_1 = text.split()

    lemmatizer = WordNetLemmatizer()
    ps = PorterStemmer()
    lemmatized_words=[]
    for w in t:
        w = ps.stem(w)
        lemmatized_words.append(lemmatizer.lemmatize(w))

        # stop_words=set(stopwords.words("english"))
    islsentence = ""
    for w in lemmatized_words:
        # if w not in stop_words:
        islsentence+=w
        islsentence+=" "

def detect_emotion(text):
    analyser=SentimentIntensityAnalyzer()
    val = analyser.polarity_scores(text)

    emotion = [val["pos"] , val["neg"] , val["neu"]]

    return emotion
