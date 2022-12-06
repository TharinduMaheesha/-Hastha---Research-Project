
import sys
from nltk.tree import *
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import nltk
import re

def parse_text(input_text):

    inputString = " "
    # os.environ['STANFORD_PARSER'] = "static/stanford-parser.jar"
    # os.environ['STANFORD_MODELS'] = "https://drive.google.com/file/d/1KC5llYiLNfZJ4dOLUY5eRQC_TmCYbSRE/view?usp=sharing"
    # os.environ['JAVAHOME'] =  "static/java.exe"


    for each in range(1,len(sys.argv)):
        inputString += sys.argv[each]
        inputString += " "

    inputString = input_text
    inputString = str(inputString).lower()
    inputString = re.sub("[^\w\s]" , ' ' , inputString)

    if len(inputString.split(" ")) < 4:
        inputString += " this that "

    # parser = stanford.StanfordParser(model_path = "static/englishPCFG.ser.gz")
    # englishtree=[tree for tree in parser.parse(inputString.split())]
    # parsetree=englishtree[0]
    # dict={}

    # parenttree= ParentedTree.convert(parsetree)
    # for sub in parenttree.subtrees():
    #     dict[sub.treeposition()]=0

    # isltree=Tree('ROOT',[])
    # i=0
    # for sub in parenttree.subtrees():
    #     if(sub.label()=="NP" and dict[sub.treeposition()]==0 and dict[sub.parent().treeposition()]==0):
    #         dict[sub.treeposition()]=1
    #         isltree.insert(i,sub)
    #         i=i+1
    #     if(sub.label()=="VP" or sub.label()=="PRP"):
    #         for sub2 in sub.subtrees():
    #             if((sub2.label()=="NP" or sub2.label()=='PRP')and dict[sub2.treeposition()]==0 and dict[sub2.parent().treeposition()]==0):
    #                 dict[sub2.treeposition()]=1
    #                 isltree.insert(i,sub2)
    #                 i=i+1


    # for sub in parenttree.subtrees():
    #     for sub2 in sub.subtrees():
    #         if(len(sub2.leaves())==1 and dict[sub2.treeposition()]==0 and dict[sub2.parent().treeposition()]==0):
    #             dict[sub2.treeposition()]=1
    #             isltree.insert(i,sub2)
    #             i=i+1

    # parsed_sent=isltree.leaves()
    parsed_sent = inputString.split(" ")

    return parsed_sent


def word_lemmatization(parsed_sent):

    stop_words = {'a', 'about', 'above', 'after', 'again', 'against', 'ain', 'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being',
    'below', 'between', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during',
    'each', 'few', 'for', 'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't",'haven', "haven't", 'having', 'he', 'her', 'here', 'hers', 'herself', 'him',
    'himself', 'his', 'how', 'if', 'in', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'mightn', "mightn't", 'more', 'most', 'mustn', "mustn't",
    'myself', 'needn', "needn't", 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're',
    's', 'same', 'shan', "shan't", 'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
    'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 'were',
    'weren', "weren't", 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", "you're",
    "you've", 'your', 'yours', 'yourself', 'yourselves' , 'would' , 'like' , "job" , "project"}

    for w in parsed_sent:
        if w in stop_words:
            parsed_sent.remove(w)


    # stop_words=set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    ps = PorterStemmer()
    lemmatized_words=[]
    for w in parsed_sent:
        # w = ps.stem(w)
        lemmatized_words.append(w)


    SSL_sentence = ""
    for w in lemmatized_words:
        if w not in stop_words:
            SSL_sentence+=w
            SSL_sentence+=" "

    return SSL_sentence

def word_tokenization(SSL_sentence):

    name=SSL_sentence

    input_text=name

    text = nltk.word_tokenize(input_text)
    result=nltk.pos_tag(text)

    return text
 


