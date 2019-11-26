from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np
import json
extend_stopWords=['disc', 'feel', 'relax', 'track', 'lyrics', 'got', 'favorate', 'listen', '34', '8217', 'think', 'song', 'songs', 'cd', 'album', 'quot', 'music', 'amazon', 'good', 'great', 'words', 'sound', 'really', 'best', 'like', 'just', 'man', 'love', 'said']
stopWords=list(stop_words.ENGLISH_STOP_WORDS)+extend_stopWords
try:
    data=open("Digital_Music_5.json", 'r')
except IOError:
    print("file open failed!")
    sys.exit()

reviews={}
for line in data:
    k = json.loads(line)
    if k['asin'] not in reviews:
        reviews[k['asin']]=[]
    reviews[k['asin']].append(k['reviewText'])

def tfidf(itemID, showNum=99999):
    test=reviews[itemID] #이이템 별 리뷰들을 모아둔다.
    tfidf=TfidfVectorizer(stop_words=stopWords).fit(test)
    testArray=tfidf.transform(test).toarray()
    result=np.zeros(len(testArray[0]))

    for list in testArray:
        result+=list

    result/=len(testArray)
    resultWithIdx=[]

    keyList={}
    for k, v in tfidf.vocabulary_.items():
        keyList[v]=k

    for i in range(len(testArray[0])):
        resultWithIdx.append((result[i], keyList[i]))

    resultWithIdx.sort(reverse=True)
    ret=[]
    for i in range(min(len(resultWithIdx), showNum)):
        ret.append(resultWithIdx[i][1])
    return ret