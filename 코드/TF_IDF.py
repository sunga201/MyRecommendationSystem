from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import file_preprocessor as fpp
import pickle
extend_stopWords=['quite', 'voice', 'did', 'sings', 'truly', 'theres', 'im', 'recording', 'recodings', 'collection', 'work', 'voice', 'did', 'theyre', 'ive', 'buy', 'people', 'wrong', 'greatest', 'laid', 'long', 'minute', 'singing', 'time', 'definitely', 'dont', 'song', 'songs', 'cd', 'album', 'day', 'version', 'albums', 'singer', 'disc', 'feel', 'relax', 'track', 'lyrics', 'got', 'favorate', 'listen', '34', '8217', 'think', 'quot', 'music', 'amazon', 'good', 'great', 'words', 'sound', 'really', 'best', 'like', 'just', 'man', 'love', 'said']
stopWords=list(stop_words.ENGLISH_STOP_WORDS)+extend_stopWords

class Tf_Idf:
    def __init__(self):
        file_pp=fpp.File_pp()
        self.reviews=file_pp.get_review_matrix()
        self.tfidf_list=[]

    def tfidf(self, itemID, show_num=99999):
        test=self.reviews[itemID] #이이템 별 리뷰들을 모아둔다.
        try:
            tfidf=TfidfVectorizer(stop_words=stopWords, min_df=2).fit(test) # 최소한 2개 이상의 리뷰에서 등장한 단어만 체크한다.
        except:
            tfidf = TfidfVectorizer(stop_words=stopWords).fit(test) # 리뷰가 한개인 경우
        testArray=tfidf.transform(test).toarray()
        result=np.zeros(len(testArray[0]))
    
        for list in testArray:
            result+=list
    
        result/=len(testArray)
        result_with_idx=[]
    
        keyList={}
        for k, v in tfidf.vocabulary_.items():
            keyList[v]=k
    
        for i in range(len(testArray[0])):
            result_with_idx.append((result[i], keyList[i]))
    
        result_with_idx.sort(reverse=True)
        ret=[]
        for i in range(min(len(result_with_idx), show_num)):
            if result_with_idx[i][0]>0:
                ret.append([result_with_idx[i][1], result_with_idx[i][0]])
        return ret

    def tfidf_all(self, item_list, show_num=99999):
        try: #tfidf 값을 저장한 리스트 파일이 존재할 경우
            with open('tfidf_list.txt', 'rb') as f:
                self.tfidf_list=pickle.load(f)
            print('loading TF-IDF values...')

        except: #tfidf 값을 저장한 리스트 파일이 존재하지 않을 경우
            print('calculating TF-IDF values...')
            for item in item_list:
                self.tfidf_list.append([item, self.tfidf(item, show_num)])
            with open('tfidf_list.txt', 'wb') as f:
                pickle.dump(self.tfidf_list, f)

    def get_tfidf(self, item, show_num):
        ret=[]
        for element in self.tfidf_list:
            if element[0]==item:
                for i in range(min(len(element[1]), show_num)):
                    ret.append(element[1][i][0])
                return ret