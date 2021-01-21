import json
import numpy as np
import sys

class File_pp:
    # 싱글톤 패턴
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"): # _instance 속성이 없다면
            print("init new called!")
            cls._instance = super().__new__(cls) # File_pp 클래스 객체 생성
        return cls._instance # File_pp._instance 리턴

    def __init__(self):
        cls=type(self)
        if not hasattr(cls, "_init"):
            print("init init called!")
            try:
                self.data = open("./reviewData/Digital_Music_5.json", 'r')
            except IOError:
                print("file open failed!")
                sys.exit()

            self.items={}
            self.users={}
            self.reviews={}
            self.rating=None
            self.cal_rating_review_matrix()
            cls._init=True
    
    def cal_rating_review_matrix(self):
        i=0
        j=0
        for line in self.data:
            k = json.loads(line)
            if k['asin'] not in self.items or k['asin'] not in self.reviews:
                if k['asin'] not in self.items:
                    self.items[k['asin']] = i
                    i += 1
                    
                if k['asin'] not in self.reviews:
                    self.reviews[k['asin']] = []
                    
            if k['reviewerID'] not in self.users:
                self.users[k['reviewerID']] = j
                j += 1

            self.reviews[k['asin']].append(k['reviewText'].replace('\'', '').replace('&#', ''))

        items_size = len(self.items)
        users_size = len(self.users)
        try: # 초기 평점 행렬 파일이 있으면 불러오기
            self.rating=np.load('initial_matrix.npy')
            print('Loading initial matrix...')
            
        except: #초기 평점 행렬이 없으면 새로 만들기
            
            print('Calculating intial matrix...')
            self.rating = np.zeros((items_size, users_size))
            self.data.seek(0)
            for line in self.data:
                k = json.loads(line)
                self.rating[self.items[k['asin']], self.users[k['reviewerID']]] = k['overall']

            np.save('initial_matrix.npy', self.rating)

    def get_rating_matrix(self):
        return self.rating, self.items, self.users
    
    def get_review_matrix(self):
        return self.reviews

    def get_cosine_similarity(self, a, b): #a와 b에는 평점 행렬에서 각 아이템 a와 b가 받은 평점들을 가진 리스트가 들어온다.
        return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))