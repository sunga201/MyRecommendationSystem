import json
import numpy as np

class File_pp:
    pp=None
    def __init__(self):
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
        
    def get_Instance():
        if File_pp.pp==None:
            File_pp.pp=File_pp()
        return File_pp.pp
    
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

    def cal_item_similarity(self):
        sim_matrix=np.array([[0]*len(self.items)*5], dtype=list).reshape(3568, 5) # 3568 X 5 크기의 배열에 item similarity 정보 저장
        for i in range(len(self.items)):
            print('i : ', i)
            tmp_list=[]
            for j in range(len(self.items)):
                if i==j: tmp_list.append(np.zeros(len(self.items)))
                else:
                    sim=self.get_cosine_similarity(self.rating[i, :], self.rating[j, :])
                    tmp_list.append((j, sim))
            tmp_list.sort(key=lambda x : x[1], reverse=True)
            sim_matrix[i]=tmp_list[:5]
        np.save('sim_matrix.npy', sim_matrix)
        return sim_matrix

    def get_item_similarity(self):
        sim_matrix=[]
        try:
            sim_matrix=np.load('sim_matrix.npy', allow_pickle=True)
        except IOError:
            sim_matrix=self.cal_item_similarity()
        return sim_matrix