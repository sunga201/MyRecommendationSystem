import json
import numpy as np

class File_pp:
    pp=None
    def __init__(self):
        try:
            self.data = open("Digital_Music_5.json", 'r')
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
        self.rating = np.zeros((items_size, users_size))

        self.data.seek(0)
        for line in self.data:
            k = json.loads(line)
            self.rating[self.items[k['asin']], self.users[k['reviewerID']]] = k['overall']

    def get_rating_matrix(self):
        return self.rating, self.items, self.users
    
    def get_review_matrix(self):
        return self.reviews
