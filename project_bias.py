import sys
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
import time

class Make_Rating_Matrix:
    def __init__(self):
        try:
            self.data = open("Digital_Music_5.json", 'r')
        except IOError:
            print("file open failed!")
            sys.exit()

    def get_rating_matrix(self):
        items={}
        users={}
        i=0
        j=0
        for line in self.data:
            k = json.loads(line)
            if k['asin'] not in items:
                items[k['asin']] = i
                i += 1
            if k['reviewerID'] not in users:
                users[k['reviewerID']] = j
                j += 1

        items_size = len(items)
        users_size = len(users)
        rating = np.zeros((items_size, users_size))

        self.data.seek(0)
        for line in self.data:
            k = json.loads(line)
            rating[items[k['asin']], users[k['reviewerID']]] = k['overall']
        return rating, items, users

class Make_Graph:
    def __init__(self, train_set_error, test_set_error):
        self.train_set_error=train_set_error
        self.test_set_error=test_set_error

        x = np.arange(len(self.train_set_error))
        line1=plt.plot(self.train_set_error, 'r')
        line2=plt.plot(self.test_set_error, 'y')
        plt.grid(axis='both')
        plt.xlabel('epoch')
        plt.ylabel('error')
        plt.xticks(np.arange(0, len(x), step=5))
        print(len(x)//5)
        plt.legend(labels=('training data', 'test data'))

    def show_graph(self):
        plt.show()

class Recommend_Engine_SGD:
    def __init__(self, lambda_q=1, lambda_pt=1, lr=0.0001):
        '''
        :param lambda_q : lambda 1, default value is 1
        :param lambda_pt : lambda 2, default value is 1
        :param lr : sgd's learning rate

        matrix_row_len : row length of rating matrix, num of item
        matrix_col_len : column length of rathing matrix, num of user
        test_set : test set of rating matrix, choose 5,000 nonzero elements
        train_set : training set, replace the selected elements in the test_set with zeros
        total_mean : mean of original rating matrix's non-zero elements
        item_bias : item bias
        user_bias : user bias
        '''
        m=Make_Rating_Matrix()

        self.matrix, self.items, self.users=m.get_rating_matrix()
        self.matrix_row_len=len(self.matrix)
        self.matrix_col_len=len(self.matrix[0])
        self.rating_matrix=np.zeros((1, 1))
        self.train_set, self.test_set = self.make_train_test()
        #self.mat_q, self.mat_pt=cal_SVD(self.train_set)
        self.lr=lr
        self.total_mean=self.get_mean()
        self.item_bias, self.user_bias=self.get_bias()
        self.train_set_error=[]
        self.test_set_error=[]
        self.trained=0 # if rating matrix is trained, variable trained turns to 1.
        self.loaded=0 # if admin load rating matrix, variable loaded turns to 1.

        self.load_SVD()

    # cal_SVD() : Perform SVD with original rating matrix and save the results.
    def cal_SVD(self):
        factor_k =100
        U, s, V = svds(self.matrix, factor_k)
        S = np.zeros((len(s), len(s)))
        for i in range(len(s)):
            S[i][i] = s[i]
        PT = np.dot(S, V)
        np.save('SVD_Q.npy', U)
        np.save('SVD_PT.npy', PT)

    # load_SVD() : load SVD
    def load_SVD(self):
        try:
            self.mat_q = np.load('SVD_Q.npy')
            self.mat_pt = np.load('SVD_PT.npy')
        except IOError:
            print('Make SVD...')
            self.cal_SVD()
            self.load_SVD()
    
    # get_mean() : Returns the average of nonzero elements in the original rating matrix
    def get_mean(self):
        nonzero_row, nonzero_col=self.get_nonzero(self.matrix)
        res=0
        for i in range(len(nonzero_row)):
            row=nonzero_row[i]
            col=nonzero_col[i]
            res+=self.matrix[row, col]
        return res/len(nonzero_row)

    # get_bias() : Returns item bias, user bias
    def get_bias(self):
        i_bias=np.zeros(self.matrix_row_len)
        u_bias=np.zeros(self.matrix_col_len)

        for idx in range(self.matrix_row_len):
            i_bias[idx]=np.mean(self.matrix[idx, self.matrix[idx, :].nonzero()])-self.total_mean

        for idx in range(self.matrix_col_len):
            u_bias[idx] = np.mean(self.matrix[self.matrix[:, idx].nonzero(), idx])-self.total_mean
        return i_bias, u_bias

    '''
        get_nonzero() : Returns two vector that stores indices of nonzero elements of given matrix.
                        nonzero_row : store row index
                        nonzero_col : store column index
                        matrix[nonzero_row[i], nonzero_col[i]] = nonzero element of matrix
    '''
    def get_nonzero(self, matrix):
        nonzero_row, nonzero_col = matrix.nonzero()
        return nonzero_row, nonzero_col

    #make_train_test() : make train set and test set from the original matrix.
    def make_train_test(self):
        train_set=self.matrix.copy()
        test_set=np.zeros(self.matrix.shape)

        nonzero_row, nonzero_col=self.get_nonzero(train_set)
        idx_list=np.random.choice(range(len(nonzero_row)), 5000, replace=False) # num of test set : 5000

        for idx in idx_list:
           test_set[nonzero_row[idx], nonzero_col[idx]]=train_set[nonzero_row[idx], nonzero_col[idx]]
           train_set[nonzero_row[idx], nonzero_col[idx]]=0

        return train_set, test_set

    #train() : Train the training set with SGD.
    def train(self, iter):
        self.train_set_error=[] #clear train set error.
        self.test_set_error=[] #clear test set error.
        if self.trained==1: #if rating matrix is already trained, check
            print('rating matrix is already trained. do you want to train again?(y, n)')
            k=input()
            if k=='n':
                return

        nonzero_row, nonzero_col=self.get_nonzero(self.train_set) #Returns the row number and column number of non-zero elements.
        start=time.time()
        prev_rmse=0.
        for i in range(iter):
            print("iter ", i)
            chk=1
            for idx in range(len(nonzero_row)):
                row=nonzero_row[idx] # Row number of nonzero element in train set.
                col=nonzero_col[idx] # Column number of nonzero element in train set.
                baseline=self.get_baseline(row, col) #set baseline estimate
                err=2*(self.train_set[row, col] - (baseline + np.dot(self.mat_q[row, :], self.mat_pt[:, col])))

                if err > 100000 :
                    print("err exceeded over 100000. change learning rate and try again.")
                    return

                self.mat_q[row, :] += self.lr * (err * self.mat_pt[:, col] - self.mat_q[row, :]) #update q
                self.mat_pt[:, col] += self.lr * (err * self.mat_q[row, :] - self.mat_pt[:, col]) #update pt

                self.item_bias[row] += self.lr * (err - self.item_bias[row]) #update item bias
                self.user_bias[col] += self.lr * (err - self.user_bias[col]) #update user bias

            train_rmse=self.get_rmse(self.train_set)
            test_rmse=self.get_rmse(self.test_set)
                
            print("train set error : ", train_rmse)
            print("test set error : ", test_rmse)

            self.train_set_error.append(train_rmse)
            self.test_set_error.append(test_rmse)

            if abs(prev_rmse - test_rmse) <= 1e-3: # End training when the RMSE difference in test set is less than 0.001.
                print("train finished!")
                print('-'*5, 'reult matrix','-'*5)
                self.rating_matrix=self.get_final_matrix()
                print(self.rating_matrix)
                np.save('rating_matrix.npy', self.rating_matrix)
                np.save('train_set_error.npy', self.train_set_error)
                np.save('test_set_error.npy', self.test_set_error)
                print("time : ", time.time() - start)
                self.trained=1
                return
            prev_rmse=test_rmse

    # get_baseline() : Return baseline estimate for given row number and column number.
    def get_baseline(self, row, col):
        return self.total_mean + self.item_bias[row] + self.user_bias[col]

    # get_final_matrix() : Return learning matrix.
    def get_final_matrix(self):
        tmp=np.full(self.matrix.shape, self.total_mean)
        for i in range(self.matrix_col_len):
            tmp[:, i]+=self.item_bias

        for i in range(self.matrix_row_len):
            tmp[i, :]+=self.user_bias

        tmp+=np.dot(self.mat_q, self.mat_pt)
        return tmp

    #load_rating_matrix() : load rating matrix, train set error, test set error and store them in each variables.
    def load_rating_matrix(self):
        self.rating_matrix=np.load('rating_matrix.npy')
        self.train_set_error=list(np.load('train_set_error.npy'))
        self.test_set_error=list(np.load('test_set_error.npy'))

        if len(self.rating_matrix) > 1:
            self.loaded=1
            return True
        else: return False

    #get_rmse() : return RMSE for given matrix.
    def get_rmse(self, matrix):
        sum=0
        nonzero_row, nonzero_col = self.get_nonzero(matrix)
        pred_matrix=self.get_final_matrix()
        for idx in range(len(nonzero_row)):
            row=nonzero_row[idx]
            col=nonzero_col[idx]
            target=matrix[row, col]
            sum+=(target-pred_matrix[row, col])**2
        sum=np.sqrt(sum/len(nonzero_row))
        return sum

    #show_rating_matrix() : print rating matrix.
    def show_rating_matrix(self):
        if self.loaded==0 and self.trained==0:
            print(self.matrix)
            return

        print(self.rating_matrix)
        print("training set error rate : ", self.train_set_error[-1])
        print("test set error rate : ", self.test_set_error[-1])

    #show_graph() : show error graph.
    def show_graph(self):
        if self.trained==0 and self.loaded==0:
            print("Please train or load the rating matrix.")
            return
        graph=Make_Graph(self.train_set_error ,self.test_set_error)
        graph.show_graph()

    #recommend_item() : recommend k items for user.
    def recommend_item(self):
        userID=input("Enter your ID : ")
        num=int(input("Enter number of items : "))
        user_idx=self.users[userID]
        nonzero_row, nonzero_col=self.get_nonzero(self.matrix)
        testk=[]
        for i in range(len(nonzero_col)):
            if nonzero_col[i]==user_idx:
                row = nonzero_row[i]
                testk.append(row)

        test_list=self.rating_matrix[:, user_idx]
        rec_list=[]
        itemIDs = list(self.items.keys())
        for i in range(len(test_list)):
            rec_list.append([itemIDs[i], test_list[i]])
        for i in testk:
            rec_list[i][1]=0
        rec_list.sort(key=lambda x : x[1], reverse=True)
        for i in range(num):
            print('%d. %s %lf'%(i+1, rec_list[i][0], rec_list[i][1]))