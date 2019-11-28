import sys
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
import time
import random
import file_preprocessor as fpp
from sklearn.preprocessing import MinMaxScaler

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
        plt.legend(labels=('training data', 'test data'))

    def show_graph(self):
        plt.show()

class Recommend_Engine:
    def __init__(self, lambda_q=1, lambda_pt=1, lr=0.0001):
        '''
        :param lambda_q : lambda 1, default value is 1
        :param lambda_pt : lambda 2, default value is 1
        :param lr : sgd's learning rate

        row_len : row length of rating matrix, num of item
        col_len : column length of rathing matrix, num of user
        test_set : test set of rating matrix, choose 5,000 nonzero elements
        train_set : training set, replace the selected elements in the test_set with zeros
        total_mean : mean of original rating matrix's non-zero elements
        item_bias : item bias
        user_bias : user bias
        '''
        file_pp=fpp.File_pp.get_Instance() # json 파일을 읽어와 전처리를 수행한다.
        self.matrix, self.items, self.users=file_pp.get_rating_matrix() # 전처리를 통해 구한 평점, item, user matrix를 리턴받는다.
        self.row_len=len(self.matrix)
        self.col_len=len(self.matrix[0])
        self.train_set, self.test_set = self.make_train_test()
        self.rating_matrix=np.zeros((1, 1))
        #self.mat_q, self.mat_pt=cal_SVD(self.train_set)
        self.lr=lr
        self.total_mean=self.get_mean()
        self.item_bias, self.user_bias=self.get_bias()
        self.train_set_error=[]
        self.test_set_error=[]
        self.trained=0 # if rating matrix is trained, variable trained turns to 1.
        self.loaded=0 # if admin load rating matrix, variable loaded turns to 1.
        self.load_SVD()

    re=None
    def get_instance(lambda_q=1, lambda_pt=1, lr=0.0001):
        if Recommend_Engine.re==None:
            Recommend_Engine.re=Recommend_Engine(lambda_q, lambda_pt, lr)
        return Recommend_Engine.re
        
    # cal_SVD() : Perform SVD with original rating matrix and save the results.
    def cal_SVD(self):
        factor_k = 100
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
        i_bias=np.zeros(self.row_len)
        u_bias=np.zeros(self.col_len)

        for idx in range(self.row_len):
            i_bias[idx]=np.mean(self.matrix[idx, self.matrix[idx, :].nonzero()])-self.total_mean

        for idx in range(self.col_len):
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

    #make_train_test() : 주어진 평점 행렬에서 train set과 test set을 분리한다.
    def make_train_test(self):
        #np.random.seed(0)
        train_set=self.matrix.copy()
        test_set=np.zeros(self.matrix.shape)

        nonzero_row, nonzero_col=self.get_nonzero(train_set)
        test_set_num=1000
        idx_list=np.random.choice(range(len(nonzero_row)), test_set_num, replace=False) # test_set_num을 통해 test set에 들어갈 원소 수를 정해준다.
        for idx in idx_list:
           test_set[nonzero_row[idx], nonzero_col[idx]]=train_set[nonzero_row[idx], nonzero_col[idx]]
           train_set[nonzero_row[idx], nonzero_col[idx]]=0

        return train_set, test_set

    #train() : Train the training set with SGD.
    def train_SGD(self, iter):
        self.train_set_error=[] #clear train set error.
        self.test_set_error=[] #clear test set error.
        if self.trained==1: #if rating matrix is already trained, check
            print('rating matrix is already trained. do you want to train again?(y, n)')
            k=input()
            if k=='n':
                return
            self.load_SVD()
        nonzero_row, nonzero_col=self.get_nonzero(self.train_set) #Returns the row number and column number of non-zero elements.
        nonzero_list=list(zip(nonzero_row, nonzero_col))
        start=time.time()
        prev_rmse=0.
        for i in range(iter):
            np.random.shuffle(nonzero_list)
            print("iter ", i)
            chk=1

            for idx in range(len(nonzero_row)):
                row=nonzero_list[idx][0] # Row number of nonzero element in train set.
                col=nonzero_list[idx][1] # Column number of nonzero element in train set.
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

    def train_minibatch(self, iter, batch_size):
        self.train_set_error=[] #clear train set error.
        self.test_set_error=[] #clear test set error.
        if self.trained==1: #if rating matrix is already trained, check
            print('rating matrix is already trained. do you want to train again?(y, n)')
            k=input()
            if k=='n':
                return
            self.load_SVD()

        nonzero_row, nonzero_col=self.get_nonzero(self.train_set) #Returns the row number and column number of non-zero elements.
        nonzero_list = list(zip(nonzero_row, nonzero_col))
        start=time.time()
        prev_rmse=0.
        for i in range(iter):
            np.random.shuffle(nonzero_list)
            print("iter ", i)
            chk=1
            tmp = 0

            while tmp<len(nonzero_row):
                tmp_mat_q = np.zeros(self.mat_q.shape)
                tmp_mat_pt = np.zeros(self.mat_pt.shape)

                tmp_item_bias = np.zeros(len(self.item_bias))
                tmp_user_bias = np.zeros(len(self.user_bias))

                for idx in range(tmp, min(len(nonzero_row), tmp+batch_size)):
                    row=nonzero_list[idx][0] # Row number of nonzero element in train set.
                    col=nonzero_list[idx][1] # Column number of nonzero element in train set.
                    baseline=self.get_baseline(row, col) #set baseline estimate
                    err=2*(self.train_set[row, col] - (baseline + np.dot(self.mat_q[row, :], self.mat_pt[:, col])))

                    if err > 100000 :
                        print("err exceeded over 100000. change learning rate and try again.")
                        return

                    tmp_mat_q[row, :] += self.lr * (err * self.mat_pt[:, col] - self.mat_q[row, :]) #update q
                    tmp_mat_pt[:, col] += self.lr * (err * self.mat_q[row, :] - self.mat_pt[:, col]) #update pt

                    tmp_item_bias[row] += self.lr * (err - self.item_bias[row]) #update item bias
                    tmp_user_bias[col] += self.lr * (err - self.user_bias[col]) #update user bias

                self.mat_q+=tmp_mat_q
                self.mat_pt+=tmp_mat_pt

                self.item_bias+=tmp_item_bias
                self.user_bias+=tmp_user_bias
                tmp+=batch_size

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
                #self.rating_matrix=self.rescaling(self.rating_matrix)
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

    # get_baseline() : Return baseline estimate for given row number and column number.
    def get_baseline(self, row, col):
        return self.total_mean + self.item_bias[row] + self.user_bias[col]

    # get_final_matrix() : Return learning matrix.
    def get_final_matrix(self):
        tmp=np.full(self.matrix.shape, self.total_mean)
        for i in range(self.col_len):
            tmp[:, i]+=self.item_bias

        for i in range(self.row_len):
            tmp[i, :]+=self.user_bias

        tmp+=np.dot(self.mat_q, self.mat_pt)
        return tmp

    def rescaling(self, matrix):
        recover_matrix=np.zeros(self.rating_matrix.shape) # 초기에 존재하던 평점정보들을 백업해둔다.
        copy_matrix=self.rating_matrix
        nonzero_row, nonzero_col=self.get_nonzero(self.matrix)
        for i in range(len(nonzero_row)): # 예측 평점 행렬에서 초기 평점정보들은 0으로 하고 이 값들을 recover_matrix에 백업해둔다.
            recover_matrix[nonzero_row[i], nonzero_col[i]] = copy_matrix[nonzero_row[i], nonzero_col[i]]
            copy_matrix[nonzero_row[i], nonzero_col[i]] = 0

        max=copy_matrix.max()
        min=copy_matrix.min()
        denom=max-min
        copy_matrix=((copy_matrix-min)/denom)*4+1 #값의 범위를 1~5로 스케일링
        for i in range(len(nonzero_row)): # 백업해뒀던 초기 평점정보들을 복원한다.
            copy_matrix[nonzero_row[i], nonzero_col[i]] = recover_matrix[nonzero_row[i], nonzero_col[i]]
        return copy_matrix

    #load_rating_matrix() : load rating matrix, train set error, test set error and store them in each variables.
    def load_rating_matrix(self):
        self.rating_matrix=np.load('rating_matrix.npy')
        self.train_set_error=list(np.load('train_set_error.npy'))
        self.test_set_error=list(np.load('test_set_error.npy'))
        self.rating_matrix = self.rescaling(self.rating_matrix)
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
        
    def get_matrix(self):
        return self.matrix
    
    def get_matrices(self):
        return self.rating_matrix, self.matrix, self.items, self.users

