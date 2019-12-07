import sys
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
import time
import random
import file_preprocessor as fpp

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
    def __init__(self, lr=0.0001):
        '''
        row_len : 평점 행렬의 행의 길이, 상품 갯수를 의미한다.
        col_len : 평점 행렬의 열의 길이, 사용자의 수를 의미한다.
        test_set : 평점 행렬에서 뽑아낸 test set을 저장한다.
        train_set : 평점 행렬에서 뽑아낸 train set을 저장한다.
        total_mean : 전체 평점 평균
        item_bias : 상품별 바이어스 저장, 특정 상품의 평점 평균 - 전체 평점 평균
        user_bias : 사용자별 바이어스 저장, 특정 사용자의 평점 평균 - 전체 평점 평균
        '''

        file_pp=fpp.File_pp.get_Instance() # json 파일을 읽어와 전처리를 수행하는 클래스. 싱글턴 패턴을 이용해 다른 클래스와 내용을 공유한다.
        self.matrix, self.items, self.users=file_pp.get_rating_matrix() # 전처리를 통해 구한 평점, item, user matrix를 리턴받는다.
        self.row_len=len(self.matrix)
        self.col_len=len(self.matrix[0])
        self.train_set, self.test_set = self.make_train_test()
        self.rating_matrix=np.zeros((1, 1)) #임시로 선언. 수정
        self.lr=lr
        self.total_mean=self.get_mean()
        self.item_bias, self.user_bias=self.get_bias()
        self.train_set_error=[]
        self.test_set_error=[]
        self.trained=0 # 이미 학습을 수행했는 지를 체크한다.
        self.loaded=0 # 저장된 파일로부터 예측 평점 행렬을 불러왔는지 체크한다.
        self.load_SVD() # 저장된 파일로부터 SVD 결과를 불러온다. 파일이 없으면 SVD를 수행하여 파일로 저장하고 이를 리턴한다.

    re=None
    def get_instance(lr=0.0001): # 여러 클래스에서 Recommender_Engine 클래스의 인스턴스를 공유할 수 있도록 한다.
        if Recommend_Engine.re==None:
            Recommend_Engine.re=Recommend_Engine(lr=lr)
        return Recommend_Engine.re
        
    # cal_SVD() : 평점 행렬에 SVD를 수행하고 그 결과를 파일로 저장한다.
    def cal_SVD(self):
        factor_k = 100
        U, s, V = svds(self.matrix, factor_k)
        S = np.zeros((len(s), len(s)))
        for i in range(len(s)):
            S[i][i] = s[i]
        PT = np.dot(S, V)
        np.save('SVD_Q.npy', U)
        np.save('SVD_PT.npy', PT)

    # load_SVD() : SVD 파일을 불러온다. 없으면 cal_SVD를 통해 새로 만든다.
    def load_SVD(self):
        try:
            self.mat_q = np.load('SVD_Q.npy')
            self.mat_pt = np.load('SVD_PT.npy')
        except IOError:
            print('Make SVD...')
            self.cal_SVD()
            self.load_SVD()
    
    # get_mean() : 전체 평점 평균을 반환한다.
    def get_mean(self):
        nonzero_row, nonzero_col=self.get_nonzero(self.matrix)
        res=0
        for i in range(len(nonzero_row)):
            row=nonzero_row[i]
            col=nonzero_col[i]
            res+=self.matrix[row, col]
        return res/len(nonzero_row)

    # get_bias() : 상품 바이어스, 사용자 바이어스를 리스트로 묶어 리턴한다.
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
        test_set_num=1000 # 테스트 샘플 갯수
        idx_list=np.random.choice(range(len(nonzero_row)), test_set_num, replace=False) # test_set_num을 통해 test set에 들어갈 원소 수를 정해준다.
        for idx in idx_list:
           test_set[nonzero_row[idx], nonzero_col[idx]]=train_set[nonzero_row[idx], nonzero_col[idx]]
           train_set[nonzero_row[idx], nonzero_col[idx]]=0

        return train_set, test_set


    #train_SGD(iter) : SGD를 이용하여 평점 행렬을 학습시킨다. iter는 iteration 횟수를 결정한다.
    def train_SGD(self, iter):
        self.train_set_error = []  # train set의 에러를 저장한다.
        self.test_set_error = []  # test set의 에러를 저장한다.

        nonzero_row, nonzero_col = self.get_nonzero(self.train_set)  # train set에서 0이 아닌 원소들의 행 번호, 열 번호를 리턴한다.
        nonzero_list = list(zip(nonzero_row, nonzero_col))  # 열 번호, 행 번호를 묶어서 하나의 점으로 만든 뒤, 리스트에 넣어 사용한다.
        start = time.time()  # 수행시간 측정
        prev_rmse = 0.
        for i in range(iter):
            np.random.shuffle(nonzero_list)
            print("epoch : ", i + 1)
            chk = 1

            for idx in range(len(nonzero_row)):
                row = nonzero_list[idx][0]  # Row number of nonzero element in train set.
                col = nonzero_list[idx][1]  # Column number of nonzero element in train set.
                baseline = self.get_baseline(row, col)  # set baseline estimate
                err = 2 * (self.train_set[row, col] - (baseline + np.dot(self.mat_q[row, :], self.mat_pt[:, col])))

                '''if err > 100000 :
                    print("err exceeded over 100000. change learning rate and try again.")
                    return'''

                self.mat_q[row, :] += self.lr * (err * self.mat_pt[:, col] - self.mat_q[row, :])  # update q
                self.mat_pt[:, col] += self.lr * (err * self.mat_q[row, :] - self.mat_pt[:, col])  # update pt

                self.item_bias[row] += self.lr * (err - self.item_bias[row])  # update item bias
                self.user_bias[col] += self.lr * (err - self.user_bias[col])  # update user bias

                if idx % 5000 == 0:
                    print('idx : ', idx)
                    train_rmse = self.get_rmse(self.train_set)
                    test_rmse = self.get_rmse(self.test_set)

                    print("train set error : ", train_rmse)
                    print("test set error : ", test_rmse)
                    print('')
                    self.train_set_error.append(train_rmse)
                    self.test_set_error.append(test_rmse)

                    if abs(prev_rmse - test_rmse) <= 1e-5:  # 학습 전과 후의 RMSE 차이가 10^-6 이하인 경우 학습 종료
                        print("train finished!")
                        print('-' * 5, 'reult matrix', '-' * 5)
                        self.rating_matrix = self.get_matMul()  # 학습이 끝난 p과 qt를 곱해서
                        self.rating_matrix = self.rescaling(self.rating_matrix)  # 리스케일링
                        print(self.rating_matrix)
                        np.save('mat_q.npy', self.mat_q)
                        np.save('mat_pt.npy', self.mat_pt)
                        np.save('train_set.npy', self.train_set)
                        np.save('test_set.npy', self.test_set)
                        np.save('train_set_error.npy', self.train_set_error)
                        np.save('test_set_error.npy', self.test_set_error)
                        print("time : ", time.time() - start)
                        self.trained = 1
                        return
                    prev_rmse = test_rmse

        if abs(prev_rmse - test_rmse) <= 1e-3:  # End training when the RMSE difference in test set is less than 0.001.
            print("train finished!")
            print('-' * 5, 'reult matrix', '-' * 5)
            self.rating_matrix = self.get_matMul()
            self.rating_matrix = self.rescaling(self.rating_matrix)
            print(self.rating_matrix)
            np.save('rating_matrix.npy', self.rating_matrix)
            np.save('train_set_error.npy', self.train_set_error)
            np.save('test_set_error.npy', self.test_set_error)
            print("time : ", time.time() - start)
            self.trained = 1
            return

    def train_minibatch(self, iter, batch_size):
        self.train_set_error=[] #clear train set error.
        self.test_set_error=[] #clear test set error.
        '''if self.trained==1: #if rating matrix is already trained, check
            print('rating matrix is already trained. do you want to train again?(y, n)')
            k=input()
            if k=='n':
                return
            self.load_SVD()'''

        nonzero_row, nonzero_col=self.get_nonzero(self.train_set) #train set들의 열 번호, 행 번호를 받는다.
        nonzero_list = list(zip(nonzero_row, nonzero_col))
        start=time.time()
        prev_rmse=0.
        for i in range(iter):
            print('epoch : ', i+1)
            np.random.shuffle(nonzero_list)
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

                    tmp_mat_q[row, :] += self.lr * (err * self.mat_pt[:, col] - self.lambda_q * self.mat_q[row, :]) #기울기 변화량을 구해서 임시 배열에 저장해둔다.
                    tmp_mat_pt[:, col] += self.lr * (err * self.mat_q[row, :] - self.lambda_pt * self.mat_pt[:, col]) #기울기 변화량을 구해서 임시 배열에 저장해둔다.

                    tmp_item_bias[row] += self.lr * (err - self.lambda_bx * self.item_bias[row]) #바이어스 변화량을 구해서 임시 배열에 저장해둔다.
                    tmp_user_bias[col] += self.lr * (err - self.lambda_bi * self.user_bias[col]) #바이어스 변화량을 구해서 임시 배열에 저장해둔다.

                self.mat_q+=tmp_mat_q #mini batch 크기만큼 저장해둔 기울기 변화량들을 동시에 적용한다.
                self.mat_pt+=tmp_mat_pt #mini batch 크기만큼 저장해둔 기울기 변화량들을 동시에 적용한다.

                self.item_bias+=tmp_item_bias #mini batch 크기만큼 저장해둔 바이어스 변화량들을 동시에 적용한다.
                self.user_bias+=tmp_user_bias #mini batch 크기만큼 저장해둔 바이어스 변화량들을 동시에 적용한다.
                tmp+=batch_size

                if (tmp/batch_size)%500==0: #수정해야됨
                    train_rmse=self.get_rmse(self.train_set)
                    test_rmse=self.get_rmse(self.test_set)

                    print("train set error : ", train_rmse)
                    print("test set error : ", test_rmse)
                    print() # 한줄 띄우기

                    self.train_set_error.append(train_rmse)
                    self.test_set_error.append(test_rmse)

                    if abs(prev_rmse - test_rmse) <= 1e-5: # 기울기 반영 전과 반영 후 rmse값의 차이가 10^-5 이하이면 학습을 종료한다.
                        print("train finished!")
                        print('-'*5, 'reult matrix','-'*5)
                        self.rating_matrix=self.get_matMul()
                        self.rating_matrix=self.rescaling(self.rating_matrix)
                        print(self.rating_matrix)
                        np.save('mat_q.npy', self.mat_q)
                        np.save('mat_pt.npy', self.mat_pt)
                        np.save('train_set_error.npy', self.train_set_error)
                        np.save('test_set_error.npy', self.test_set_error)
                        print("time : ", time.time() - start)
                        self.trained=1
                        return
                    
                    prev_rmse=test_rmse

    def get_obj_function(self):
        nonzero_row, nonzero_col=self.get_nonzero(self.train_set)
        sum=0
        for i in range(len(nonzero_row)):
            row=nonzero_row[i]
            col=nonzero_col[i]
            baseline=self.get_baseline(row, col)
            sum+=((self.train_set[row, col] - (baseline + np.dot(self.mat_q[row, :], self.mat_pt[:, col])))**2).sum() +\
                 self.lambda_pt * self.get_norm(self.mat_pt[:, col]) + self.lambda_q * self.get_norm(self.mat_q[row, :]) +\
                 self.lambda_bx * (self.item_bias[row]**2) + self.lambda_bi *((self.user_bias[col])**2)
        return sum

    def get_norm(self, list):
        return (list**2).sum()

    # get_baseline() : Return baseline estimate for given row number and column number.
    def get_baseline(self, row, col):
        return self.total_mean + self.item_bias[row] + self.user_bias[col]

    # get_final_matrix() : 행렬 q와 pt의 행렬곱 결과를 리턴한다.
    def get_matMul(self):
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
        self.mat_pt=np.load('mat_pt.npy')
        self.mat_q=np.load('mat_q.npy')
        self.test_set=np.load('test_set.npy')
        self.train_set=np.load('train_set.npy')
        self.rating_matrix=self.get_matMul()
        self.train_set_error=list(np.load('train_set_error.npy'))
        self.test_set_error=list(np.load('test_set_error.npy'))
        #self.rating_matrix = self.rescaling(self.rating_matrix)
        if len(self.rating_matrix) > 1:
            self.loaded=1
            return True
        else: return False

    #get_rmse() : 주어진 행렬의 RMSE 값을 구한다.
    def get_rmse(self, matrix, tRow=0, tCol=0):
        sum=0
        nonzero_row, nonzero_col = self.get_nonzero(matrix)
        pred_matrix=self.get_matMul()
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

    def partial_training(self, row, col, val):
        self.trained=1
        baseline = self.get_baseline(row, col)  # baseline estimate 설정
        err=9999
        prev_err=0
        print('test : ', self.test_set[row, col])
        self.test_set[row, col]=0.1
        print('test! : ', self.test_set[row, col])
        print(self.get_rmse(self.test_set, row, col))
        self.train_SGD(500)
        self.rating_matrix=self.get_matMul()
        return self.rating_matrix, self.test_set

class ItemItemCFWithCosSim:
    def __init__(self):
        self.load_train_test()
        self.pp=fpp.File_pp.get_Instance()
        self.matrix, self.items, self.users = self.pp.get_rating_matrix()

    def get_norm(dict):
        sum = 0
        for i in dict.values():
            sum += i ** 2
        return math.sqrt(sum)

    def get_similarity(self, a, b): #a와 b에는 평점 행렬에서 각 아이템 a와 b가 받은 평점들을 가진 리스트가 들어온다.
        return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

    def load_train_test(self):
        self.test_set = np.load('test_set.npy')
        self.train_set = np.load('train_set.npy')

    def get_nonzero(self, matrix):
        nonzero_row, nonzero_col = matrix.nonzero()
        return nonzero_row, nonzero_col

    def CF(self):
        row_list, col_list=self.get_nonzero(self.test_set)
        nonzero_list=list(zip(row_list, col_list))
        test_matrix=np.zeros(self.matrix.shape)
        i=0
        for element in nonzero_list:
            i+=1
            nonzero=self.train_set[:, element[1]].nonzero()[0] #train set에 있는 아이템들 중 test set에 있는 element의 사용자가 평점을 매긴 아이템들의 인덱스를 저장하고 있다.

            sim_list=[]
            for k in nonzero:
                sim_list.append((k, self.get_similarity(self.matrix[element[0], :], self.matrix[k, :])))

            sim_list.sort(key=lambda x : x[1], reverse=True)
            if sim_list[0][1]==0:
                print('zero : ', element)
                print(sim_list)

            div=0
            for i in range(min(len(sim_list), 3)):
                test_matrix[element[0], element[1]]+=sim_list[i][1]*self.train_set[sim_list[i][0], element[1]]
                div+=sim_list[i][1]
            if div==0 : div=1 # 만일

            test_matrix[element[0], element[1]]/=div
            print('result : ', test_matrix[element[0], element[1]], 'real value : ', self.test_set[element[0], element[1]])

        print('rmse : ', self.get_rmse(test_matrix, self.test_set))


    def get_rmse(self, pred_matrix, matrix):
        sum = 0
        nonzero_row, nonzero_col = self.get_nonzero(matrix)
        for idx in range(len(nonzero_row)):
            row = nonzero_row[idx]
            col = nonzero_col[idx]
            target = matrix[row, col]
            print('test element: ', (target - pred_matrix[row, col]) ** 2)
            sum += (target - pred_matrix[row, col]) ** 2
        sum = np.sqrt(sum / len(nonzero_row))
        return sum