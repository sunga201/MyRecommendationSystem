import cal_SGD as calSGD
import User_mode as Umod
import file_preprocessor as fpp
import TF_IDF
import os

class Rec_Menu:
    def __init__(self):
        print('파일 전처리 진행중...')
        self.fpp=fpp.File_pp() # 데이터 전처리 수행
        self.tf_idf=TF_IDF.Tf_Idf()
        self.rec = calSGD.Recommend_Engine(lr=0.0002)
        self.calculated=False
        self.loaded=self.load_rating_matrix()
        if not self.loaded:
            print('저장된 예측 평점 행렬이 없습니다. 관리자 모드를 통해 예측 평점 행렬을 만들어주세요.')
        else:
            print('Prediction matrix loaded.')
        _, _, self.items, _ = self.rec.get_matrices()
        self.tf_idf.tfidf_all(self.items.keys())  # 모든 item의 TF-IDF 값을 미리 계산해둔다.

        while True:
            print('-' * 5, 'music recommendation system', '-' * 5)
            print('1. 사용자 모드')
            print('2. 관리자 모드')
            print('3. 종료')
            select = input('select : ')

            if select == '1':
                self.user_mode()
            elif select == '2':
                self.admin_mode()
            elif select == '3':
                print()
                return
            else:
                print("올바른 숫자를 입력해주세요. (1-3)\n")

    def load_rating_matrix(self):
        return self.rec.load_rating_matrix()

    def user_mode(self):
        um=Umod.User_Mode(self.loaded, self.calculated, self.tf_idf, self.rec)

    def admin_mode(self):
        while True:
            print("\n", '-' * 5, 'admin mode', '-' * 5)
            print('1. 예측 평점 행렬 계산')
            print('2. 예측 평점 행렬 보기')
            print('3. 에러 그래프 출력')
            print('4. RMSE 비교')
            print('5. 초기 평점 행렬 파일 제거')
            print('6. 돌아가기')
            select = input('select : ')

            if select == '1':
                while True:
                    print('1. SGD')
                    print('2. SGD with minibatch')
                    print('3. 메뉴로 돌아가기')
                    mode=input('사용할 학습 모드를 선택해주세요 : ')

                    if mode=='1':
                        self.rec.train_SGD(30)
                        self.calculated = 1

                    elif mode=='2':
                        self.rec.train_minibatch(30, 128)
                        self.calculated = 1

                    elif mode=='3':
                        return
                    else:
                        print('올바른 숫자를 입력해주세요. (1-3)\n')
                    if self.calculated : break

            elif select == '2':
                self.rec.show_rating_matrix()

            elif select == '3':
                self.rec.show_graph()

            elif select=='4':
                try:
                    self.rec.compare_RMSE()
                except FileNotFoundError:
                    print('먼저 예측 평점 행렬을 생성해주세요.')

            elif select=='5': #저장되어 있는 초기 평점 행렬을 제거한다. 추천 목록에서 특정 항목을 제거함으로써 변화한
                              # 초기 평점을 초기화하기 위해 사용
                try:
                    os.remove('./initial_matrix.npy')
                except:
                    print('파일이 존재하지 않습니다.')

            elif select=='6':
                print()
                return
            else:
                print("올바른 숫자를 입력해주세요. (1-6)\n")

m=Rec_Menu()
