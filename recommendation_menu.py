import project_bias as pb
import User_mode as Umod
import file_preprocessor as fpp
import TF_IDF

class Rec_Menu:
    def __init__(self):
        fpp.File_pp.get_Instance()
        self.tf_idf=TF_IDF.Tf_Idf()
        self.rec = pb.Recommend_Engine.get_instance(lr=0.0002)
        _, _, self.items, _=self.rec.get_matrices()
        self.tf_idf.tfidf_all(self.items.keys())  # 모든 item의 TF-IDF 값을 미리 계산해둔다.
        self.loaded=0
        self.calculated=0
        print('-' * 5, 'music recommendation system ver 0.1', '-' * 5)
        while True:
            print('1. user mode')
            print('2. admin mode')
            print('3. quit')
            select = input('select : ')

            if select == '1':
                self.user_mode()
            elif select == '2':
                self.admin_mode()
            elif select == '3':
                print()
                return
            else:
                print("Please enter a vaild number. (1-3)\n")

    def user_mode(self):
        um=Umod.User_Mode(self.loaded, self.calculated, self.tf_idf, self.rec)

    def admin_mode(self):
        while True:
            print("\n", '-' * 5, 'admin mode', '-' * 5)
            print('1. calculate rating matrix')
            print('2. load rating matrix')
            print('3. show rating matrix')
            print('4. show error graph')
            print('5. quit')
            select = input('select : ')

            if select == '1':
                print('1. SGD')
                print('2. SGD with minibatch')
                mode=input('사용할 학습 모드를 선택해주세요 : ')

                if mode=='1':
                    self.rec.train_SGD(500)
                    self.calculated = 1
                elif mode=='2':
                    self.rec.train_minibatch(500, 128)
                    self.calculated = 1
                else:
                    print('올바른 숫자를 입력해주세요.')

            elif select == '2':
                check=self.rec.load_rating_matrix()
                if check:
                    print("Load completed.")
                    self.loaded=1
                else:
                    print("Load failed. File not exist")

            elif select == '3':
                self.rec.show_rating_matrix()

            elif select == '4':
                self.rec.show_graph()
        
            elif select=='5':
                print()
                return
            else:
                print("Please enter a vaild number. (1-5)\n")

m=Rec_Menu()
