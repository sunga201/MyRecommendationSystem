import project_bias as pb
import TF_IDF
import time

class User_Mode:
    def __init__(self, loaded, calculated, tf_idf):
        self.rec_list=[]
        self.loaded=loaded
        self.calculated=calculated
        self.rs=pb.Recommend_Engine.get_instance()
        self.rating_matrix, self.matrix, self.items, self.users=self.rs.get_matrices()
        self.num=0
        self.recommended=False
        start=time.time()
        self.tf_idf=tf_idf
        self.show_user_menu()

    def show_user_menu(self):
        if self.loaded == 0 and self.calculated == 0:
            print("Please load or calculate rating matrix from admin mode.\n")
            return

        self.userID = input("Enter your ID : ")
        while self.userID not in self.users.keys():
            print('User ID is incorrect. Please recheck your ID.')
            self.userID = input("Enter your ID : ")

        while True:
            print("\n", '-' * 5, 'user mode', '-' * 5)
            print('1. Show recommended items')
            print('2. Delete items from recommendation list')
            print('3. recommendation for keyword')
            print('4. quit')
            select = input('select : ')

            if select == '1':
                self.recommend_item()

            elif select == '2':
                self.delete_item()

            elif select == '4':
                print()
                return

            else:
                print("Please enter a vaild number. (1-2)\n")

    # recommend_item() : recommend k items for user.
    def recommend_item(self):
        self.recommended=True
        self.num = int(input("Enter number of items : "))
        self.show_recommand_music()

    def show_recommand_music(self):
        user_idx = self.users[self.userID]  # Get user index
        nonzero_row, nonzero_col = self.rs.get_nonzero(self.rs.get_matrix())  # 기존에 존재하던 평점들의 인덱스를 저장한다.
        item_idx = []

        for i in range(len(nonzero_col)):
            if nonzero_col[i] == user_idx:  # 해당 유저가 평가한 상품들을 item_idx 리스트에 저장한다.
                row = nonzero_row[i]
                item_idx.append(row)
        test_list = self.rating_matrix[:, user_idx]  # 예측 평점 행렬에서 해당 유저가 가지는 각 상품들의 평점을 test_list에 저장한다.
        self.rec_list = []
        itemIDs = list(self.items.keys())  # 인덱스를 입력하면 그 인덱스에 해당하는 아이템 ID를 리턴한다.

        for i in range(len(test_list)):
            self.rec_list.append([itemIDs[i], test_list[i]])
        for i in item_idx:
            self.rec_list[i][1] = 0  # 기존에 있었던 평점은 0으로 취급하여 추천 목록에 들어가지 않게 한다.

        self.rec_list.sort(key=lambda x: x[1], reverse=True) # self.rec_list는 추후에 추천 목록에서 아이템을 지울 때도 사용한다.
        print('-'*40, '추천 목록', '-'*40)
        print('-'*10, 'item', '-'*5, 'expected rate', '-'*22, 'keyword',  '-'*23)
        for i in range(self.num):
            print('%-2d.  |  %s  |   %lf   |    ' % (i + 1, self.rec_list[i][0], self.rec_list[i][1]), end=' ')
            print(self.tf_idf.get_tfidf(self.rec_list[i][0], 5))


    def delete_item(self):
        if not self.recommended:
            self.num = int(input("Enter number of items : "))

        self.show_recommand_music()
        while True:
            chk=0
            num_list=input('\n추천목록에서 제거하고 싶은 음악의 번호를 입력해주세요. : ').replace(',', ' ').split()# 사용자로부터 지우고 싶은 음악의 번호 여러 개를 받는다.
            print() #한줄 띄우기
            removed_list=[]
            for idx_str in num_list:
                try:
                    idx=int(idx_str)-1
                    if idx<0 or idx>self.num-1:
                        raise ValueError
                except ValueError or AttributeError:
                    print('올바른 숫자를 입력해주세요.')
                    chk = 1
                    break
                removed_list.append(self.rec_list[idx])
            if not chk: break

        for item in removed_list:
            row=self.items[item[0]] # 예측 평점 행렬에서 해당 아이템의 번호
            col=self.users[self.userID] # 예측 평점 행렬에서 사용자의 번호
            self.rating_matrix[row][col]=0 # 사용자가 선호하지 않는다고 말한 항목들은 평점을 0으로 한다.
            self.rec_list.remove(item)

        #self.get_similar_items(row)

        print(' '*10, '다음 아이템들이 추천 목록에서 삭제되었습니다.', ' '*10)
        
        self.show_recommand_music()
        

        return

    '''def get_similar_items(self, item_idx):'''
        
