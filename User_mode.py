import project_bias as pb
import TF_IDF

class User_Mode:
    def __init__(self, loaded, calculated):
        self.rec_list=[]
        self.loaded=loaded
        self.calculated=calculated
        self.rs=pb.Recommend_Engine_SGD.get_instance()
        self.rating_matrix, self.matrix, self.items, self.users=self.rs.get_matrices()
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
        num = int(input("Enter number of items : "))
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

        '''print('-----test TF-IDF-----')
        for i in item_idx:
            print('item ID : ', itemIDs[i], 'TF-IDF : ', TF_IDF.tfidf(itemIDs[i]))'''

        for i in range(len(test_list)):
            self.rec_list.append([itemIDs[i], test_list[i]])
        for i in item_idx:
            self.rec_list[i][1] = 0  # 기존에 있었던 평점은 0으로 취급하여 추천 목록에 들어가지 않게 한다.

        self.rec_list.sort(key=lambda x: x[1], reverse=True)
        for i in range(num):
            print('%d. %s %lf   ' % (i + 1, self.rec_list[i][0], self.rec_list[i][1]), end=' ')
            print('이런 키워드를 가지고 있습니다 : ', TF_IDF.tfidf(self.rec_list[i][0], 3))


    '''def delete_item(self, userID):
        print('-------Current recommendation list-------')
        if (self.rec_list)
            return'''