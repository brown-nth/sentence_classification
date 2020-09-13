# sentence_classification
                                                       
                                            SENTENCE CLASSIFICATION 

RANDOM FORESTS ALGORITHM

1) Dữ liệu : 
Bộ dữ liệu review phim từ trang IMDB

Kết quả lấy từ cuộc thi :

	https://www.kaggle.com/c/word2vec-nlp-tutorial

2) Xử lý dữ liệu : 

a) Sử dụng thư viện nltk để xử lý văn bản thô : 
-  BeautifulSoup để loại bỏ thẻ html
- Thư viện re để loại các ký tự không phải chữ viết 
- Loại bỏ các stop word (trong tiếng anh ) như : a , an ,the , …
-  Sử dụng WordNetLemmatizer để nhóm các từ cùng nghĩa : programmer,    programming , ... --> program

b) 
Sử dụng one-hot : Sau khi clean data , sử dụng thư viện sklearn.feature_extraction.text.CountVectorizer để encode văn bản thành  numeric representation

Cải tiến hơn sử dụng gensim để dựng mô hình word2vec từ các câu trong tập labeled_train_data và unlabeled_train_data . 
Chọn các hyperparameter :
num_features = 300                         
min_word_count = 40                       
num_workers = 4       
context = 10                                                                                             
downsampling = 1e-3

Sử dụng 2 cách để encode vector cho mỗi review là tính mean words và sử dụng Kmean để divide các word trong review vào các cluster

C1 : Để tạo vector cho 1 review , ta tính trung bình của vector các từ (Hàm meanFeatureVect ) 

C2 : Sử dụng thư viện sklearn.cluster.KMeans với n_clusters bằng . Mỗi review được encode thành một mảng n_clusters phần tử với giá trị thứ i là số phần tử trong review  nằm ở cluster thứ i 


3) Xây dựng RandomForestClassifier từ thư viện sklearn.ensemble:

Dựng model với số cây n_estimators = 100
Fit model bằng các vector của từng review được tính ở trên 
Predict tập test rồi tiến hành xuất ra file .csv 

4) Kết quả : 
	
- Sử dụng one-hot : 
	Time to load data :  455.5291039943695 
Time to run model :  127.03707575798035
Accuracy : 61.28% 

	- Sử dụng word2vec kết hợp với tính mean: 
	Time to preprocess data:  378.60493659973145
Time to train word2vec : 96.25785827636719
Time to train random forest:  41.89255738258362
All time :  620.8874177932739
Accuracy : 83.12%

- Sử dụng word2vec kết hợp kmean : 
Time to preprocess data:  406.32320070266724
Time to train word2vec : 109.31126523017883
Time to train random forest:  65.77478456497192
Time to train KMean:  749.0132741928101
All time:  1333.8472437858582
Accuracy : 84.24%




5) Source code : 

	https://github.com/kiyoshitaro/sentence_classification


CNNs
Xử lý dữ liệu:
a) Sử dụng thư viện nltk để xử lý văn bản thô : 
-  BeautifulSoup để loại bỏ thẻ html
- Thư viện re để loại các ký tự không phải chữ viết 
- Loại bỏ các stop word (trong tiếng anh ) như : a , an ,the , …
-  Sử dụng WordNetLemmatizer để nhóm các từ cùng nghĩa : programmer,    programming , ... --> program

b) Sử dụng gensim để dựng mô hình word2vec từ các câu trong tập labeled_train_data và unlabeled_train_data . 

Chọn các hyperparameter :
num_features = 80 (do giới hạn bộ nhớ trong mô hình)                       
min_word_count = 40                       
num_workers = 4       
context = 10                                                                                             
downsampling = 1e-3

c) Mỗi review chỉ sử dụng 100 từ đầu tiên để đánh giá , nếu review không đủ 100 từ thì chèn vào các vector [0,0,....] cho đủ
	
Mô hình:
Coi mỗi review là một image với kích thước 100*80 
Ta dùng 3 kernel với kích thước lần lượt 3*80, 5*80, 7*80 quét qua 
Ở hidden layer thứ nhất dùng relu là activate function thu được 3 vector 1*k, sử dụng reduce_mean cho các vector đó rồi concat lại  , ở lớp  cuối sử dụng full_connected và  softmax_cross_entropy_with_logits_v2 để đưa ra phân loại 
Sử dụng MomentumOptimizer với learning_rate=0.01 và momentum=0.9 để minimize loss function
Chạy chương trình trên Google Colab
Guide : 
Kết quả:
	      - Chạy  đến khi nào loss ở training nhỏ hơn 0.2 thì dừng, thu được kết quả : 
Accuracy : 82.36% 
