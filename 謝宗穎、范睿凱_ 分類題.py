import pandas as pd #匯入檔案
from sklearn.model_selection import train_test_split #將資料切割為訓練及測試集
from sklearn.preprocessing import LabelEncoder #把每個類別對應到某個整數
from sklearn.feature_selection import SelectKBest #選出N個分數最高的特徵
from sklearn.feature_selection import f_regression #適用於回歸模型的特徵評估
from sklearn.tree import DecisionTreeClassifier #Logistic 分類模型

train_data = pd.read_csv('train.csv') #匯入檔案
train_data = train_data.drop(['PassengerId'], axis = 1) #刪除欄位（PassengerId）
X_data = train_data.drop(['Survived'], axis = 1) #刪除欄位（Survived）
y_data = train_data['Survived'] 

X_data = X_data.dropna(axis = 1, how = 'any')

encoder = LabelEncoder()
X_encoded = pd.DataFrame(X_data,columns=X_data.columns).apply(lambda col:encoder.fit_transform(col))

kbest = SelectKBest(f_regression,k = 5)
X_new = kbest.fit_transform(X_encoded, y_data)

X_train, X_test, y_train, y_test = train_test_split(X_new, y_data,train_size=0.7, test_size=0.3,random_state=0)
model = DecisionTreeClassifier(max_depth = 7, criterion = 'gini', splitter ='best',
                               min_samples_split =2, min_samples_leaf=1, min_weight_fraction_leaf=0
                               ,max_features=5,max_leaf_nodes=None,min_impurity_decrease=0,
                               class_weight='balanced', random_state = 0)

model.fit(X_train, y_train)
score =model.score(X_test, y_test)
print('Score: ', score)
print('Accuracy: ' + str(score*100) + '%')


