# Importing the libraries
import pandas as pd
import numpy as np

# Importing the dataset
dataset = pd.read_csv('bank-main.csv')
XX = dataset.iloc[:, [0,1,5,6,10,11,13,14,15,16,17,18,19]].values
X = pd.DataFrame(XX)
yy = dataset.iloc[:, 20].values
y=pd.DataFrame(yy)

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
for i in range (1,4):
    X.values[:, i] = labelencoder_X.fit_transform(X.values[:, i])
X.values[:,7]= labelencoder_X.fit_transform(X.values[:, 7])
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
#One-hot encoding
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred_lr = classifier.predict(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred_knn = classifier.predict(X_test)

# Fitting svm classifier to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf')
classifier.fit(X_train,y_train)
# Predicting the Test set results
y_pred_svm = classifier.predict(X_test)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 1)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred_dt = classifier.predict(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred_rf = classifier.predict(X_test)

#Merging the predictions
y_predm=pd.DataFrame({'Logistic Regression':y_pred_lr,'K-Nearest Neighbour':y_pred_knn,'SVM':y_pred_svm,'Decision Tree Classification':y_pred_dt,'Random Forest':y_pred_rf})
#Generating a dataframe of zeroes
zero_data=np.zeros(shape=(1,8238))
y_pred=pd.DataFrame(zero_data)
y_pred=y_predm.mode(axis=1)
y_predict=y_pred.iloc[:,0].values

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
acc=(cm[0][0]+cm[1][1])/np.sum(cm)*100
print("Accuracy is : ",acc)