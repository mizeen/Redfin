import sklearn as sk
import pandas as pd
import re
import numpy as np

all_data =pd.read_csv("faves3.csv")
all_data=pd.DataFrame(all_data)
# all_data.dropna(subset=['PRICE'],inplace=True)
y=all_data['LIKE'] #Target
for ind in y.index:
	if y[ind]== 'YES':
		y[ind]=1
	else:
		y[ind]=2
y=y.astype('int')
X=all_data[['PRICE','BEDS','STATE','PROPERTY TYPE']]





from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_state=vectorizer.fit_transform(X['STATE'])
print(X_state.toarray())
X_ptype=vectorizer.fit_transform(X['PROPERTY TYPE'])
print(X_ptype.toarray())


a=X_state.toarray()

b=X_ptype.toarray()
print(np.shape(a))
print(np.shape(b))
d=X[['PRICE','BEDS']]

d=np.array(d)
print(np.shape(d))

# f=np.concatenate((d,e),axis=1)
# print(np.shape(f))

c=np.concatenate((a,b), axis=1)
print(np.shape(c))
# print(c)
X=np.concatenate((c,d),axis=1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


from miz_nn import LogisticRegression
def accuracy(y_true,y_pred):
	accuracy = np.sum(y_true==y_pred)/len(y_true)
	return accuracy
regressor=LogisticRegression(lr=.0001,n_iters=1000)
regressor.fit(X_train,y_train)

predictions=regressor.predict(X_test)
print("Miz LR accuracy: ", accuracy(y_test,predictions))



from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
classifier_knn = KNeighborsClassifier(n_neighbors = 3)
classifier_knn.fit(X_train, y_train)
y_pred = classifier_knn.predict(X_test)
# Finding accuracy by comparing actual response values(y_test)with predicted response value(y_pred)
print("KNN Accuracy:", metrics.accuracy_score(y_test, y_pred))



from sklearn import svm
clf_svm=svm.SVC(kernel='linear')
clf_svm.fit(X_train,y_train)
print('svm',clf_svm.score(X_test,y_test))


from sklearn.tree import DecisionTreeClassifier 
clf_tree=DecisionTreeClassifier()
clf_tree.fit(X_train,y_train)
print('Dtree',clf_tree.score(X_test,y_test))


# import tensorflow as tf
# from tensorflow import keras
# import tensorflow_decision_forests as tfdf



# X_train['STATE']=vectorizer.fit_transform(X_train['STATE'])
# X_test['STATE']=vectorizer.transform(X_test['STATE'])
# y= vectorizer.fit_transform(y)

# # print(y)
# y=pd.DataFrame(y)
# # y.to_csv('C:\\Users\\mizan\\Documents\\MIT Courses\\ytest.csv')




# # X=all_data[['PRICE','BEDS','PROPERTY TYPE','STATE']]

# X['STATE']=vectorizer.transform(X['STATE'])
# # v2=vectorizer.fit_transform(X['STATE'])
# X['PROPERTY TYPE']=vectorizer.transform(X['PROPERTY TYPE'])
# print(v2.toarray())
# X=pd.DataFrame(X)
# X.to_csv('C:\\Users\\mizan\\Documents\\MIT Courses\\Xtest.csv')



# for ind in X.index:
# 	if X['STATE'][ind]== 'PA':
# 		X['STATE'][ind]= 1
# 	else:
# 		X['STATE'][ind]=2
# for ind in X.index:
# 	x=(re.search("MULTI", X["PROPERTY TYPE"][ind]))
# 	if x:
# 		X["PROPERTY TYPE"][ind]=1
# 	else:
# 		X["PROPERTY TYPE"][ind]=2

# print(X.head)
# print(y.head)
# # print(y.series.info())
# print('done')











# # clf_tree.predict(X_test[5])
# # print(X_test[5])


# a=clf_tree.predict(X_test)
# print(a)
# print(X_test.to_string())