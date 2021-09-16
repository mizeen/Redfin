import sklearn as sk
import pandas as pd
import re

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

# X=all_data[['PRICE','BEDS','PROPERTY TYPE','STATE']]
X=all_data[['PRICE','BEDS','STATE','PROPERTY TYPE']]

for ind in X.index:
	if X['STATE'][ind]== 'PA':
		X['STATE'][ind]= 1
	else:
		X['STATE'][ind]=2
for ind in X.index:
	x=(re.search("MULTI", X["PROPERTY TYPE"][ind]))
	if x:
		X["PROPERTY TYPE"][ind]=1
	else:
		X["PROPERTY TYPE"][ind]=2

print(X.head)
print(y.head)
# print(y.series.info())
print('done')

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.85)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn import metrics
# classifier_knn = KNeighborsClassifier(n_neighbors = 3)
# classifier_knn.fit(X_train, y_train)
# y_pred = classifier_knn.predict(X_test)
# # Finding accuracy by comparing actual response values(y_test)with predicted response value(y_pred)
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# from sklearn import svm

# clf_svm=svm.SVC(kernel='linear')
# clf_svm.fit(X_train,y_train)
# print('svm',clf_svm.score(X_test,y_test))


from sklearn.tree import DecisionTreeClassifier 
clf_tree=DecisionTreeClassifier()
clf_tree.fit(X_train,y_train)
# clf_tree.predict(X_test[5])
# print(X_test[5])

print('Dtree',clf_tree.score(X_test,y_test))
a=clf_tree.predict(X_test)
print(a)
print(X_test.to_string())




# from sklearn.externals import joblib
# joblib.dump(classifier_knn, 'Redfin_classifier_knn.joblib')