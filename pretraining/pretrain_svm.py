"""Pretrains base SVM model."""

import numpy as np
import io,csv

from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

act_name1 = ['I','He','She','They','Your','Time','Clinic','Doctor','Bank','Meatshop']
act_name2 = ['Need','Like','Where','Name','Fathers Name','What']
act_name3 = ['Help','Water','Medicine','Bread','What']
act_name4 = ['Dont','One']

path = r'C:\Users\Karush\.spyder-py3'

# activity-1 svm
n_comp = 80
X1 = np.zeros((0,126))
y1 = np.zeros((0,0))
for j in range(0,len(act_name1)):
    fio = io.open(path + act_name1(j) + '.csv', 'rt')
    sign1 = csv.reader(fio)
    sign1 = list(sign1)
    sign1 = np.array(sign1).astype(np.float)
    X1 = np.r_['0,2', X1, sign1]
    y1 = np.r_['0,2', y1, j*np.ones((len(sign1), 0))]
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3)
X_train1[np.isnan(X_train1)] = np.mean(X_train1[~np.isnan(X_train1)])
X_train1 = StandardScaler().fit_transform(X_train1)
pca = PCA(n_components=n_comp)
X_train1 = pca.fit_transform(X_train1)

X_test1[np.isnan(X_test1)] = np.mean(X_test1[~np.isnan(X_test1)])
X_test1 = StandardScaler().fit_transform(X_test1)
X_test1 = pca.fit_transform(X_test1)

svclassifier1 = BaggingClassifier(SVC(kernel='rbf',degree=8))
svclassifier1.fit(X_train1,y_train1)
svclassifier1.predict(X_test1)


# activity-2 svm
n_comp = 80
X2 = np.zeros((0,126))
y2 = np.zeros((0,0))
for j in range(0,len(act_name2)):
    fio = io.open(path + act_name2(j) + '.csv', 'rt')
    sign2 = csv.reader(fio)
    sign2 = list(sign2)
    sign2 = np.array(sign2).astype(np.float)
    X2 = np.r_['0,2', X2, sign2]
    y2 = np.r_['0,2', y2, j*np.ones((len(sign2), 0))]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3)
X_train2[np.isnan(X_train2)] = np.mean(X_train2[~np.isnan(X_train2)])
X_train2 = StandardScaler().fit_transform(X_train2)
pca = PCA(n_components=n_comp)
X_train2 = pca.fit_transform(X_train2)

X_test2[np.isnan(X_test2)] = np.mean(X_test2[~np.isnan(X_test2)])
X_test2 = StandardScaler().fit_transform(X_test2)
X_test2 = pca.fit_transform(X_test2)

svclassifier2 = BaggingClassifier(SVC(kernel='rbf',degree=8)) #rbf
svclassifier2.fit(X_train2,y_train2)
svclassifier2.predict(X_test2)


# activity-3 svm
n_comp = 80
X3 = np.zeros((0,126))
y3 = np.zeros((0,0))
for j in range(0,len(act_name3)):
    fio = io.open(path + act_name3(j) + '.csv', 'rt')
    sign3 = csv.reader(fio)
    sign3 = list(sign3)
    sign3 = np.array(sign3).astype(np.float)
    X3 = np.r_['0,2', X3, sign3]
    y3 = np.r_['0,2', y3, j*np.ones((len(sign3), 0))]
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.3)
X_train3[np.isnan(X_train3)] = np.mean(X_train3[~np.isnan(X_train3)])
X_train3 = StandardScaler().fit_transform(X_train3)
pca = PCA(n_components=n_comp)
X_train3 = pca.fit_transform(X_train3)

X_test3[np.isnan(X_test3)] = np.mean(X_test3[~np.isnan(X_test3)])
X_test3 = StandardScaler().fit_transform(X_test3)
X_test3 = pca.fit_transform(X_test3)

svclassifier3 = BaggingClassifier(SVC(kernel='rbf',degree=8)) #rbf
svclassifier3.fit(X_train3,y_train3)
svclassifier3.predict(X_test3)


# activity-4 svm
n_comp = 80
X4 = np.zeros((0,126))
y4 = np.zeros((0,0))
for j in range(0, len(act_name4)):
    fio = io.open(path + act_name4(j) + '.csv', 'rt')
    sign4 = csv.reader(fio)
    sign4 = list(sign4)
    sign4 = np.array(sign4).astype(np.float)
    X4 = np.r_['0,2', X4, sign4]
    y4 = np.r_['0,2', y4, j*np.ones((len(sign4), 0))]
X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y4, test_size=0.3)
X_train4[np.isnan(X_train4)] = np.mean(X_train4[~np.isnan(X_train4)])
X_train4 = StandardScaler().fit_transform(X_train4)
pca = PCA(n_components=n_comp)
X_train4 = pca.fit_transform(X_train4)

X_test4[np.isnan(X_test4)] = np.mean(X_test4[~np.isnan(X_test4)])
X_test4 = StandardScaler().fit_transform(X_test4)
X_test4 = pca.fit_transform(X_test4)

svclassifier4 = BaggingClassifier(SVC(kernel='poly',degree=8))
svclassifier4.fit(X_train4,y_train4)
svclassifier4.predict(X_test4)
