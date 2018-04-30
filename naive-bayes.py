from getEmbeddings import getEmbeddings
from sklearn.naive_bayes import GaussianNB
import numpy as np

xtr,xte,ytr,yte = getEmbeddings("datasets/train.csv")
np.save('./xtr',xtr)
np.save('./xte',xte)
np.save('./ytr',ytr)
np.save('./yte',yte)

xtr = np.load('./xtr.npy')
xte = np.load('./xte.npy')
ytr = np.load('./ytr.npy')
yte = np.load('./yte.npy')

gnb = GaussianNB()
gnb.fit(xtr,ytr)
y_pred = gnb.predict(xte)
m = yte.shape[0]
n = (yte != y_pred).sum()
print("Accuracy = " + format((m-n)/m*100, '.2f') + "%")   # 72.94%
