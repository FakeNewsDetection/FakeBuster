from getEmbeddings import getEmbeddings
from sklearn.naive_bayes import GaussianNB
import numpy as np
print("keras")
import keras
print("keras backend")
from keras import backend as K
print("keras np_utils")
from keras.utils import np_utils
print("keras Sequential")
from keras.models import Sequential
print("keras layers")
from keras.layers import Dense, Dropout, LSTM, Embedding, Input, RepeatVector
print("keras SGD")
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder

'''
xtr,xte,ytr,yte = getEmbeddings("./train.csv")

np.save('./xtr',xtr)
np.save('./xte',xte)
np.save('./ytr',ytr)
np.save('./yte',yte)
'''

xtr = np.load('./xtr.npy')
xte = np.load('./xte.npy')
ytr = np.load('./ytr.npy')
yte = np.load('./yte.npy')

'''
gnb = GaussianNB()

y_pred = gnb.fit(xtr,ytr).predict(xtr)

print("Number of mislabeled points out of a total %d points : %d" % (ytr.shape[0],(ytr != y_pred).sum()))
'''
from sklearn.svm import SVC

clf = SVC()
clf.fit(xtr, ytr)
y_pred = clf.predict(xte)
print("Number of mislabeled points out of a total %d points : %d" % (yte.shape[0],(yte != y_pred).sum()))


'''
# ## 4. Define Keras Model

def baseline_model():
    # neural network
    model = Sequential()
    model.add(Dense(256, input_dim=300, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.5))
    model.add(Dense(80, activation='relu', kernel_initializer='normal'))
    model.add(Dense(2, activation="softmax", kernel_initializer='normal'))

    # gradient descent
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    # not 100% sure what "compile" does, maybe just runs the gradient descent algorithm
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


print("define keras model")

model = baseline_model()
model.summary()


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(xtr, ytr, test_size=0.2, random_state=42)

label_encoder = LabelEncoder()
label_encoder.fit(y_train)
encoded_y = np_utils.to_categorical((label_encoder.transform(y_train)))
label_encoder.fit(y_test)
encoded_y_test = np_utils.to_categorical((label_encoder.transform(y_test)))


estimator=model.fit(x_train, encoded_y, validation_data = (x_test, encoded_y_test), epochs=20, batch_size=64)




print("done")
'''
