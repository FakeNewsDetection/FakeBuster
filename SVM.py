class SVM:
    
    def __init__(self, kernel, C):
        self.kernel = 'linear'
        self.C = 1.0
        self.clf = svm.SVC(kernel= self.kernel, C = self.C)
        
    def train(self, train_inputs,train_labels):
        model = self.clf.fit(train_inputs,train_labels)
        return model
        
    def test(self, test_inputs):
        prediction = self.clf.predict(test_inputs)
        return prediction
