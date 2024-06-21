import sklearn
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
from sklearn import svm, metrics
import pickle
import os

#load data
data = data=sklearn.datasets.load_digits()
parameters = {
    'kernel': 'rbf',       # or 'linear', 'poly', 'sigmoid'
    'Tol': 1e-3,           # tolerance for stopping criteria
    'Max_Iteration': 1000  # maximum number of iterations
}

#Split the data into test andtrain
train_data, test_data, train_labels, test_labels = sklearn.model_selection.train_test_split(data.data, data.target, train_size=0.80)
print(train_data,train_labels)

#Train a model using random
model = svm.SVC(kernel=parameters['kernel'],tol=float(parameters['Tol']),max_iter=int(parameters['Max_Iteration'])).fit(train_data,train_labels)

#test the model
result = model.score(test_data, test_labels)
print(result)

#save the model
filename = 'digits_model.pkl'
pickle.dump(model, open(filename, 'wb'))
