
import numpy as np
# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
 
dataset = np.loadtxt("svm_decision_train.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:18]
z = dataset[:,18:19]
Y= z.flatten()
dataset = np.loadtxt("svm_decision_test.csv", delimiter=",")
x_test = dataset[:,0:18]
m_test= dataset[:,18:19]
y_test=m_test.flatten()
 
# training a linear SVM classifier
from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X, Y)
svm_predictions = svm_model_linear.predict(x_test)
 
# model accuracy for X_test  
accuracy = svm_model_linear.score(x_test, y_test)
 
# creating a confusion matrix
cm = confusion_matrix(y_test, svm_predictions)
print accuracy
print cm
print svm_predictions

