
# Random Forest Classification
from sklearn import model_selection
import numpy
from sklearn.ensemble import RandomForestClassifier
dataset = numpy.loadtxt("complete.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:33]
Y = dataset[:,33:39]
seed = 7
num_trees = 100
max_features = 33
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
dataset = numpy.loadtxt("test.csv", delimiter=",")
x_test = dataset[:,0:33]
y_test= dataset[:,33:39]
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, model)
print accuracy
name=[1,2,3,4,5,6]
from sklearn.metrics import confusion_matrix
cm = pd.DataFrame(confusion_matrix(y_test, predicted), columns=name, index=names)
sns.heatmap(cm, annot=True)