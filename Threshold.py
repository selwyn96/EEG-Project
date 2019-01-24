# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(8)
# load pima indians dataset
dataset = numpy.loadtxt("complete.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:33]
Y = dataset[:,33:39]
# create model
model = Sequential()
model.add(Dense(50, input_dim=33, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(6, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=500, batch_size=10)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#
dataset = numpy.loadtxt("test.csv", delimiter=",")
x_test = dataset[:,0:33]
y_test= dataset[:,33:39]
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
classes  = model.predict(x_test, batch_size=128)
print classes
print loss_and_metrics
for i in range(len(classes)):
         j=numpy.argmax(classes[i])
         y=classes[i]
         if y[j]>0.90:
         	print numpy.argmax(classes[i])+1
         else:
         	print 'none of the above'
















