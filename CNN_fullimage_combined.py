#code for Biometric Identification using CNN (shallow CNN architecture)
import numpy as np
from scipy.io import loadmat
list1=['H','E','N']
list2=['1','2','3','4','5','6','7','8']
list5=[[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]]
#sub1,sub3,sub4,sub5 -> 10questions
#sub2,sub6-> 9 questions
list3=['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19']
i=0
j=0
image=0
temps=0
counts=0
label=[[1,0,0,0,0,0,0,0]]
print "sub"+list2[i]+"_"+list1[j]+".mat"
while i<8:
	print i
	

	j=0
	while j<2:
			
		data=loadmat("sub"+list2[i]+"_"+list1[j]+"_1"+".mat")
		nos=[0,1,2,3,4,6,10,12,14,16,26]
		k=0
		list4=['Yes_Segment_','No_Segment_']
		while k<2:
			
			l=0
			if (i==1 or i ==5 or i==6 or i==7):
				while l<19:
					x= data[list4[k]+list3[l]]
					m=0
					

					while m<11:

						a=nos[m]
						electrode=x[a]
						final= electrode
                            
						if(m==0):
							temp=final
						if(m==1):
							out=np.vstack((temp,final))
						if(m>1):
							out=np.vstack((out,final))
						m=m+1

						
					label=np.vstack((label,list5[i]))

					l=l+1
					if l==1:
						temp2=out
					    
					elif l==2:
					    image2=np.concatenate([[temp2],[out]])
					elif l>2:
					    image2=np.concatenate([image2,[out]])
			        
				if k==0:
					temp3=image2
				if k==1:
					image3=np.concatenate([temp3,image2])
				k=k+1
				



			elif (i==0 or i==2 or i==3 or i==4):
				while l<19:
					
					x= data[list4[k]+list3[l]]
					
					
					counts=counts+1
					if(counts>1):
						label=np.vstack((label,list5[i]))
					m=0
					
					while m<11:

						a=nos[m]
						electrode=x[a]
						final= electrode
						if(m==0):
							temp4=final
						if(m==1):
							out=np.vstack((temp4,final))
						if(m>1):
							out=np.vstack((out,final))
						m=m+1
						

          
					if l==0:
					    temp6=out

					elif l==1:
					    image2=np.concatenate([[temp6],[out]])
					elif l>1:
					    image2=np.concatenate([image2,[out]])
					l=l+1
			        
				if k==0:
					temp7=image2
				if k==1:
					image3=np.concatenate([temp7,image2])
				k=k+1
				
		if j==0:
			temps=image3
		elif j>0:
			image4=np.concatenate([temps,image3])
		j=j+1

            
        			
		

        

	if i==0:
		temp9=image4
	if i==1:
		output=np.concatenate([temp9,image4])
	if i>1:
		output=np.concatenate([output,image4])
	i=i+1


print output.shape
print label.shape


X_1=output
Y_1=label

#code for Biometric Identification using CNN (shallow CNN architecture)
import numpy as np
from scipy.io import loadmat
list1=['H','E','N']
list2=['1','2','3','4','5','6','7','8']
list5=[[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]]
#sub1,sub3,sub4,sub5 -> 10questions
#sub2,sub6-> 9 questions
list3=['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19']
i=0
j=0
image=0
temps=0
counts=0
label=[[1,0,0,0,0,0,0,0]]
print "sub"+list2[i]+"_"+list1[j]+".mat"
while i<7:
	print i
	

	j=0
	while j<2:
			
		data=loadmat("sub"+list2[i]+"_"+list1[j]+"_2"+".mat")
		nos=[0,1,2,3,4,6,10,12,14,16,26]
		k=0
		list4=['Yes_Segment_','No_Segment_']
		while k<2:
			
			l=0
			if (i==1 or i ==5 or i==6 or i==7):
				while l<19:
					x= data[list4[k]+list3[l]]
					m=0
					

					while m<11:

						a=nos[m]
						electrode=x[a]
						final= electrode
                            
						if(m==0):
							temp=final
						if(m==1):
							out=np.vstack((temp,final))
						if(m>1):
							out=np.vstack((out,final))
						m=m+1

						
					label=np.vstack((label,list5[i]))

					l=l+1
					if l==1:
						temp2=out
					    
					elif l==2:
					    image2=np.concatenate([[temp2],[out]])
					elif l>2:
					    image2=np.concatenate([image2,[out]])
			        
				if k==0:
					temp3=image2
				if k==1:
					image3=np.concatenate([temp3,image2])
				k=k+1
				



			elif (i==0 or i==2 or i==3 or i==4):
				while l<19:
					
					x= data[list4[k]+list3[l]]
					
					
					counts=counts+1
					if(counts>1):
						label=np.vstack((label,list5[i]))
					m=0
					
					while m<11:

						a=nos[m]
						electrode=x[a]
						final= electrode
						if(m==0):
							temp4=final
						if(m==1):
							out=np.vstack((temp4,final))
						if(m>1):
							out=np.vstack((out,final))
						m=m+1
						

          
					if l==0:
					    temp6=out

					elif l==1:
					    image2=np.concatenate([[temp6],[out]])
					elif l>1:
					    image2=np.concatenate([image2,[out]])
					l=l+1
			        
				if k==0:
					temp7=image2
				if k==1:
					image3=np.concatenate([temp7,image2])
				k=k+1
				
		if j==0:
			temps=image3
		elif j>0:
			image4=np.concatenate([temps,image3])
		j=j+1

            
        			
		

        

	if i==0:
		temp9=image4
	if i==1:
		output=np.concatenate([temp9,image4])
	if i>1:
		output=np.concatenate([output,image4])
	i=i+1

print output.shape
print label.shape


X_2=output
Y_2=label


X= np.concatenate([X_1,X_2])
Y= np.concatenate([Y_1,Y_2])


input_shape=(11,10000)
sr=1000
             
'''
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from kapre.time_frequency import Spectrogram
from kapre.utils import Normalization2D
from kapre.augmentation import AdditiveNoise
import keras
from keras import optimizers
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


classifier = Sequential()

classifier.add(Spectrogram(n_dft=512, n_hop=256,padding='same',input_shape=input_shape, 
	power_spectrogram=2.0,return_decibel_spectrogram=False, trainable_kernel=False,image_data_format='default'))

classifier.add(AdditiveNoise(power=0.2))
classifier.add(Normalization2D(str_axis='freq'))
#Layer 1
classifier.add(Conv2D(24, (1, 1), input_shape = (232,11, 1000), activation = 'relu'))
keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dense(units = 128, activation = 'relu'))
keras.layers.Dropout(0.5, noise_shape=None, seed=None)
#Layer 2
classifier.add(Conv2D(48, (1,1), input_shape = (32, 32,24), activation = 'relu'))
keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dense(units = 128, activation = 'relu'))
keras.layers.Dropout(0.5, noise_shape=None, seed=None)
#Layer 3
classifier.add(Conv2D(96, (1,1), input_shape = (32, 32,24), activation = 'relu'))
keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
classifier.add(MaxPooling2D(pool_size = (1, 1)))
classifier.add(Dense(units = 128, activation = 'relu'))
keras.layers.Dropout(0.5, noise_shape=None, seed=None)
#Layer 4
classifier.add(Flatten())
classifier.add(Dense(units = 6, activation = 'softmax'))

classifier.compile(optimizer = 'SGD', loss='categorical_crossentropy', metrics = ['accuracy'])

X_train, X_test, y_train,y_test = train_test_split(X, Y,test_size=0.1)
classifier.fit(X_train,y_train,batch_size=10,epochs=8,validation_data=(X_test,y_test))
score,acc = classifier.evaluate(X_test,y_test,batch_size=10)
classes  = classifier.predict(X_test, batch_size=128)

print score
print acc
print classes
'''

# Create your first MLP in Keras
import itertools
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
import matplotlib.pyplot as plt
from keras import optimizers
from sklearn.metrics import confusion_matrix
import numpy
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from kapre.time_frequency import Spectrogram
from kapre.utils import Normalization2D
from kapre.augmentation import AdditiveNoise
import keras
from keras import optimizers
import tensorflow as tf
from sklearn.metrics import accuracy_score

# fix random seed for reproducibility
numpy.random.seed(8)
# load pima indians dataset

# split into input (X) and output (Y) variables

# fix random seed for reproducibility
seed=7
kfold = KFold(n_splits=6, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X,Y):
	classifier = Sequential()
	classifier.add(Spectrogram(n_dft=512, n_hop=256,padding='same',input_shape=input_shape, 
	power_spectrogram=2.0,return_decibel_spectrogram=False, trainable_kernel=False,image_data_format='default'))
	classifier.add(AdditiveNoise(power=0.2))
	classifier.add(Normalization2D(str_axis='freq'))
	classifier.add(Conv2D(24, (4, 4), input_shape = (64,64,11), activation = 'relu'))
	keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
	classifier.add(MaxPooling2D(pool_size = (2, 2)))
	classifier.add(Dense(units = 128, activation = 'relu'))
	keras.layers.Dropout(0.5, noise_shape=None, seed=None)
	classifier.add(Conv2D(48, (3,3), input_shape = (32, 32,11), activation = 'relu'))
	keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
	classifier.add(MaxPooling2D(pool_size = (2, 2)))
	classifier.add(Dense(units = 128, activation = 'relu'))
	keras.layers.Dropout(0.5, noise_shape=None, seed=None)
	classifier.add(Conv2D(96, (2,2), input_shape = (16, 16,11), activation = 'relu'))
	keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
	classifier.add(MaxPooling2D(pool_size = (2, 2)))
	classifier.add(Dense(units = 128, activation = 'relu'))
	keras.layers.Dropout(0.5, noise_shape=None, seed=None)
	classifier.add(Flatten())
	classifier.add(Dense(units = 8, activation = 'softmax'))
	classifier.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['accuracy'])
	
	
	# Fit the model
	classifier.fit(X[train], Y[train], epochs=15, batch_size=10, verbose=0)
	# evaluate the model
	scores = classifier.evaluate(X[test], Y[test], verbose=0)
	print("%s: %.2f%%" % (classifier.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))












