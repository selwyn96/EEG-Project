#code for Biometric Identification using CNN (shallow CNN architecture)
import numpy as np
from scipy.io import loadmat
list1=['H','E','N']
list2=['1','2','3','4','5','6']
list5=[[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]
#sub1,sub3,sub4,sub5 -> 10questions
#sub2,sub6-> 9 questions
list3=['01','02','03','04','05','06','07','08','09','10']
i=0
j=0
image=0
temps=0
counts=0
label=[[1,0,0,0,0,0]]
#print "sub"+list2[i]+"_"+list1[j]+".mat"
while i<6:
	print i
	

	j=0
	while j<2:
			
		data=loadmat("sub"+list2[i]+"_"+list1[j]+".mat")
		nos=[0,1,2,3,4,6,10,12,14,16,26]
		k=0
		list4=['YES_Segment_','NO_Segment_']
		while k<2:
			
			l=0
			if (i==1 or i ==5):
				while l<9:
					x= data[list4[k]+list3[l]]
					max=950
					min=-50
					while max<2500:
						m=0
						count=0
						max=max+50
						min=min+50
						
						


						while m<11:

							a=nos[m]
							electrode=x[a]
							final= electrode[min:max]
                            
							if(m==0):
								temp=final
							if(m==1):
								out=np.vstack((temp,final))
							if(m>1):
								out=np.vstack((out,final))
							m=m+1

						if min==0:
							
						 	temp12=out
						if (min==50):

						 	image=np.concatenate([[temp12],[out]])
						if min>50:
						 	image=np.concatenate([image,[out]])
						label=np.vstack((label,list5[i]))

					l=l+1
					if l==1:
						temp2=image
					    
					elif l==2:
					    image2=np.concatenate([temp2,image])
					elif l>2:
					    image2=np.concatenate([image2,image])
			        
				if k==0:
					temp3=image2
				if k==1:
					image3=np.concatenate([temp3,image2])
				k=k+1
				



			elif (i==0 or i==2 or i==3 or i==4):
				while l<10:
					
					x= data[list4[k]+list3[l]]
					max=950
					min=-50
					while max<2500:
						counts=counts+1
						if(counts>1):
							label=np.vstack((label,list5[i]))
						m=0
						count=0
						max=max+50
						min=min+50
						while m<11:
							a=nos[m]
							electrode=x[a]
							final= electrode[min:max]
							if(m==0):
								temp4=final
							if(m==1):
								out=np.vstack((temp4,final))
							if(m>1):
								out=np.vstack((out,final))
							m=m+1
						

						if min==0:
							temp5=out
						elif min==50:
							image=np.concatenate([[temp5],[out]])
						elif min>50:
							image=np.concatenate([image,[out]])

					    
					    
						



                     
					if l==0:
					    temp6=image

					elif l==1:
					    image2=np.concatenate([temp6,image])
					elif l>1:
					    image2=np.concatenate([image2,image])
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


X=output
Y=label
#(7192, 6)----->Label shape
#(7192, 11, 1000)------>output shape

input_shape=(11,1000)
sr=250
             

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
classifier.add(Conv2D(24, (1, 1), input_shape = (7192,11, 1000), activation = 'relu'))
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

classifier.compile(optimizer = 'RMSprop', loss='categorical_crossentropy', metrics = ['accuracy'])


X_train, X_test, y_train,y_test = train_test_split(X, Y,test_size=0.3)
classifier.fit(X_train,y_train,batch_size=10,epochs=2,validation_data=(X_test,y_test))
score,acc = classifier.evaluate(X_test,y_test,batch_size=10)
classes  = classifier.predict(X_test, batch_size=128)

print score
print acc
print classes




