#code for Biometric Identification using CNN (shallow CNN architecture)
import numpy as np
from scipy.io import loadmat
from numpy import array
import pyeeg
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
Band=array([8,12,30,100])
power=[]
power_ratio=[]
sum=0
sum=np.array(sum)
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
						count=0
						max=450
						min=-50
						a=nos[m]
						electrode=x[a]
						sum=0
						counter=0

						while max<10000:

							max=max+50
							min=min+50

							final= electrode[min:max]
							power,power_ratio=pyeeg.bin_power(final,Band,1000)
							power_ratio=np.array(power_ratio)
							sum=sum+power_ratio
							counter=counter+1

						

						sum=sum/counter

						if m==0:
							
						 	temp12=sum
						if (m==1):

						 	image=np.append(temp12,sum)
						if m>1:
						 	image=np.append(image,sum)
						m=m+1
						

					l=l+1
					if l==1:
						temp2=image
					    
					elif l==2:
					    image2=np.vstack((temp2,image))
					elif l>2:
					    image2=np.vstack((image2,image))
					label=np.vstack((label,list5[i]))
			        
				if k==0:
					temp3=image2
				if k==1:
					image3=np.vstack((temp3,image2))
				k=k+1
				



			elif (i==0 or i==2 or i==3 or i==4):
				while l<19:
					
					x= data[list4[k]+list3[l]]
					m=0
					while m<11:
						a=nos[m]
						electrode=x[a]
						
						
						count=0
						
						max=450
						min=-50
						sum=0
						counter=0
						while max<10000:
							max=max+50
							min=min+50
							final= electrode[min:max]
							power,power_ratio=pyeeg.bin_power(final,Band,1000)
							power_ratio=np.array(power_ratio)
							sum=sum+power_ratio
							counter=counter+1

						
						sum=sum/counter

						if m==0:
						 	temp5=sum
						if (m==1):
						 	image=np.append(temp5,sum)
						if m>1:
						 	image=np.append(image,sum)
						m=m+1

						
							
						
                     
					if l==0:
					    temp6=image

					elif l==1:
					    image2=np.vstack((temp6,image))
					elif l>1:
					    image2=np.vstack((image2,image))

					l=l+1
					counts=counts+1
					if(counts>1):
							label=np.vstack((label,list5[i]))

			        
				if k==0:
					temp7=image2
				if k==1:
					image3=np.vstack((temp7,image2))
				k=k+1
				
		if j==0:
			temps=image3
		elif j>0:
			image4=np.vstack((temps,image3))
		j=j+1
        

	if i==0:
		temp9=image4
	if i==1:
		output=np.vstack((temp9,image4))
	if i>1:
		output=np.vstack((output,image4))
	i=i+1


X_1=output
Y_1=label
print X_1.shape
print label.shape
print X_1
print label


#code for Biometric Identification using CNN (shallow CNN architecture)
import numpy as np
from scipy.io import loadmat
from numpy import array
import pyeeg
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
Band=array([8,12,30,100])
power=[]
power_ratio=[]
sum=0
sum=np.array(sum)
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
			if (i==1 or i ==5 or i==6 ):
				while l<19:
					x= data[list4[k]+list3[l]]
					m=0
					while m<11:
						count=0
						max=450
						min=-50
						a=nos[m]
						electrode=x[a]
						sum=0
						counter=0

						while max<10000:

							max=max+50
							min=min+50

							final= electrode[min:max]
							power,power_ratio=pyeeg.bin_power(final,Band,1000)
							power_ratio=np.array(power_ratio)
							sum=sum+power_ratio
							counter=counter+1

						

						sum=sum/counter

						if m==0:
							
						 	temp12=sum
						if (m==1):

						 	image=np.append(temp12,sum)
						if m>1:
						 	image=np.append(image,sum)
						m=m+1
						

					l=l+1
					if l==1:
						temp2=image
					    
					elif l==2:
					    image2=np.vstack((temp2,image))
					elif l>2:
					    image2=np.vstack((image2,image))
					label=np.vstack((label,list5[i]))
			        
				if k==0:
					temp3=image2
				if k==1:
					image3=np.vstack((temp3,image2))
				k=k+1
				



			elif (i==0 or i==2 or i==3 or i==4):
				while l<19:
					
					x= data[list4[k]+list3[l]]
					m=0
					while m<11:
						a=nos[m]
						electrode=x[a]
						
						
						count=0
						
						max=450
						min=-50
						sum=0
						counter=0
						while max<10000:
							max=max+50
							min=min+50
							final= electrode[min:max]
							power,power_ratio=pyeeg.bin_power(final,Band,1000)
							power_ratio=np.array(power_ratio)
							sum=sum+power_ratio
							counter=counter+1

						
						sum=sum/counter

						if m==0:
						 	temp5=sum
						if (m==1):
						 	image=np.append(temp5,sum)
						if m>1:
						 	image=np.append(image,sum)
						m=m+1

						
							
						
                     
					if l==0:
					    temp6=image

					elif l==1:
					    image2=np.vstack((temp6,image))
					elif l>1:
					    image2=np.vstack((image2,image))

					l=l+1
					counts=counts+1
					if(counts>1):
							label=np.vstack((label,list5[i]))

			        
				if k==0:
					temp7=image2
				if k==1:
					image3=np.vstack((temp7,image2))
				k=k+1
				
		if j==0:
			temps=image3
		elif j>0:
			image4=np.vstack((temps,image3))
		j=j+1

            
        			
		

        

	if i==0:
		temp9=image4
	if i==1:
		output=np.vstack((temp9,image4))
	if i>1:
		output=np.vstack((output,image4))
	i=i+1


X_2=output
Y_2=label
print X_2.shape
print label.shape
print X_2
print label

X= np.vstack((X_1,X_2))
Y= np.vstack((Y_1,Y_2))
print X.shape
print Y.shape
print X
print Y


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

# fix random seed for reproducibility
numpy.random.seed(8)
# load pima indians dataset

# split into input (X) and output (Y) variables

# fix random seed for reproducibility
seed=7
kfold = KFold(n_splits=6, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X,Y):
	model = Sequential()
	model.add(Dense(80, bias_initializer='he_normal', input_dim=33, activation='relu'))
	model.add(Dense(50, bias_initializer='he_normal', activation='relu'))
	model.add(Dense(30, bias_initializer='he_normal', activation='relu'))
	model.add(Dense(16, bias_initializer='he_normal', activation='relu'))
	model.add(Dense(8, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
	# Fit the model
	model.fit(X[train], Y[train], epochs=400, batch_size=10, verbose=0)
	# evaluate the model
	scores = model.evaluate(X[test], Y[test], verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
























'''
# create model
model = Sequential()
model.add(Dense(80, bias_initializer='he_normal', input_dim=33, activation='relu'))
model.add(Dense(50, bias_initializer='he_normal', activation='relu'))
model.add(Dense(30, bias_initializer='he_normal', activation='relu'))
model.add(Dense(16, bias_initializer='he_normal', activation='relu'))
model.add(Dense(8, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
x=0
total=0
X_train, X_test, y_train,y_test = train_test_split(X,Y,test_size=0.3)
model.fit(X_train,y_train,batch_size=10,epochs=400,validation_data=(X_test,y_test))
total=0
i=0
while i<50:
  X_train, X_test, y_train,y_test = train_test_split(X, Y,test_size=0.3)
  score,acc = model.evaluate(X_test,y_test,batch_size=10)
  classes  = model.predict(X_test, batch_size=10)
  total=total+acc
  i=i+1
  print acc
final=total/50
print final
print counter
'''

