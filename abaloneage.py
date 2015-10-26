#to do:
#PCA
#confusion matrix
#GBR (done?)
#



#Predicting the age of abalones by using various attributes
#the age is determined by counting the rings on shells
#the data is labeled

#total of 28 different possible ring numbers in this dataset. Using classification,
#it procudes 28 classes, tough for classification.
#idea enlightened from Johannes FurnKranz's paper: Pairwise Classification as an Ensemble Technique
import numpy as np
import sys
import csv
from sklearn.cross_validation import train_test_split 
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

#using stdin to read the input files
#format for command: pyhton abalone_age.py abalone_rings.txt

#parsing info into matrices with numpy
#require the input format to be csv
#need to import csv and numpy as np
def csv_parsing (raw_data, header=False):
	with open(raw_data) as f:
		data=csv.reader(f)
		attributes=[]
		tag=[]

		for entry in data:
			if header:
				header=False
			else:
				attributes.append(entry[0:-1])
				tag.append(entry[-1])
		attributes=np.array(attributes)
		tag=np.array(tag)

	return attributes, tag




def K_NNR(train_x, train_y, test_x, test_y, k, normalization = True):
	if normalization:
		#train_x_new=train_x/train_x.max(axis=0)
		#test_x_new=test_x/test_x.max(axis=0)
		mean=train_x.mean(axis=0)
		std=train_x.std(axis=0)
		train_x_new=(train_x-mean)/std
		test_mean=test_x.mean(axis=0)
		test_std=test_x.std(axis=0)
		test_x_new=(test_x-test_mean)/test_std

	else:
		train_x_new=train_x
		test_x_new=test_x

	#using euclidean distance to determine the closeness of two points
	prediction=[]
	for i in xrange(len(test_x_new)):
		
		candidates=[]
		#calcluate the euclidean distance and find the indices of those
		#having the shortest distance
		shortest_distance_idx=np.argsort(\
			[np.linalg.norm(train_x_row - test_x_new[i])for train_x_row in train_x_new])

		#find the k nearest neighbors
		for j in range(k):
			candidates.append(train_y[shortest_distance_idx[j]])
		#select the neighbor that has the most frequent appearance
		#if there's a tie, choose the first one
		prediction.append(np.bincount(candidates).argmax())
	
	#checking the accuracy
	count=0

	for l in range(len(test_y)):
		#if prediction[l]<=test_y[l]+1 and prediction[l]>=test_y[l]-1:
		if prediction[l]==test_y[l]:
			count+=1
	accuracy=float(count)/len(test_y)

	print "%d-NNR: The accuracy of %d nearest neighbor is %f" % (k,k, accuracy)

def K_NNR_sklearn(train_x, train_y, test_x, test_y, k, normalization = True):
	K_NNR_model=KNN(n_neighbors=k)
	K_NNR_model.fit(train_x, train_y)
	accuracy=K_NNR_model.score(test_x, test_y)

	print "%d-NNR using sklearn: The accuracy is %f" %(k, accuracy)

def random_forest(train_x, train_y, test_x, test_y):

	#create model
	forest=RF(n_estimators=100)
	#fit the training data into model
	forest=forest.fit(train_x, train_y)
	#predict the outcome using test value
	accuracy = forest.score(test_x, test_y)

	print "Random Forest: the accuracy is %f" %accuracy

def GradBR(train_x, train_y, test_x, test_y):
	regressor=GBC()
	regressor.fit(train_x, train_y)
	accuracy= regressor.score(test_x, test_y)

	print "Gradient Boosted Regressor: accuracy is %f" %accuracy
#checking the length of input

def confu_mat_random_forest(train_x, train_y, test_x, test_y):
	forest=RF(n_estimators=100)
	forest=forest.fit(train_x, train_y)
	prediction=forest.predict(test_x)
	conf_mat=confusion_matrix(test_y, prediction)

	return conf_mat

try:
	len(sys.argv)>1
except IndexError:
	print 'command format should be "python file_name data_dame".'
else:
	abalone_attri, abalone_tag = csv_parsing(sys.argv[1])
	#changing M, F, I to 1, 2, and 3, respectively
	for i in xrange(len(abalone_attri)):
		if abalone_attri[i][0]=='M':
			abalone_attri[i][0]=1
		elif abalone_attri[i][0]=='F':
			abalone_attri[i][0]=2
		else:
			abalone_attri[i][0]=3

	#convert all the values into numeric from strings
	abalone_attri=abalone_attri.astype(np.float)
	abalone_tag=abalone_tag.astype(np.float)

	#reducing number of classes by combining classes
	#max is 29 (28 classes, missing ring 28)
	#so divide into 7 classes, each class holding 4 old-classes
	
	tag_mapping={(1,4):0, (5,8):1, (9,12):2, (13,16):3, (17,20):4, (21,24):5, (25,29):6}
	new_tag=[]
	for tag in abalone_tag:
		for m in tag_mapping:
			if m[0]<=tag<=m[1]:
				new_tag.append(tag_mapping[m])
	abalone_tag=new_tag
	
	


	#using PCA to decrease the number of components
	pca= PCA(5)
	pca.fit(abalone_attri)
	#print pca.get_covariance()
	abalone_attri=pca.fit_transform(abalone_attri)

	#splitting the data into train samples and test samples
	#by half and half
	train_x, test_x, train_y, test_y = \
	train_test_split(abalone_attri, abalone_tag, test_size=0.5, random_state=1)
	
	#using K-NNR algorithm
	#K_NNR(train_x, train_y, test_x, test_y, 15, normalization=True)

	#sklearn's K-NNR algorithm
	K_NNR_sklearn(train_x, train_y, test_x, test_y, 15)


	#using random forest algorithm
	#with 100 estimators
	random_forest(train_x, train_y,test_x, test_y)

	#Gradient Boosted Classifier
	GradBR(train_x, train_y, test_x, test_y)

	#qucik check of the confusion matrix for random forest
	conf_mat=confu_mat_random_forest(train_x, train_y, test_x, test_y)
	plt.matshow(conf_mat)
	plt.title('Confusion matrix')
	plt.colorbar()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()




