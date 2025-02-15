from os import listdir
import numpy as np
import cv2
import math
from sklearn.svm import SVR
import scipy.io as sio
from scipy.stats import pearsonr
import pandas as pd

def sigmoid(x):#Sigmoid activation function
  return 1 / (1 + math.exp(-x))


def conv(X_matrix,filter):#Does convolution for a matrix and filter
	X=np.zeros([len(X_matrix)-len(filter)+1,len(X_matrix[0])-len(filter)+1])
	for j in range(0,len(X_matrix)-len(filter)+1,1):
		for k in range(0,len(X_matrix[0])-len(filter[0])+1,1):
			for m in range(0,len(filter),1):
				for n in range(0,len(filter[0]),1):
					X[j,k]+=X_matrix[j+m][k+n]*filter[m][n];
	
	for j in range(0,len(X_matrix)-len(filter)+1,1):#Applying ReLU Activation function
		for k in range(0,len(X_matrix[0])-len(filter[0])+1,1):
				if X[i][j]<0:
					X[i][j]=0;

	return X;

def pool(X_matrix,filter,stride):#Does max pooling for X_matrix
	n_h=math.ceil((len(X_matrix)-len(filter))/stride)+1
	n_w=math.ceil((len(X_matrix[0])-len(filter))/stride)+1
	X=np.zeros([n_h,n_w]);
	index=np.zeros([n_h,n_w]);
	for j in range(0,n_h,1):
		for k in range(0,n_w,1):
			for m in range(0,len(filter),1):
				for n in range(0,len(filter[0]),1):
					if X[j,k]<X_matrix[j+m][k+n]:
						X[j,k]=X_matrix[j+m][k+n];	
						index[j,k]=(j+m,k+n)
	return X,index;


def fully_connected(X_matrix,filter):
	X = np.asarray(X_matrix).reshape(-1)
	output=0;
	for i in range(0,len(X)):
		output+=X[i]*filter[i];
	sig_output=sigmoid(output)
	return sig_output;



def get_feature(img1,img2,y,img1_name,img2_name,value):
	
	u1,s1,v1=np.linalg.svd(img1,full_matrices=True)
	u2,s2,v2=np.linalg.svd(img2,full_matrices=True)
	v1=np.transpose(v1)
	v2=np.transpose(v2)
	temp_a=[]
	feature=[]
	rows=min(len(u1),len(v1))
	if len(img1)==len(img2) and len(img1[0])==len(img2[0]):
		for i in range(0,rows):
			r=np.array(u1[i])
			r1=np.array(u2[i])
			rc=np.array(v1[i])
			rc1=np.array(v2[i])
			x=np.dot(r,r1)
			x1=np.dot(rc,rc1)
			feature.append((np.add(x,x1)))
		for i in range(len(feature),512):
			feature.append((0))
		np_feature=np.array(feature);
		np_feature=np_feature.reshape(32,16)
		print(len(np_feature),len(np_feature[0]))
		temp_a.append({"y":y,"x":np_feature,"img1":img1_name,"img2":img2_name})
	
	return temp_a;

def get_training_images(training_data,dmos):
	
	file = open("./Live Database/databaserelease2/wn/info.txt","r") 
	list_file_wn=file.readlines();
	file.close();

	x=[]
	
	l=len(list_file_wn);
	
	for i in range(0,l,1):
		list_file_wn[i]=list_file_wn[i].split();
	
	j=0
	for i in range(0,l,2):
		if list_file_wn[i][0] in training_data:
			img=cv2.imread("./Live Database/databaserelease2/wn/"+str(list_file_wn[i][1]),0)
			s=int(list_file_wn[i][1][3:len(list_file_wn[i][1])-4])
			kk=get_feature(training_data[list_file_wn[i][0]],img,dmos[459+s],list_file_wn[i][0],"wn/"+list_file_wn[i][1],0)
			if len(kk)!=0:
				x.append(kk)
		j=j+1

	file = open("./Live Database/databaserelease2/fastfading/info.txt","r") 
	list_file_wn=file.readlines();
	file.close();
	l=len(list_file_wn);
	
	for i in range(0,l,1):
		list_file_wn[i]=list_file_wn[i].split();
	
	j=0
	for i in range(1,l,1):
		if list_file_wn[i][0] in training_data:
			img=cv2.imread("./Live Database/databaserelease2/fastfading/"+str(list_file_wn[i][1]),0)
			s=int(list_file_wn[i][1][3:len(list_file_wn[i][1])-4])
			kk=get_feature(training_data[list_file_wn[i][0]],img,dmos[807+s],list_file_wn[i][0],"fastfading/"+list_file_wn[i][1],0)
			if len(kk)!=0:
				x.append(kk)
		j=j+1

	file = open("./Live Database/databaserelease2/gblur/info.txt","r") 
	list_file_wn=file.readlines();
	file.close();
	l=len(list_file_wn);
	
	for i in range(0,l,1):
		list_file_wn[i]=list_file_wn[i].split();
	
	j=0
	for i in range(0,l,2):
		if list_file_wn[i][0] in training_data:
			img=cv2.imread("./Live Database/databaserelease2/gblur/"+str(list_file_wn[i][1]),0)
			s=int(list_file_wn[i][1][3:len(list_file_wn[i][1])-4])
			kk=get_feature(training_data[list_file_wn[i][0]],img,dmos[633+s],list_file_wn[i][0],"gblur/"+list_file_wn[i][1],0)
			if len(kk)!=0:
				x.append(kk)
		j=j+1

	file = open("./Live Database/databaserelease2/jp2k/info.txt","r") 
	list_file_wn=file.readlines();
	file.close();
	l=len(list_file_wn);
	
	for i in range(0,l,1):
		list_file_wn[i]=list_file_wn[i].split();
	
	j=0
	for i in range(0,l,1):
		if list_file_wn[i][0] in training_data:
			img=cv2.imread("./Live Database/databaserelease2/jp2k/"+str(list_file_wn[i][1]),0)
			s=int(list_file_wn[i][1][3:len(list_file_wn[i][1])-4])
			kk=get_feature(training_data[list_file_wn[i][0]],img,dmos[-1+s],list_file_wn[i][0],"jp2k/"+list_file_wn[i][1],0)
			if len(kk)!=0:
				x.append(kk)
		j=j+1

	file = open("./Live Database/databaserelease2/jpeg/info.txt","r") 
	list_file_wn=file.readlines();
	file.close();
	l=len(list_file_wn);
	
	for i in range(0,l,1):
		list_file_wn[i]=list_file_wn[i].split();
	
	j=0
	for i in range(0,l,1):
		if list_file_wn[i][0] in training_data:
			img=cv2.imread("./Live Database/databaserelease2/jpeg/"+str(list_file_wn[i][1]),0)
			s=int(list_file_wn[i][1][3:len(list_file_wn[i][1])-4])
			kk=get_feature(training_data[list_file_wn[i][0]],img,dmos[226+s],list_file_wn[i][0],"jpeg/"+list_file_wn[i][1],0)
			if len(kk)!=0:
				x.append(kk)
		j=j+1
	

	out_x=[]
	out_y=[]
	out_img1=[]
	out_img2=[]

	for i in range(0,len(x)):
		if len(x[i][0]["x"])!=0: 
			out_x.append(x[i][0]["x"])
			out_y.append(x[i][0]["y"])
			out_img1.append(x[i][0]["img1"])
			out_img2.append(x[i][0]["img2"])
	return out_x,out_y,out_img1,out_img2;

def main():
	ref_images_name=listdir("./Live Database/databaserelease2/refimgs")#To get list of files available in the refimg folder.
	
	
	print(len(ref_images_name));

	train=math.floor(len(ref_images_name)*0.9)#10 fold cross validation
	print(train)
	counter=0;
	mat_contents = sio.loadmat('dmos.mat')
	dmos=mat_contents["dmos"];
	for i in range(0,len(ref_images_name)):
		im=cv2.imread("./Live Database/databaserelease2/refimgs/"+str(ref_images_name[i]),0)
		images[ref_images_name[i]]=im;
		
# for fold in range(1,10):#Doing 10 fold cross validation
	train_images={}
	test_images={}
	
	for i in range(0,counter):
		# img=cv2.imread("./Live Database/databaserelease2/refimgs/"+str(ref_images_name[i]),0)
		train_images[ref_images_name[i]]=images[ref_images_name[i]]

	if counter==27:
		for i in range(0,2):
			img=cv2.imread("./Live Database/databaserelease2/refimgs/"+str(ref_images_name[counter]),0)
			test_images[ref_images_name[counter]]=images[ref_images_name[counter]]
			counter+=1
	else:
		for i in range(0,3):
			img=cv2.imread("./Live Database/databaserelease2/refimgs/"+str(ref_images_name[counter]),0)
			test_images[ref_images_name[counter]]=images[ref_images_name[counter]]
			counter+=1

	for i in range(counter,len(ref_images_name)):
		img=cv2.imread("./Live Database/databaserelease2/refimgs/"+str(ref_images_name[i]),0)
		train_images[ref_images_name[i]]=images[ref_images_name[i]]
	
	
	X,Y,corr_img,dist_img=get_training_images(train_images,dmos[0])
	print(X[0],Y[0])

	max_features=0;
	filter=[[1,1,1],[1,1,1],[1,1,1]]#Filter for 1st forward pass, It will get modified by Stochastic gradient
	filter_pool=[[0,0],[0,0]]
	filter_cc=[1]*105
	stride=2
	learning_rate=0.1
	for i in range(0,len(X)):
		
		#Forward Propagation

		relu_cnn_output=conv(X[i],filter);
		maxpool_output,index_pool=pool(relu_cnn_output,filter_pool,stride)
		
		input_full= np.asarray(maxpool_output).reshape(-1)
		output=fully_connected(maxpool_output,filter_cc)
		y=Y[0]/100.0
		
		#Backward Propagation for parameters of fully connected layer
		for j in range(0,105,1):
			filter_cc[j]=filter_cc[j]+learning_rate*(Y[0]-output)*output*(1-output)*input_full[j];

		#Backward Propagation for parameter of Covolutional Layer
		for j in range(0,3,1):
			for k in range(0,3,1):
				#Write Backprop equation.

	# print("PLCC for Training Data:")
	# print(pearsonr(y_rbf,Y))

	x_test,y_test,corr_img_test,dist_img_test=get_training_images(test_images,dmos[0]);

	# T = np.zeros((len(x_test),min_features));

	# for i in range(0,len(x_test)):
	# 	for j in range(0,len(x_test[i])):
	# 		T[i][j]=x_test[i][j]
	
	# y_rbf_test=fi.predict(T)

	# print("PLCC for Testing Data:")
	# print(pearsonr(y_test,y_rbf_test))

# main();
print(fully_connected(conv([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]],[[0,1,0],[0,1,0],[0,1,0]])));