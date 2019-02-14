
# coding: utf-8

# In[1]:


#load_data
import scipy.io
import numpy as np
import mne
from scipy import signal
data=scipy.io.loadmat('dataset_BCIcomp1.mat')
data_test=data['x_test']
data_train=data['x_train']
label_train=data['y_train'].reshape(1,-1)-1
label=scipy.io.loadmat('y_test.mat')
label_test=label['y_test'].reshape(1,-1)-1
print(label_test.shape)
print(label_train.shape)
y_train=label_train[0]
y_test=label_test[0]
print(y_train.shape)
print(y_test.shape)
b,a=signal.butter(8,[(16/128),(64/128)],'bandpass')
buffer_x_test=signal.filtfilt(b,a,data_test,axis=0)
buffer_x_train=signal.filtfilt(b,a,data_train,axis=0)
print(buffer_x_test.shape)
all_x_train=np.transpose(buffer_x_train,[2,1,0])
all_x_test=np.transpose(buffer_x_test,[2,1,0])
x_train=all_x_train[:,0::2,448:896]
print(x_train.shape)
x_test=all_x_test[:,0::2,448:896]
print(x_test.shape)


# In[2]:


#decompose and reconstruct EEG signals
import pywt
db4=pywt.Wavelet('db4')
cA3,cD3,cD2,cD1= pywt.wavedec(x_train[1,1,:],db4,mode='symmetric',level=3)
print(cD2.shape)
print(cA3.shape)
print(cD1.shape)
print(cD3.shape)
cD2=np.zeros(117)
cA3=np.zeros(62)
cD1=np.zeros(227)
x_zz3=pywt.waverec([cA3,cD3,cD2,cD1],db4)
import pywt
db4=pywt.Wavelet('db4')
def Dwt(X):
    cA3,cD3,cD2,cD1 = pywt.wavedec(X,db4,mode='symmetric',level=3)
    return cA3,cD3,cD2,cD1
def cD3_features(x):
    Bands_D3=np.empty((x.shape[0],x.shape[1],448))
    for i in range(x.shape[0]):
        for ii in range(x.shape[1]):
            cA3,cD3,cD2,cD1=Dwt(x[i,ii,:])
            cA3=np.zeros(62)
            cD2=np.zeros(117)
            cD1=np.zeros(227)
            Bands_D3[i,ii,:]=pywt.waverec([cA3,cD3,cD2,cD1],db4)
    return Bands_D3
def cD2_features(x):
    Bands_D2=np.empty((x.shape[0],x.shape[1],448))
    for i in range(x.shape[0]):
        for ii in range(x.shape[1]):
            cA3,cD3,cD2,cD1=Dwt(x[i,ii,:])
            cA3=np.zeros(62)
            cD3=np.zeros(62)
            cD1=np.zeros(227)
            Bands_D2[i,ii,:]=pywt.waverec([cA3,cD3,cD2,cD1],db4)
    return Bands_D2
x_train_d3=cD3_features(x_train)
x_train_d2=cD2_features(x_train)
x_test_d3=cD3_features(x_test)
x_test_d2=cD2_features(x_test)
print(x_train_d3.shape)
print(x_test_d3.shape)
print(x_train_d2.shape)
print(x_test_d2.shape)


# In[3]:


#calcutue AR coef of each  channel in EEG signals of different frequency bands
from statsmodels.tsa.ar_model import AR
def get_ARcoef(x):
    model=AR(x)
    model_fit=model.fit(maxlag=5)
    coef=model_fit.params
    return coef
def get_features(x,y):
    for i in range(140):
        for j in range(2):
            y[i,j]=get_ARcoef(x[i,j])
    return y


# In[5]:


d3_train_coef=np.zeros((140,2,6))
d3_test_coef=np.zeros((140,2,6))
d2_train_coef=np.zeros((140,2,6))
d2_test_coef=np.zeros((140,2,6))
d3_train_features=get_features(x_train_d3,d3_train_coef)
d2_train_features=get_features(x_train_d2,d2_train_coef)
d3_test_features=get_features(x_test_d3,d3_test_coef)
d2_test_features=get_features(x_test_d2,d2_test_coef)
print(d3_train_features.shape)


# In[6]:


#concatenate and normalize the coef as extracted features
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
acc=[]
ss = preprocessing.StandardScaler()
X_train=ss.fit_transform(np.concatenate((d3_train_features[:,0,:],d3_train_features[:,1,:],d2_train_features[:,0,:],d2_train_features[:,1,:]),axis=1))
X_test=ss.transform(np.concatenate((d3_test_features[:,0,:],d3_test_features[:,1,:],d2_test_features[:,0,:],d2_test_features[:,1,:]),axis=1))
print(X_train.shape)
print(X_test.shape)


# In[7]:


#use SVM to classify the features 
from sklearn import svm
clf=svm.SVC(C=0.8,kernel='rbf')
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print(y_test)
print(y_pred)
acc=accuracy_score(y_test,y_pred)
print(acc)

