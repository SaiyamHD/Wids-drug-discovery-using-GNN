import numpy as np
import pandas as pd
import math
import time

def break_data(x, y, size):
    indices=np.arange(size)
    np.random.shuffle(indices)
    x=x[indices] 
    y=y[indices]
    break_pt=int(0.8*size)
    x_train,x_test=x[:break_pt,],x[break_pt:,]
    y_train,y_test=y[:break_pt,],y[break_pt:,]
    return x_train, x_test, y_train, y_test
def test(x,y,w,b):
    size=y.shape[0]
    z=np.dot(x,w)+b
    #print (z)
    f=1/(1+np.exp(-z))
    print (f.shape)
    predictions = (f >= 0.45).astype(int)
    result=predictions-y
    print (f"Percentage right: {(np.mean(result==0)*100): .4f}%")
    print (f"Percentage false positives: {float(np.count_nonzero(result==1)*100)/size: .4f}%")
    print (f"Percentage false negatives: {float(np.count_nonzero(result==-1)*100)/size: .4f}%")
def normalise(x,columns):
    x[:, columns] = (x[:, columns] - np.mean(x[:, columns], axis=0)) / np.std(x[:, columns], axis=0)

start_time=time.time()
data=pd.read_csv(r"E:\Academics in IITB\Wids Drug Discovery using GNN\Heart disease data.csv")
data=data.dropna()

x=data[['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']].values
y=data[['TenYearCHD']].values
size=y.shape[0]
normalise(x,[1,4,9,10,11,12,13,14])
x_train,x_test,y_train,y_test=break_data(x,y,size)

t=float(0.1)
w=np.zeros(x_train.shape[1], dtype=float)
b=float(0)
prev_cost=float(1)
curr_cost=float(0)
y_train = y_train.ravel() 
y_test= y_test.ravel()

while abs( prev_cost-curr_cost) > 1e-6:
    prev_cost=curr_cost
    z=np.dot(x_train,w)+b
    f = 1 / (1 + np.exp(-np.clip(z, -500, 500))) 
    curr_cost= np.mean(-np.multiply(y_train,np.log(f))-np.multiply((1-y_train),np.log(1-f)))
    grad_w=(np.dot((f-y_train),x_train))/(0.8*size)
    grad_b=np.mean(f-y_train)
    w-=grad_w*t
    b-=grad_b*t

end_time=time.time()
time_taken=end_time-start_time

test(x_test,y_test,w,b)
print(f"Weights: {np.around(w, decimals=4)}")
print(f"Bias: {b: .4f}")
print(f"Time taken= {time_taken: .4f} s" )
