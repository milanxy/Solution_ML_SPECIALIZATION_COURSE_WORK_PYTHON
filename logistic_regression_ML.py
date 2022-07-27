import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pylab as plt
X= np.loadtxt("ex2data1.txt", usecols=(0,1), delimiter=',')
Y= np.loadtxt("ex2data1.txt", usecols=(2), delimiter=',')
#print(y.shape)
#print(X[:10,:])
m,n =X.shape
print(m,n)


#compute initial cost when w,b is initialized to 0
w_initial =np.zeros(n)
b_initial=0


def sigmoid(z):
    g=1/(1+np.exp(-z))
    return g

def compute_cost(x,y,w,b):
    m, n = x.shape
    loss =0
    for i in range(m):
        z=np.dot(x[i],w)+b
        g=sigmoid(z)
        loss = loss - y[i]*np.log(g)-(1-y[i])*np.log(1-g)
    total_cost = loss/m
    return total_cost
cost = compute_cost(X,Y,w_initial,b_initial)
print(float(cost))


#Compute Gradient to get dj_dw and dj_db from earlier w,b
def compute_gradient(x,y,w,b):
    m,n=X.shape
    dj_dw=np.zeros(n)
    dj_db =0
    for i in range(m):

        f = sigmoid(np.dot(x[i], w)+b)
        err_i = f -y[i]
        for j in range(n):
             dj_dw[j]+=err_i*x[i,j]
        dj_db += err_i
    for j in range(n):        
        dj_dw[j] = dj_dw[j]/m
        
    dj_db = dj_db/m
    return dj_dw, dj_db
#Test with initial w and b
dj_dw, dj_db = compute_gradient(X, Y, w_initial, b_initial)
print(dj_dw)
# Compute gradient descent over iterations
def gradient_descent(x,y,w_in,b_in,cost_fun, gradient_fun, alpha, iteration):
    m,n =X.shape
    J_history = []
    W_history = []
    for i in range(iteration):
        dj_dw, dj_db = gradient_fun(x, y, w_in, b_in)
        for j in range(n):
            w_in[j] =w_in[j] -alpha*dj_dw[j]
        b_in =b_in -alpha*dj_db
        if (i < 100000 ):
            cost = cost_fun(x,y,w_in,b_in)
            J_history.append(cost)
            W_history.append(w_in)
        if (i % 1000 == 0):
            print (cost)
    return w_in, b_in, J_history, W_history
    np.random.seed(1)
intial_w = 0.01 * (np.random.rand(2).reshape(-1,1) - 0.5)
initial_b = -8


# Some gradient descent settings
iterations = 200000
alpha_std = 0.001

w,b, J_history, W_history = gradient_descent(X,Y, w_initial, b_initial, compute_cost, compute_gradient, alpha_std, iterations)

#Now you have weights of your hypothesis model, you need to predict 
def predict(x,w,b):
    m,n=x.shape
    p = np.zeros(m)
    for i in range(m):
        z_wb =0
        f_wb =0
        z_wb = np.dot(x[i],w) +b
        f_wb=sigmoid(z_wb)
        if (f_wb < 0.5):
            p[i] = 0
        else:
            p[i] =1
    return p 
np.random.seed(1)
tmp_w = np.random.randn(2)
tmp_b = 0.3    
tmp_X = np.random.randn(4, 2) - 0.5

tmp_p = predict(tmp_X, tmp_w, tmp_b)
print(f'Output of predict: shape {tmp_p.shape}, value {tmp_p}')
#Compute accuracy on our training set
p = predict(X, w,b)
#print('Train Accuracy: %f'%(np.mean(p == Y) * 100))
Counting =0
for i in range(m):
    if (p[i] == Y[i]):
        Counting += 1

print('Train Accuracy:', float(Counting/m))