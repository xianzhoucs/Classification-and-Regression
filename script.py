import numpy as np
from scipy.optimize import minimize
import math
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

# u = np.unique(y)
# k = u.shape[0]
# u contains '1, 2, 3, 4, 5', y contains 5 labels

def ldaLearn(X, y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix

    # IMPLEMENT THIS METHOD
    u = np.unique(y)
    k = u.shape[0]
    n = X.shape[0]
    d = X.shape[1]

    Xy = np.concatenate((X, y), axis = 1)
    means = np.zeros([k, d + 1], float)

    for i in range(1, k + 1):
        Xy_i = Xy[Xy[:, d] == u[i - 1], :]
        means_i = Xy_i.mean(axis = 0)
        means[i - 1, :] = means_i
    means = np.transpose(means[:, :d])

    xbar = np.average(means, axis = 1)
    xi_xbar = X - xbar
    xi_xbar_t = np.transpose(xi_xbar)
    covmat = np.dot(xi_xbar_t, xi_xbar)
    covmat = covmat / n

    return means, covmat


def qdaLearn(X, y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # IMPLEMENT THIS METHOD
    u = np.unique(y)
    k = u.shape[0]
    n = X.shape[0]
    d = X.shape[1]

    Xy = np.concatenate((X, y), axis=1)
    means = np.zeros([d, k],float)
    covmats = [np.zeros((d, d))] * k # construct a list covmats, contains k arrays, each d * d

    for i in range(k):
        Xy_i = Xy[Xy[:, d] == u[i], :]
        Xy_i = Xy_i[:, :d]
        means[:, i] = np.mean(Xy_i,axis=0)
        xi_xbar = Xy_i - np.transpose(means[: , i])
        xi_xbar_t = np.transpose(xi_xbar)
        covmat = np.dot(xi_xbar_t, xi_xbar)
        covmats[i] = covmat / Xy_i.shape[0]

    return means, covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    det_cov = np.linalg.det(covmat)
    cov_inv = np.linalg.inv(covmat)
    k = means.shape[1]  #k is the number of classes
    d = Xtest.shape[1]
    dominator = ((2 * np.pi) ** (d / 2)) * (np.sqrt(det_cov))

    y_prediction = np.zeros((Xtest.shape[0], k)) # create a N * k matrix, for voting

    for i in range(k):
        x_means = (Xtest - means[:, i]) # caculate xtest minus means, x_means will be a n * 2 matrix
        x_means_t = np.transpose(x_means)
        dot = np.dot(x_means, np.transpose(cov_inv) ) # 100*2 2*2
        powerofexp = np.sum(x_means * dot, axis = 1) # Caculate the power of exp for 100 vectors
        exp_power = np.exp(-(1/2) * powerofexp)
        y_prediction[:, i] = exp_power / dominator

    ypred = np.argmax(y_prediction, axis = 1) + 1 # python is from 0, so add 1
    ytest = ytest.flatten()
    length = len(ytest)

    acc_count = 0 # set the initial value of acc_count
    for i in range(length):
        if ytest[i] == ypred[i]:
            acc_count = acc_count + 1
    acc = 100 * (acc_count / length)

    return acc, ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    k = means.shape[1]
    d = Xtest.shape[1]
    y_prediction = np.zeros((Xtest.shape[0], k))

    for i in range(k):
        covmat = covmats[i]
        det_cov = np.linalg.det(covmat)
        cov_inv = np.linalg.inv(covmat)
        dominator = ((2 * np.pi) ** (d / 2)) * (np.sqrt(det_cov))
        x_means = (Xtest - means[:, i])
        dot = np.dot(x_means, np.transpose(cov_inv))  # 100*2 2*2
        powerofexp = np.sum(x_means * dot, axis=1)  # Caculate the power of exp for 100 vectors
        exp_power = np.exp(-(1 / 2) * powerofexp)
        y_prediction[:, i] = exp_power / dominator

    ypred = np.argmax(y_prediction, axis=1) + 1  # python is from 0, so add 1
    ytest = ytest.flatten()
    length = len(ytest)

    acc_count = 0  # set the initial value of acc_count
    for i in range(length):
        if ytest[i] == ypred[i]:
            acc_count = acc_count + 1
    acc = 100 * (acc_count / length)

    return acc, ypred


def learnOLERegression(X, y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1

    # IMPLEMENT THIS METHOD
    w1 = np.linalg.inv(np.dot(np.transpose(X), X))
    w2 = np.dot(np.transpose(X), y)
    w = np.dot(w1, w2)
    return w

def testOLERegression(w, t_input, t_target):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse

    # IMPLEMENT THIS METHOD
    n = t_input.shape[0]
    dev = t_target - np.dot(t_input, w)
    MLE = np.dot(np.transpose(dev), dev) / n
    mse = MLE[0,0]
    return mse


def learnRidgeRegression(X, y, lambd):
    # Inputs:
    # X = N x d
    # y = N x 1
    # lambd = ridge parameter (scalar)
    # Output:
    # w = d x 1

    # IMPLEMENT THIS METHOD
    colnum = len(X.T)
    w1 = np.linalg.inv(np.dot(np.transpose(X), X) + np.dot(lambd, np.identity(colnum)))
    w2 = np.dot(w1, np.transpose(X))
    w = np.dot(w2, y)

    return w


def regressionObjVal(w, X, y, lambd):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    # w - a N x 1 column vector

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda

    # IMPLEMENT THIS METHOD

    # print("=====start=====")

    # The Learning Rate alpha.

    # Gradient of the function J definition.
    size = np.shape(X)
    sizeD = size[1]
    w = w.reshape(sizeD, 1)

    diff = np.dot(X, w) - y
    error = 1 / 2 * np.dot(np.transpose(diff), diff) + 1 / 2 * lambd * np.dot(np.transpose(w), w)
    error_grad = np.dot(np.transpose(X), diff) + lambd * w

    error = error.flatten()
    error_grad = error_grad.flatten()

    return error, error_grad

# problem 5
def mapNonLinear(x, p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xp - (N x (p+1))

    # IMPLEMENT THIS METHOD

    Xp = np.zeros((x.shape[0], p + 1))
    x = x. reshape(len(x), 1)
    for i in range(p + 1):
        Xp[:, i] = x[:, 0] ** i

    return Xp


# Main script

# Problem 1
# load the sample data
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# import training data and test data
# data in diabetes.pickle and sample.pickle are the same
#spath = '/Users/xianzhou/Desktop/Assignment1/basecode/sample.pickle'
#X,y,Xtest,ytest = pickle.load(open(spath,'rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
ytest_f = ytest.flatten()
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest_f)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest_f)
plt.title('QDA')

plt.show()

# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')
# import training data and test data
# path = '/Users/xianzhou/Desktop/Assignment1/basecode/diabetes.pickle'
#X,y,Xtest,ytest = pickle.load(open(path,'rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    #print(mses3_train[i])
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()

# Problem 5
pmax = 7
lambda_opt = 0.06 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
