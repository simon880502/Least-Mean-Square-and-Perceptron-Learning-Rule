# -*- coding: utf-8 -*-
"""
Created on Thu May  7 08:52:39 2020

@author: fju
"""

import numpy as np
import matplotlib.pyplot as plt
#Load Data
def Main(data):
    X = np.loadtxt(data+".txt",delimiter=',')
    y=np.array(X[::,-1::],dtype=int)
#    np.random.shuffle(y)
    X=X[::,0:-1:]
    One=np.full((X.shape[0],1),1)
    X=np.hstack((One,X))
    del One 
    w=np.random.random((1,X.shape[1]))*0.1
    
    #Def plot function
    def plot(X,y,w):
        for i in range(len(X)):
            if y[i]==1:
                plt.scatter(X[i,1],X[i,2], color='red')
            else:
                plt.scatter(X[i,1],X[i,2], color='blue')
        x = np.linspace(min(X[::,1])-2,max(X[::,1])+2, 1000)
        plt.plot(x,(w[0,0]-w[0,1]*x)/w[0,2],linewidth=3)
        plt.title(data)
        print('PLOT:')
        plt.show()
    
    #Def symmetric hardlimit fumction
    def s_hardlim(y):
        if y<=0:
            return -1
        else:
            return 1
    
    def plt_check(X):
        E=0
        for i in range(len(X)):
            y_bar=s_hardlim(np.inner(w,X[i]))
            if y[i][0]!=y_bar:
                E+=1
        return E
    #    print(E)
    
    #---Perceptron--- 
    
        #VARIABLE SET UP
    
    plt_LR=10
    epoch=0
    epochBound=20
    print('Training {} start'.format(data))
    while(True):

        epoch+=1
        for i in range(X.shape[0]):
            y_bar=s_hardlim(np.inner(w,X[i]))
    #        y_bar=np.inner(w,X[i])
            w=w+((plt_LR*(y[i]-y_bar))*X[i])
        
        if plt_check(X)==0:
            print('\nSTOP REASON：',end='')
            print("All data is classified correctly")
            break
        if epoch>=epochBound:
#            print('B')
            print('\nSTOP REASON：',end='')
            print('epoch is greater the epochbound\nHas {0:d} error(s)'\
                  .format(plt_check(X)))
            break
    print('Training End after {} epoch(s)\n\nw = {}'.format(epoch,w[0]))
    plot(X,y,w)
    print('='*50)
datalist=['dataset1','dataset2','dataset3','dataset4']
for i in datalist:
    print('\n',i,'\n')
    Main(i)
#for i in range(20):
#
#    Main("dataset4")