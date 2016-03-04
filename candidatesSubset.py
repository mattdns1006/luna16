import numpy as np
import pandas as pd
import pdb

# Split both dataframes into train/test 
def splitDataFrame(data,splitPercentage):
    nObs = data.shape[0]
    idx = np.arange(nObs) 
    trainIdx, testIdx = idx[:np.floor(nObs*splitPercentage)], idx[np.floor(nObs*splitPercentage):]
    return data.iloc[trainIdx], data.iloc[testIdx]

if __name__ == "__main__":

    # Import, shuffle dataset and split into the two classes
    data = pd.read_csv("/home/msmith/luna16/CSVFILES/candidatesCleaned.csv")
    randomPerm = np.random.permutation(data.shape[0])
    data = data.iloc[randomPerm] #Shuffle
    class0, class1 = [data.loc[(data.Class==i)] for i in 0,1]
    print "Total data set is of size %d. Distribution of class; %d type 0, %d type 1." % (data.shape[0], class0.shape[0], class1.shape[0]) 
    splitPercentage = 0.8
    train0, test0 = splitDataFrame(class0,splitPercentage)
    train1, test1 = splitDataFrame(class1,splitPercentage)
    train0.to_csv("/home/msmith/luna16/CSVFILES/candidatesClass0Train.csv",index=0)
    train1.to_csv("/home/msmith/luna16/CSVFILES/candidatesClass1Train.csv",index=0)
    test0.to_csv("/home/msmith/luna16/CSVFILES/candidatesClass0Test.csv",index=0)
    test1.to_csv("/home/msmith/luna16/CSVFILES/candidatesClass1Test.csv",index=0)
    print "Number of training examples of class 0,1; %d, %d." % (train0.shape[0],train1.shape[0])
    print "Number of testing examples of class 0,1; %d, %d." % (test0.shape[0],test1.shape[0])


