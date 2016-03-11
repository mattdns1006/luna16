import numpy as np
import pandas as pd
import pdb

if __name__ == "__main__":
    for idx in xrange(10):
        print("Splitting data into train test using subset %d as test set" %idx)
        data = pd.read_csv("/home/msmith/luna16/CSVFILES/candidatesCleaned.csv")
        test = data[data.seriesuidFullPath.str.contains("subset"+str(idx))]
        train = data[data.seriesuidFullPath.str.contains("subset[^"+str(idx)+"]")]
        assert train.shape[0] + test.shape[0] == data.shape[0], "test + training sizes do not add up"
        commonCols = train.columns.tolist()                         
        check = pd.merge(train, test, on=commonCols, how='inner') 
        assert check.shape[0]==0, "There are common values between train and test"

        # Split training set into the two classes 
        train0, train1 = [train.loc[(train.Class==i)] for i in 0,1]
        test0, test1 = [test.loc[(test.Class==i)] for i in 0,1]
        train0.to_csv("/home/msmith/luna16/CSVFILES/subset"+str(idx)+"/candidatesClass0Train.csv",index=0)
        train1.to_csv("/home/msmith/luna16/CSVFILES/subset"+str(idx)+"/candidatesClass1Train.csv",index=0)
        test0.to_csv("/home/msmith/luna16/CSVFILES/subset"+str(idx)+"/candidatesClass0Test.csv",index=0)
        test1.to_csv("/home/msmith/luna16/CSVFILES/subset"+str(idx)+"/candidatesClass1Test.csv",index=0)
