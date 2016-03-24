import pandas as pd
import pdb

if __name__ == "__main__":
    path = "/home/msmith/luna16/CSVFILES/"
    for ext in ["subset40/","subset59/"]:
        print("-"*50)
        print("For folder: " + ext)
        print("-"*50)
        for ext2 in ["candidatesClass0Train.csv","candidatesClass1Train.csv","candidatesTest.csv"]:
            file = path + ext + ext2
            data = pd.read_csv(file)
            print(file + " includes subsets:")
            print(data.seriesuidFullPath.apply(lambda x: x.split("/")[4]).value_counts())

    
    pdb.set_trace()
