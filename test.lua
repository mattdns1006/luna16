require "nn"
require "cunn"
dofile("getBatch.lua")

model = torch.load("models/model1.model")
C0 = csvToTable("CSVFILES/candidatesClass0Test.csv",clipMin,clipMax,sliceSize)
C1 = csvToTable("CSVFILES/candidatesClass1Test.csv",clipMin,clipMax,sliceSize)
C0:getNewScan()
C1:getNewScan()






