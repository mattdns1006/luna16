require "nn"
require "cunn"
require "cutorch"
require "optim"
require "torch"
require "xlua"
dofile("imageCandidates.lua")
dofile("3dInterpolation3.lua")
shuffle = require "shuffle"

---------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------   GLOBAL VARS & PARAMS   	   ----------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------
-- GLobal vars
nObs = table.getn(candidateCsv)
trainProportion = 0.8
angleMax = 0.3
sliceSize = 64 
clipMin = -1014 -- clip sizes determined from ipython nb
clipMax = 500
nClasses = 2

batchSize = 3 

-- Shuffle the data and split into train and test
candidateCsv = shuffle.shuffle(candidateCsv)
train, test = shuffle.split(candidateCsv,trainProportion)
print("Number of train, test using a percentage of "..trainProportion .. " ==> " ..#train.. ", "..#test.. ".")

epoch = 1

--- For each epoch iterate through batchs
from = 1
to = from + batchSize - 1
	-- For each batch in an epoch
	batch = shuffle.getBatch(train,from,to)

	-- For each image file in a batch. 1/ Load it 2/ Rotate it /3 Place it in batch tensor
	xBatchTensor = torch.Tensor(batchSize,1,sliceSize,sliceSize,sliceSize)
	yBatchTensor = torch.Tensor(batchSize,1)
	for k,v in ipairs(batch) do 
		obs = Candidate:new(batch,k)
	 	x = rotation3d(obs, angleMax, sliceSize, clipMin, clipMax):reshape(1,sliceSize,sliceSize,sliceSize)
		y = obs.Class
		xBatchTensor[k] = x 
		yBatchTensor[k] = y 
	end

	--model
	--
	
	







