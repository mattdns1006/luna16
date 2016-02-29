require "nn"
require "cunn"
require "cutorch"
dofile("imageCandidates.lua")
dofile("3dInterpolation3.lua")
models = require "models"
shuffle = require "shuffle"
threads = require 'threads'

nthread = 4
njob = 10

trainProportion = 0.8
angleMax = 0.7
sliceSize = 48 
clipMin = -1014 -- clip sizes determined from ipython nb
clipMax = 500
nClasses = 2

-- Load data
--candidateCsv = csvToTable("CSVFILES/candidatesCleaned.csv")
train = csvToTable("CSVFILES/candidatesTrainBalanced8.csv")
test = csvToTable("CSVFILES/candidatesTestBalanced8.csv")
batchSize = 1 

-- Function to get batches
function getBatch(data,from,batchSize)
	if from + batchSize - 1  >= #data then to = #data else to = from + batchSize - 1 end -- check to see if at end of data
	
	local batch = shuffle.getBatch(data,from,to)
	--print(batch)

	-- For each image file in a batch. 1/ Load it 2/ Rotate it /3 Place it in batch tensor
	local xBatchTensor = torch.Tensor(batchSize,1,sliceSize,sliceSize,sliceSize):cuda()
	local yBatchTensor = torch.Tensor(batchSize,1):cuda()
	for k,v in ipairs(batch) do 
		obs = Candidate:new(batch,k)
		-- Rotate depends on first epoch i.e if we are on the first epoch we do not rotate to speed things up
		--if epoch == 1 then rotate = 0 else rotate = 1 end
		x = rotation3d(obs, angleMax, sliceSize, clipMin, clipMax,1):reshape(1,sliceSize,sliceSize,sliceSize)
		y = obs.Class
		xBatchTensor[k] = x 
		yBatchTensor[k] = y 
	end

	return xBatchTensor, yBatchTensor, batch
	
end

pool = threads.Threads(
	nthread,
	function(threadid)
	print("Starting thread number " ..threadid)
end
)

jobdone = 0
for i=1,njob do 
	pool:addjob(
	getBatch(train,





