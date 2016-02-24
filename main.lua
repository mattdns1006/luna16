require "nn"
require "cunn"
require "cutorch"
require "optim"
require "torch"
require "xlua"
dofile("imageCandidates.lua")
dofile("3dInterpolation3.lua")
models = require "models"
shuffle = require "shuffle"

----------------------------------------------------------------------------------------------------------------------
------------------------------------   GLOBAL VARS & PARAMS   	   ----------------------------------------------------
----------------------------------------------------------------------------------------------------
-- GLobal vars
trainProportion = 0.8
angleMax = 0.3
sliceSize = 64 
clipMin = -1014 -- clip sizes determined from ipython nb
clipMax = 500
nClasses = 2
batchSize = 1

-- Optimizer
optimState = {
	learningRate = 0.0001,
	beta1 = 0.9,
	beta2 = 0.999,
	epsilon = 1e-8
}
optimMethod = optim.adam

-- Model
model = models.model1()
print("Model == >",model)
--criterion = nn.MSECriterion()
criterion = nn.BCECriterion()

-- Add confusion matrix
classes = {0,1}
cm = optim.ConfusionMatrix(classes)

-- Load data
--candidateCsv = csvToTable("CSVFILES/candidatesCleaned.csv")
train = csvToTable("CSVFILES/candidatesTrainBalanced8.csv")
test = csvToTable("CSVFILES/candidatesTestBalanced8.csv")


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
		x, xSub  = rotation3d(obs, angleMax, sliceSize, clipMin, clipMax):reshape(1,sliceSize,sliceSize,sliceSize)
		y = obs.Class
		xBatchTensor[k] = x 
		yBatchTensor[k] = y 
	end

	return xBatchTensor, yBatchTensor, batch
	
end
	

function training()

	function CUDA()
		model = model:cuda()
		criterion = criterion:cuda()
		print("==> GPU")
	end
	CUDA()

	if model then parameters,gradParameters = model:getParameters() end

	epoch = 1
	batchLosses = {}
	
	for i = 1, #train, batchSize do 

		xlua.progress(i,#train)

		inputs, targets, batch  = getBatch(train,i,batchSize)
		inputs = inputs:cuda()
		targets = targets:cuda()
		
		function feval(x)
			if x~= parameters then
				parameters:copy(x)
			end

			loss = 0

			predictions = model:forward(inputs)
			loss = criterion:forward(predictions,targets)

			dLoss_d0 = criterion:backward(predictions,targets)
			model:backward(inputs, dLoss_d0)

			--cm:add(predictions,targets:reshape(batchSize))
			--batchAcc = torch.cmul(predictions,targets):sum()/targets:size()[1]
			return loss, gradParameters

		end

		_, batchLoss = optimMethod(feval,parameters,optimState)
		batchLosses[#batchLosses + 1] = batchLoss[1]
		batchLossesTensor = torch.Tensor(batchLosses)
		print("Overall mean ==> " .. batchLossesTensor:mean())

		-- Moving average
		ma = 5 
		if batchLossesTensor:size()[1] > ma then 
			print("Moving average of last "..ma.. " batches mean == > " .. batchLossesTensor[{{-ma,-1}}]:mean() )
		end


	end
end







