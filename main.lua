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
batchSize = 2 

-- Optimizer
optimState = {
	learningRate = 0.001,
	beta1 = 0.9,
	beta2 = 0.999,
	epsilon = 1e-8
}
optimMethod = optim.adam

-- Model
model = models.model1()
criterion = nn.MSECriterion()

-- CUDA


-- Load data
--candidateCsv = csvToTable("CSVFILES/candidatesCleaned.csv")
train = csvToTable("CSVFILES/candidatesTrain8.csv")
test = csvToTable("CSVFILES/candidatesTest8.csv")


-- Function to get batches
function getBatch(data,from,batchSize)
	from = 1
	if from + batchSize - 1  >= #data then to = #data else to = from + batchSize - 1 end -- check to see if at end of data
	
	local batch = shuffle.getBatch(data,from,to)
	print(batch)

	-- For each image file in a batch. 1/ Load it 2/ Rotate it /3 Place it in batch tensor
	local xBatchTensor = torch.Tensor(batchSize,1,sliceSize,sliceSize,sliceSize)
	local yBatchTensor = torch.Tensor(batchSize,1)
	for k,v in ipairs(batch) do 
		obs = Candidate:new(batch,k)
		x, xSub  = rotation3d(obs, angleMax, sliceSize, clipMin, clipMax):reshape(1,sliceSize,sliceSize,sliceSize)
		y = obs.Class
		xBatchTensor[k] = x 
		yBatchTensor[k] = y 
	end

	return xBatchTensor, yBatchTensor, batch
	
end
	
-- Example
--[[
xBatchTensor, yBatchTensor, batch  = getBatch(train,10,batchSize)
xBatchTensor = xBatchTensor:cuda()
yBatchTensor = yBatchTensor:cuda()
]]--

function training()

	function CUDA()
		model = model:cuda()
		criterion = criterion:cuda()
		print("==> GPU")
	end
	CUDA()

	if model then parameters,gradParameters = model:getParameters() end

	epoch = 1
	
	for i = 1, #train, batchSize do 

		xlua.progress(i,#train)

		xBatchTensor, yBatchTensor, batch  = getBatch(train,i,batchSize)
		xBatchTensor = xBatchTensor:cuda()
		yBatchTensor = yBatchTensor:cuda()
		
		function feval(x)
			if x~= parameters then
				parameters:copy(x)
			end

			loss = 0

			outputs = model:forward(xBatchTensor)
			loss = criterion:forward(outputs,yBatchTensor)

			dLoss_d0 = criterion:backward(outputs,yBatchTensor)
			model:backward(xBatchTensor, dLoss_d0)

			batchAcc = torch.cmul(outputs,yBatchTensor):sum()/yBatchTensor:size()[1]
			
			print(batchAcc)
			print(loss)
			print(outputs)
			return loss, gradParameters

		end

		_, batchLoss = optimMethod(feval,parameters,optimState)

	end
end







