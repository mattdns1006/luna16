require "nn"
require "cunn"
require "cutorch"
require "optim"
require "torch"
require "xlua"
require "gnuplot"
dofile("imageCandidates.lua")
dofile("3dInterpolation3.lua")
models = require "models"
shuffle = require "shuffle"

------------------------------------------ GLobal vars/params ---------------------------------------- 

cmd = torch.CmdLine()
cmd:text()
cmd:text("Main file for training")
cmd:text('Options')
cmd:option('-lr',0.001,'Learning rate')
cmd:option('-momentum',0.95,'Momentum')
cmd:option('-batchSize',1,'batchSize')
cmd:option('-cuda',1,'CUDA')
cmd:option('-angleMax',0.2,"Absolute maximum angle for rotating image")
cmd:option('-clipMin',-1014,'Clip image below this value to this value')
cmd:option('-clipMax',500,'Clip image above this value to this value')
cmd:option('-angleMax',0.2,"Absolute maximum angle for rotating image")
params = cmd:parse(arg)
params.rundir = cmd:string('experiment', params, {dir=true})

angleMax = params.angleMax
params.sliceSize = 96 

-- Optimizer
optimState = {
	learningRate = params.lr,
	beta1 = 0.9,
	beta2 = 0.999,
	epsilon = 1e-8
}
optimMethod = optim.sgd
--[[
optimState = {
	learningRate = params.lr,
	beta1 = 0.9,
	beta2 = 0.999,
	epsilon = 1e-8
}
optimMethod = optim.adam
]]--

-- Model
model = models.model1()
--model = torch.load("models/model1.model")
print("Model == >",model)
--criterion = nn.MSECriterion()
criterion = nn.BCECriterion()

if params.cuda == 1 then
	model = model:cuda()
	criterion = criterion:cuda()
	print("==> Placed on GPU")
end

-- Add confusion matrix -- TO DO
classes = {"0","1"}
cm = optim.ConfusionMatrix(classes)

-- Load data
--candidateCsv = csvToTable("CSVFILES/candidatesCleaned.csv")
train = csvToTable("CSVFILES/candidatesTrainBalanced8.csv")
test = csvToTable("CSVFILES/candidatesTestBalanced8.csv")

-- Example
obs = Candidate:new(train,1)
x = rotation3d(obs, angleMax, params.sliceSize, params.clipMin, params.clipMax,1):reshape(1,params.sliceSize,params.sliceSize,params.sliceSize)


-- Function to get batches
function getBatch(data,from,batchSize)
	if from + params.batchSize - 1  >= #data then to = #data else to = from + params.batchSize - 1 end -- check to see if at end of data
	
	local batch = shuffle.getBatch(data,from,to)
	--print(batch)

	-- For each image file in a batch. 1/ Load it 2/ Rotate it /3 Place it in batch tensor
	local xBatchTensor = torch.Tensor(params.batchSize,1,params.sliceSize,params.sliceSize,params.sliceSize)
	local yBatchTensor = torch.Tensor(params.batchSize,1)
	if params.cuda == 1 then 
		xBatchTensor, yBatchTensor = xBatchTensor:cuda(), yBatchTensor:cuda()
	end

	for k,v in ipairs(batch) do 
		obs = Candidate:new(batch,k)
		x = rotation3d(obs, angleMax, params.sliceSize, params.clipMin, params.clipMax,1):reshape(1,params.sliceSize,params.sliceSize,params.sliceSize)
		y = obs.Class
		xBatchTensor[k] = x 
		yBatchTensor[k] = y 
	end

	return xBatchTensor, yBatchTensor, batch
end
	

function training(display)

	if displayTrue==nil and display==1 then
		print("Initializing displays ==>")
		zoom = 0.6
		init = image.lena()
		imgZ = image.display{image=init, zoom=zoom, offscreen=false}
		imgY = image.display{image=init, zoom=zoom, offscreen=false}
		imgX = image.display{image=init, zoom=zoom, offscreen=false}
		--[[
		imgOrigZ = image.display{image=init, zoom=zoom, offscreen=false}
		imgOrigY = image.display{image=init, zoom=zoom, offscreen=false}
		imgOrigX = image.display{image=init, zoom=zoom, offscreen=false}
		]]--
		displayTrue = "not nil"
	end


	if model then parameters,gradParameters = model:getParameters() end

	while true do

		epoch = 1
		epochLosses = {}
		batchLosses = {}
		
		for i = 1, #train, params.batchSize do 

			xlua.progress(i,#train)

			inputs, targets, batch  = getBatch(train,i,params.batchSize)
			if params.cuda == 1 then
				inputs = inputs:cuda()
				targets = targets:cuda()
			end
				
			function feval(x)
				if x~= parameters then
					parameters:copy(x)
				end

				gradParameters:zero()
				loss = 0
				predictions = model:forward(inputs)
				loss = criterion:forward(predictions,targets)

				dLoss_d0 = criterion:backward(predictions,targets)
				model:backward(inputs, dLoss_d0)

				return loss, gradParameters

			end

			_, batchLoss = optimMethod(feval,parameters,optimState)
			batchLosses[#batchLosses + 1] = batchLoss[1]
			batchLossesT = torch.Tensor(batchLosses)
			local t = torch.range(1,batchLossesT:size()[1])
			gnuplot.plot({"Train loss",t,batchLossesT})
			

			-- Moving average
			ma = 5 
			if batchLossesT:size()[1] > ma then print("Moving average of last "..ma.. " batches ==> " .. batchLossesT[{{-ma,-1}}]:mean()) end

			if display == 1 and displayTrue ~= nil then 
				local idx = params.batchSize
				local class = batch[idx]:split(",")[2]

				-- Display rotated images
				image.display{image = inputs[{{idx},{},{params.sliceSize/2 +1}}]:reshape(params.sliceSize,params.sliceSize), win = imgZ, legend = class}
				image.display{image = inputs[{{idx},{},{},{params.sliceSize/2 +1}}]:reshape(params.sliceSize,params.sliceSize), win = imgY, legend = class}
				image.display{image = inputs[{{idx},{},{},{},{params.sliceSize/2 +1}}]:reshape(params.sliceSize,params.sliceSize), win = imgX, legend = class}

				--[[
				imgSub  = imgOriginal:sub(obs.z-sliceSize/2 +1 , obs.z+sliceSize/2 , obs.y-sliceSize/2+1, obs.y+sliceSize/2, obs.x - sliceSize/2 + 1, obs.x +sliceSize/2)

				-- Display original slices
				image.display{image = imgSub[sliceSize/2]:reshape(sliceSize,sliceSize), win = imgOrigZ, legend = class}
				image.display{image = imgSub[{{},{sliceSize/2}}]:reshape(sliceSize,sliceSize), win = imgOrigY, legend = class}
				image.display{image = imgSub[{{},{},{sliceSize/2}}]:reshape(sliceSize,sliceSize), win = imgOrigX, legend = class}
				]]--
			end


		end

		epoch = epoch + 1
		print("On epoch # .. " .. epoch)
	end
end







