require "nn"
require "cunn"
require "cutorch"
require "optim"
require "torch"
require "xlua"
require "gnuplot"
threads = require "threads"
dofile("imageCandidates.lua")
dofile("3dInterpolation3.lua")
dofile("getBatch.lua")
models = require "models"
shuffle = require "shuffle"

------------------------------------------ GLobal vars/params ---------------------------------------- 

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Options')
cmd:option('-train',0,'Train straight away')
cmd:option('-lr',0.000000001,'Learning rate')
cmd:option('-momentum',0.95,'Momentum')
cmd:option('-batchSize',4,'batchSize')
cmd:option('-cuda',1,'CUDA')
cmd:option('-sliceSize',64,"Size of cube around nodule")
cmd:option('-angleMax',0.2,"Absolute maximum angle for rotating image")
cmd:option('-clipMin',-1000,'Clip image below this value to this value')
cmd:option('-clipMax',500,'Clip image above this value to this value')
cmd:option('-angleMax',0.2,"Absolute maximum angle for rotating image")
cmd:option('-display',0,"Display images/plots") 
cmd:option('-useThreads',1,"Use threads or not") 
cmd:text()
params = cmd:parse(arg)
params.rundir = cmd:string('experiment', params, {dir=true})
paths.mkdir(params.rundir)


angleMax = params.angleMax

-- Optimizer
--[[
optimState = {
	learningRate = params.lr,
	beta1 = 0.9,
	beta2 = 0.999,
	epsilon = 1e-8
}
optimMethod = optim.sgd
]]--
optimState = {
	learningRate = params.lr,
	beta1 = 0.9,
	beta2 = 0.999,
	epsilon = 1e-8
}
optimMethod = optim.adam


-- Model
model = models.model1()
--model = torch.load("models/model1.model")
print("Model == >",model)
print("==> Parameters",params)
criterion = nn.MSECriterion()
--criterion = nn.BCECriterion()

if params.cuda == 1 then
	model = model:cuda()
	criterion = criterion:cuda()
	print("==> Placed on GPU")
end

-- Add confusion matrix -- TO DO
classes = {"0","1"}
cmTrain = optim.ConfusionMatrix(classes)

-- Load data
--candidateCsv = csvToTable("CSVFILES/candidatesCleaned.csv")
train = csvToTable("CSVFILES/candidatesTrainBalanced8.csv")
test = csvToTable("CSVFILES/candidatesTestBalanced8.csv")

trainingBatchSize= params.batchSize
queueLength= 25 
g_mutex=threads.Mutex()
g_tensorsForQueue={}
g_MasterTensor = torch.LongTensor(3*queueLength) --first 2 begin and end of queue
for i = 1,queueLength do
	
	g_tensorsForQueue[2*i]=torch.LongTensor(trainingBatchSize,1,params.sliceSize,params.sliceSize,params.sliceSize)
	g_tensorsForQueue[2*i-1]=torch.Tensor(trainingBatchSize,1)
	g_MasterTensor[3*i-1]=tonumber(torch.data(g_tensorsForQueue[2*i],1))
	g_MasterTensor[3*i-2]=tonumber(torch.data(g_tensorsForQueue[2*i-1],1))
	g_MasterTensor[3*i]=1
end
task = string.format([[
	threads = require 'threads'
	require 'sys'
	dofile("imageCandidates.lua")
	dofile("3dInterpolation3.lua")
	dofile("getBatch.lua")
	data = csvToTable("CSVFILES/candidatesTrainBalanced8.csv")
	--test = csvToTable("CSVFILES/candidatesTestBalanced8.csv")
	local g_mutex = threads.Mutex(%d)
	local queueLength = %d
	local g_MasterTensor = torch.LongTensor(torch.LongStorage(queueLength*3,%d))
	local trainingBatchSize = %d
	local s = %d
	while 1 do
		local ok = false
		local index = -1
		while not ok do
			g_mutex:lock()
			for i=1,queueLength do

				if g_MasterTensor[3*i]==1 then
					ok=true
					index=i
					g_MasterTensor[3*i] = 2
					break
				end
			end
			g_mutex:unlock()

			if not ok then	
				--print("full")
				sys.sleep(0.1)
			end
		end	
		local ourX = torch.LongTensor(torch.LongStorage(trainingBatchSize*s*s*s,g_MasterTensor[3*index-1])):resize(trainingBatchSize,1,s,s,s)
		local ourY = torch.Tensor(torch.Storage(trainingBatchSize,g_MasterTensor[3*index-2])):resize(trainingBatchSize,1)
		getBatch(data,trainingBatchSize,ourX,ourY,s,%d,%d,%d)
		g_mutex:lock()
		g_MasterTensor[index*3]=3
		g_mutex:unlock()
	end
]],g_mutex:id(),queueLength,tonumber(torch.data(g_MasterTensor,1)),trainingBatchSize,params.sliceSize,params.clipMin,params.clipMax,params.angleMax)
if params.useThreads then 
	print("==> Multithreading inputs")
	threads.Thread(task)
	threads.Thread(task)
	threads.Thread(task)
	threads.Thread(task)
	threads.Thread(task)
	threads.Thread(task)
end

-- Batch example
--x,y,batch = getBatch(train,5)

function retrieveBatch()
	local ok = false
	local index = -1
	while not ok do
		g_mutex:lock()
		for i=1,queueLength do

			if g_MasterTensor[3*i]== 3 then
				ok=true
				index=i
				g_MasterTensor[3*i] = 4 
				break
			end
		end
		g_mutex:unlock()

		if not ok then	
			sys.sleep(0.5)
		end
	end	
	local x = g_tensorsForQueue[2*index]
	local y = g_tensorsForQueue[2*index-1]
	g_mutex:lock()
	
	g_MasterTensor[index*3]=1
	g_mutex:unlock()
	return x,y
end

function training()

	if displayTrue==nil and params.display==1 then
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
			if not params.useThreads then 
				local xBatchTensor = torch.Tensor(params.batchSize,1,params.sliceSize,params.sliceSize,params.sliceSize)
				local yBatchTensor = torch.Tensor(params.batchSize,1)

				getBatch(train,params.batchSize,xBatchTensor,yBatchTensor,params.sliceSize,params.clipMin,params.clipMax,params.angleMax)
				inputs, targets = xBatchTensor, yBatchTensor
			else 
				inputs, targets = retrieveBatch()
			end 

			if params.cuda == 1 then
				inputs = inputs:cuda()
				targets = targets:cuda()
			end
				
			function feval(x)
				if x~= parameters then parameters:copy(x) end

				gradParameters:zero()

				predictions = model:forward(inputs)
				loss = criterion:forward(predictions,targets)

				dLoss_d0 = criterion:backward(predictions,targets)
				model:backward(inputs, dLoss_d0)

				return loss, gradParameters

			end
			-- Possibly improve this to take batch with large error more frequently
			--for i=1,2 do
			_, batchLoss = optimMethod(feval,parameters,optimState)
			batchLosses[#batchLosses + 1] = batchLoss[1]/params.batchSize
			--end
			local batchLossesT = torch.Tensor(batchLosses)
			local t = torch.range(1,batchLossesT:size()[1])

			--[[
			gnuplot.figure(1)
			--gnuplot.plot({"Train loss",t,batchLossesT})
			gnuplot.figure(2)
			--gnuplot.hist(inputs[1])
			--gnuplot.hist(parameters)
			]]--

			-- Moving average
			ma = 5 
			if batchLossesT:size()[1] > ma then print("Moving average of last "..ma.. " batches ==> " .. batchLossesT[{{-ma,-1}}]:mean()) end

			if params.display == 1 and displayTrue ~= nil then 
				local idx = 1 
				local class = "Class " .. targets[1][1]

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
		collectgarbage()
	end
end

if params.train == 1 then
	training()
end






