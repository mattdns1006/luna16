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

-- Model
model = models.model1()
--model = torch.load("models/model1.model")
print("Model == >",model)

criterion = nn.MSECriterion()
--criterion = nn.BCECriterion()

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Options')
cmd:option('-train',0,'Train straight away')
cmd:option('-lr',0.0000003,'Learning rate')
cmd:option('-momentum',0.95,'Momentum')
cmd:option('-batchSize',1,'batchSize')
cmd:option('-cuda',1,'CUDA')
cmd:option('-sliceSize',72,"Size of cube around nodule")
cmd:option('-angleMax',0.5,"Absolute maximum angle for rotating image")
cmd:option('-scalingFactor',0.86,'Scaling factor for image')
cmd:option('-clipMin',-1200,'Clip image below this value to this value')
cmd:option('-clipMax',600,'Clip image above this value to this value')
cmd:option('-useThreads',1,"Use threads or not") 
cmd:option('-display',0,"Display images/plots") 
cmd:option('-activations',0,"Show activations -- needs -display 1") 
cmd:text()

params = cmd:parse(arg)
params.model = model
params.rundir = cmd:string('results', params, {dir=true})
local logPath = "results/"..params.rundir
paths.mkdir(logPath)
logger = optim.Logger(logPath.. '/results.log') 
print("==> Parameters",params)

--Show activations need first n layers
if params.activations == 1 then
	modelActivations1 = nn.Sequential()
	for i=1,3 do modelActivations1:add(model:get(i)) end

	--print("==>Activations check " , modelActivations)
	modelActivations2 = nn.Sequential()
	for i=1,10 do modelActivations2:add(model:get(i)) end
	if params.cuda == 1 then modelActivations1:cuda(); modelActivations2:cuda(); end
end

-- Optimizer
optimState = {
	learningRate = params.lr,
	beta1 = 0.9,
	beta2 = 0.999,
	epsilon = 1e-8
}
optimMethod = optim.adam

if params.cuda == 1 then
	model = model:cuda()
	criterion = criterion:cuda()
	print("==> Placed on GPU")
end

-- Add confusion matrix -- TO DO

-- Load data
--candidateCsv = csvToTable("CSVFILES/candidatesCleaned.csv")
train = csvToTable("CSVFILES/candidatesTrainBalanced8.csv")
test = csvToTable("CSVFILES/candidatesTestBalanced8.csv")

trainingBatchSize= params.batchSize
queueLength= 12 
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
				sys.sleep(0.001)
			end
		end	
		local ourX = torch.LongTensor(torch.LongStorage(trainingBatchSize*s*s*s,g_MasterTensor[3*index-1])):resize(trainingBatchSize,1,s,s,s)
		local ourY = torch.Tensor(torch.Storage(trainingBatchSize,g_MasterTensor[3*index-2])):resize(trainingBatchSize,1)
		getBatch(data,trainingBatchSize,ourX,ourY,s,%d,%d,%f,%f)
		g_mutex:lock()
		g_MasterTensor[index*3]=3
		g_mutex:unlock()
	end
]],g_mutex:id(),queueLength,tonumber(torch.data(g_MasterTensor,1)),trainingBatchSize,params.sliceSize,params.clipMin,params.clipMax,params.angleMax,params.scalingFactor)
if params.useThreads then 
	print("==> Multithreading inputs")
	threads.Thread(task)
	threads.Thread(task)
	threads.Thread(task)
	threads.Thread(task)
	threads.Thread(task)
	threads.Thread(task)
	threads.Thread(task)
	threads.Thread(task)
	threads.Thread(task)
	threads.Thread(task)
	threads.Thread(task)
	threads.Thread(task)
end

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
			sys.sleep(0.001)
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
		if params.activations == 1 then
			activationDisplay1 = image.display{image=init, zoom=zoom, offscreen=false}
			--activationDisplay2 = image.display{image=init, zoom=zoom, offscreen=false}
		end
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

				getBatch(train,params.batchSize,xBatchTensor,yBatchTensor,params.sliceSize,params.clipMin,params.clipMax,params.angleMax,params.scalingFactor)
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
				print(loss)
				logger:add{['loss'] = loss }


				model:backward(inputs, dLoss_d0)

				return loss, gradParameters

			end
			-- Possibly improve this to take batch with large error more frequently
			_, batchLoss = optimMethod(feval,parameters,optimState)
			batchLosses[#batchLosses + 1] = batchLoss[1]/params.batchSize
			
			local batchLossesT = torch.Tensor(batchLosses)
			local t = torch.range(1,batchLossesT:size()[1])
			if i % 5 == 0 then
				gnuplot.figure(1)
				ma = 20 
				if batchLossesT:size()[1] > ma then print("Moving average of last "..ma.. " batches ==> " .. batchLossesT[{{-ma,-1}}]:mean()) end

				gnuplot.plot({"Train loss",t,batchLossesT})
			end


			if params.display == 1 and displayTrue ~= nil and i % 3 == 0 then 
				local idx = 1 
				local class = "Class = " .. targets[1][1] .. ". Prediction = ".. predictions[1]

				-- Display rotated images
				image.display{image = inputs[{{idx},{},{params.sliceSize/2 +1}}]:reshape(params.sliceSize,params.sliceSize), win = imgZ, legend = class}
				image.display{image = inputs[{{idx},{},{},{params.sliceSize/2 +1}}]:reshape(params.sliceSize,params.sliceSize), win = imgY, legend = class}
				image.display{image = inputs[{{idx},{},{},{},{params.sliceSize/2 +1}}]:reshape(params.sliceSize,params.sliceSize), win = imgX, legend = class}

				-- Display first layer activtion plane. Draw one activation plane at random and slice on first (z) dimension.
				if params.activations == 1 then 
					local activations1 = modelActivations1:forward(inputs)
					local randomFeat1 = torch.random(1,modelActivations1:get(2).nOutputPlane)
					image.display{image = activations1[{{1},{randomFeat1},{params.sliceSize/2}}]:reshape(params.sliceSize,params.sliceSize), win = activationDisplay1, legend = "Activations"}
				end

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






