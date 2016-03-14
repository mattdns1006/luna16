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
dofile("binaryAccuracy.lua")
models = require "models"
shuffle = require "shuffle"

------------------------------------------ GLobal vars/params -------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Options')
cmd:option('-lr',0.00002,'Learning rate')
cmd:option('-momentum',0.95,'Momentum')
cmd:option('-batchSize',8,'batchSize')
cmd:option('-cuda',1,'CUDA')
cmd:option('-sliceSize',36,"Length size of cube around nodule")
cmd:option('-angleMax',0.5,"Absolute maximum angle for rotating image")
cmd:option('-scalingFactor',0.75,'Scaling factor for image')
cmd:option('-clipMin',-1200,'Clip image below this value to this value')
cmd:option('-clipMax',1200,'Clip image above this value to this value')
cmd:option('-useThreads',0,"Use threads or not") 
cmd:option('-display',0,"Display images/plots") 
cmd:option('-activations',0,"Show activations -- needs -display 1") 
cmd:option('-log',0,"Make log file in /Results/") 
cmd:option('-train',0,'Train straight away')
cmd:option('-test',0,"Test") 
cmd:option('-loadModel',0,"Load model") 
cmd:option('-para',0,"Are we using a parallel network? If bigger than 0 then this is equal to number of inputs. Otherwise input number is 1.") 
cmd:option('-nInputScalingFactors',3,"Number of input scaling factors.") 
-- K fold cv options
cmd:option('-kFold',1,"Are we doing k fold? Default is to train on subsets 1-9 and test on subset0") 
cmd:option('-fold',0,"Which fold to NOT train on") 
--cmd:option('-loadModel',"model1.model","Load model") 
cmd:text()
params = cmd:parse(arg)
params.model = model
params.rundir = cmd:string('results', params, {dir=true})


-------------------------------------------- Model ---------------------------------------------------------
model = models.parallelNetwork()
--model = models.model5()
--[[
if params.kFold == 1 then -- Are we doing k fold CV
	--Save in models k fold
	modelName = string.format("models/modelsKFoldCV/model5_k%d.model",params.fold)
else 
	modelName = "models/model6.model"
end
if params.loadModel == 1 then
	print("==> Loading model "..modelName)
	model = torch.load(modelName)
end
]]--
print("Model == >",model)
print("==> Parameters",params)

-------------------------------------------- Criterion & Activations ----------------------------------------
criterion = nn.MSECriterion()
--criterion = nn.BCECriterion()


if params.log == 1 then  -- Log file
	local logPath = "results/"..params.rundir
	paths.mkdir(logPath)
	logger = optim.Logger(logPath.. '/results.log') 
end

--Show activations need first n layers
if params.activations == 1 then
	modelActivations1 = nn.Sequential()
	for i=1,3 do modelActivations1:add(model:get(i)) end
end

-------------------------------------------- Optimization --------------------------------------------------
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

-------------------------------------------- Threads -------------------------------------------------------

trainingBatchSize= params.batchSize
queueLength= 1 
g_mutex=threads.Mutex()
g_tensorsForQueue={}
g_tensorsForQueue2={}
g_MasterTensor = torch.LongTensor(3*queueLength) --first 2 begin and end of queue
for i = 1,queueLength do
--	g_tensorsForQueue[2*i]=torch.LongTensor(trainingBatchSize,1,params.sliceSize,params.sliceSize,params.sliceSize)
	g_tensorsForQueue[2*i]=torch.LongTensor(params.nInputScalingFactors)
	g_tensorsForQueue2[i]={}
	for j = 1, params.nInputScalingFactors do
	   local t = torch.LongTensor(trainingBatchSize,1,params.sliceSize,params.sliceSize,params.sliceSize)
	   g_tensorsForQueue2[i][j]=t
	   g_tensorsForQueue[2*i][j]=tonumber(torch.data(t,1))
	end
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
	
	local g_mutex = threads.Mutex(%d)
	local queueLength = %d
	local g_MasterTensor = torch.LongTensor(torch.LongStorage(queueLength*3,%d))
	local trainingBatchSize = %d
	s = %d -- SliceSize
	clipMin = %d	
	clipMax = %d	
	angleMax = %f	
	scalingFactor = %f
	test = %d
	kFold = %d
	fold = %d
	para = %d
        nInputScalingFactors = %d
	dofile("loadData.lua")
	
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
		print("writing ",index)
--		local ourX = torch.LongTensor(torch.LongStorage(trainingBatchSize*s*s*s,g_MasterTensor[3*index-1])):resize(trainingBatchSize,1,s,s,s)
                local ourX = {}
                local ourXAddresses = torch.LongTensor(torch.LongStorage(nInputScalingFactors,g_MasterTensor[3*index-1]))
                for iScaling = 1, nInputScalingFactors do
		    ourX[iScaling] = torch.LongTensor(torch.LongStorage(trainingBatchSize*s*s*s,ourXAddresses[iScaling])):resize(trainingBatchSize,1,s,s,s)
		end
		local ourY = torch.Tensor(torch.Storage(trainingBatchSize,g_MasterTensor[3*index-2])):resize(trainingBatchSize,1)

		-- With probability 0.5, 0.5 choose data from class 0 or class 1
		if torch.uniform() < 0.5 then
			getBatch(C0,trainingBatchSize,ourX,ourY,s,clipMin,clipMax,angleMax,scalingFactor,test,paraTable)
		else
			getBatch(C1,trainingBatchSize,ourX,ourY,s,clipMin,clipMax,angleMax,scalingFactor,test,paraTable)
		end
		g_mutex:lock()
		g_MasterTensor[index*3]=3
		g_mutex:unlock()
		print("written")
	end
]],g_mutex:id(),queueLength,tonumber(torch.data(g_MasterTensor,1)),trainingBatchSize,params.sliceSize,params.clipMin,params.clipMax,params.angleMax,params.scalingFactor,params.test,params.kFold, params.fold, params.para, params.nInputScalingFactors)

if params.useThreads then 
	print("==> Multithreading inputs")
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
			sys.sleep(0.1)
		end
	end	

--	local x = g_tensorsForQueue[2*index]
	local x = g_tensorsForQueue2[index]
	print("retrieving ",index)
	print(type(x))
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
		imgZ1 = image.display{image=init, zoom=zoom, offscreen=false}
		imgY1 = image.display{image=init, zoom=zoom, offscreen=false}
		imgX1 = image.display{image=init, zoom=zoom, offscreen=false}
		imgZ2 = image.display{image=init, zoom=zoom, offscreen=false}
		imgY2 = image.display{image=init, zoom=zoom, offscreen=false}
		imgX2 = image.display{image=init, zoom=zoom, offscreen=false}
		]]--
		if params.activations == 1 then
			activationDisplay1 = image.display{image=init, zoom=zoom, offscreen=false}
			--activationDisplay2 = image.display{image=init, zoom=zoom, offscreen=false}
		end
		displayTrue = "not nil"
	end


	if model then parameters,gradParameters = model:getParameters() end

	epoch = 1
	while true do


		epochLosses = {}
		batchLosses = {}
		batchLossesMA = {}
		accuraccies = {}
		n = 20000000
		
		for i = 1, n do 

			xlua.progress(i*params.batchSize,n)
			if not params.useThreads then 
				local xBatchTensor = torch.Tensor(params.batchSize,1,params.sliceSize,params.sliceSize,params.sliceSize)
				local yBatchTensor = torch.Tensor(params.batchSize,1)

				getBatch(train,params.batchSize,xBatchTensor,yBatchTensor,params.sliceSize,params.clipMin,params.clipMax,params.angleMax,params.scalingFactor,params.test,params.kFold, params.fold, params.para, params.nInputScalingFactors)


				inputs, targets = xBatchTensor, yBatchTensor
			else 
				inputs, targets = retrieveBatch()
			end 

			if params.cuda == 1 then
				for k,v in pairs(inputs) do
					inputs[k] = v:cuda()
				end
				targets = targets:cuda()
			end
				
			function feval(x)
				if x~= parameters then parameters:copy(x) end

				gradParameters:zero()

				--print("Forward")
				--print("Inputs",inputs,inputs[1][1][1][1][1][1],inputs[2][1][1][1][1][1],inputs[3][1][1][1][1][1])
				predictions = model:forward(inputs)
				loss = criterion:forward(predictions,targets)
				--print("Backward")
				dLoss_d0 = criterion:backward(predictions,targets)
				if params.log == 1 then logger:add{['loss'] = loss } end
				model:backward(inputs, dLoss_d0)

				return loss, gradParameters

			end
			-- Possibly improve this to take batch with large error more frequently
			_, batchLoss = optimMethod(feval,parameters,optimState)


			-- Performance metrics
			accuracy = binaryAccuracy(targets,predictions,params.cuda)
			loss = criterion:forward(predictions,targets)

			accuraccies[#accuraccies + 1] = accuracy
			batchLosses[#batchLosses + 1] = loss 
			accuracciesT = torch.Tensor(accuraccies)
			batchLossesT = torch.Tensor(batchLosses)
			local t = torch.range(1,batchLossesT:size()[1])
			local ma = 10
			if i > ma then 
				print(string.format("Iteration %d. MA loss of last 20 batches == > %f. MA accuracy ==> %f. Overall accuracy ==> %f ",
				i, batchLossesT[{{-ma,-1}}]:mean(), accuracciesT[{{-ma,-1}}]:mean(),accuracciesT:mean()))
			end

			--Plot
			if i % 30 == 0 then
				gnuplot.figure(1)
				gnuplot.plot({"Train loss",t,batchLossesT})
			end


			if i % 100 == 0 then
				print("==> Saving weights for ".. modelName)
				torch.save("models/"..modelName,model)
			end

			if params.display == 1 and displayTrue ~= nil and i % 5 == 0 then 
				local idx = 1 
				local class = "Class = " .. targets[1][1] .. ". Prediction = ".. predictions[1][1]

				-- Display rotated images
				-- Middle Slice
				image.display{image = inputs[{{idx},{},{params.sliceSize/2 +1}}]:reshape(params.sliceSize,params.sliceSize), win = imgZ, legend = class}
				image.display{image = inputs[{{idx},{},{},{params.sliceSize/2 +1}}]:reshape(params.sliceSize,params.sliceSize), win = imgY, legend = class}
				image.display{image = inputs[{{idx},{},{},{},{params.sliceSize/2 +1}}]:reshape(params.sliceSize,params.sliceSize), win = imgX, legend = class}
				-- Slice + 1
				--[[
				image.display{image = inputs[{{idx},{},{params.sliceSize/2 +2}}]:reshape(params.sliceSize,params.sliceSize), win = imgZ1, legend = class}
				image.display{image = inputs[{{idx},{},{},{params.sliceSize/2 +2}}]:reshape(params.sliceSize,params.sliceSize), win = imgY1, legend = class}
				image.display{image = inputs[{{idx},{},{},{},{params.sliceSize/2 +2}}]:reshape(params.sliceSize,params.sliceSize), win = imgX1, legend = class}
				-- Slice + 2 
				image.display{image = inputs[{{idx},{},{params.sliceSize/2 }}]:reshape(params.sliceSize,params.sliceSize), win = imgZ2, legend = class}
				image.display{image = inputs[{{idx},{},{},{params.sliceSize/2 }}]:reshape(params.sliceSize,params.sliceSize), win = imgY2, legend = class}
				image.display{image = inputs[{{idx},{},{},{},{params.sliceSize/2 }}]:reshape(params.sliceSize,params.sliceSize), win = imgX2, legend = class}
				]]--

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

function testing()
	local batchLosses = {}
	local accuraccies = {}
	local i = 1
	while true do
		i = i + 1
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

		predictions = model:forward(inputs)

		accuracy = binaryAccuracy(targets,predictions,params.cuda)
		loss = criterion:forward(predictions,targets)

		accuraccies[#accuraccies + 1] = accuracy
		batchLosses[#batchLosses + 1] = loss 
		accuracciesT = torch.Tensor(accuraccies)
		batchLossesT = torch.Tensor(batchLosses)
		local t = torch.range(1,batchLossesT:size()[1])
		local ma = 10
		if i > ma then 
			print(string.format("Iteration %d. MA loss of last 20 batches == > %f. MA accuracy ==> %f. Overall accuracy ==> %f ",
			i, batchLossesT[{{-ma,-1}}]:mean(), accuracciesT[{{-ma,-1}}]:mean(),accuracciesT:mean()))
		end

	end
end
 
if params.train == 1 then
	training()
elseif params.test == 1 then
	testing()
end





