require "nn"
require "cunn"
require "cutorch"
require "optim"
require "torch"
require "xlua"
require "gnuplot"
threads = require "threads"
dofile("binaryAccuracy.lua")
models = require "models"
shuffle = require "shuffle"
Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

------------------------------------------ GLobal vars/params -------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Options')
cmd:option('-lr',0.0003,'Learning rate')
cmd:option('-lrW',1.07,'Learning rate decay')
cmd:option('-momentum',0.95,'Momentum')
cmd:option('-batchSize',1,'batchSize')
cmd:option('-cuda',1,'CUDA')
cmd:option('-sliceSize',36,"Length size of cube around nodule")
cmd:option('-angleMax',0.5,"Absolute maximum angle for rotating image")
cmd:option('-scalingFactor',0.75,'Scaling factor for image')
cmd:option('-scalingFactorVar',0.01,'Scaling factor variance for image')
cmd:option('-clipMin',-1200,'Clip image below this value to this value')
cmd:option('-clipMax',1200,'Clip image above this value to this value')
cmd:option('-nThreads',5,"How many threads to load/preprocess data with?") 
cmd:option('-display',0,"Display images/plots") 
cmd:option('-displayFreq',30,"How often per iteration do we display an image? ") 
cmd:option('-activations',0,"Show activations -- needs -display 1") 
cmd:option('-log',0,"Make log file in /Results/") 
cmd:option('-run',0,'Run neral net straight away (either train or test)')
cmd:option('-test',0,"Test") 
cmd:option('-iterations',10000,"Number of examples to use.") 
cmd:option('-loadModel',0,"Load model") 
cmd:option('-para',3,"Are we using a parallel network? If bigger than 0 then this is equal to number of inputs. Otherwise input number is 1.") 
--cmd:option('-nInputScalingFactors',3,"Number of input scaling factors.") 
-- K fold cv options
cmd:option('-kFold',1,"Are we doing k fold? Default is to train on subsets 1-9 and test on subset0") 
cmd:option('-fold',0,"Which fold to NOT train on") 
--cmd:option('-loadModel',"model1.model","Load model") 
cmd:text()
params = cmd:parse(arg)
params.model = model
params.rundir = cmd:string('results', params, {dir=true})

-------------------------------------------- Model ---------------------------------------------------------
modelPath = "models/para9.model"
if params.loadModel == 1 then 
	print("==> Loading model weights ")
	model = torch.load(modelPath)
else 
	print("==> New model")
	model = models.parallelNetwork()
end
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
-------------------------------------------- Parallel Table parameters -------------------------------------------

if params.para > 0 then
	params.sliceSize = {params.sliceSize,params.sliceSize,64}
	params.scalingFactor = {0.55,1,2.5}
	params.scalingFactorVar = {0.1,0.01,0.001}
	params.angleMax = {0.9,0.5,0.01}
	print("==> Slices ")
	print(params.sliceSize)
	print("==> Scaling factors ")
	print(params.scalingFactor)
	print("==> Scaling factor variances ")
	print(params.scalingFactorVar)
	print("==> Max rotation angles ")
	print(params.angleMax)
end
-------------------------------------------- Loading data with threads ---------------------------------------------------

print(string.format("==> Using %d threads ",params.nThreads))
do
	local options = params -- make an upvalue to serialize over to donkey threads
	donkeys = Threads(
		params.nThreads,
		function()
			require 'torch'
		end,
		function(idx)
			print("==> Initializing threads.")
			params = options -- pass to all donkeys via upvalue
			loadData = require "loadData"
			loadData.Init()
			print("==> Initialized.")
		end
		)
end
function displayImageInit()
	if displayTrue==nil and params.display==1 then
		print("Initializing displays ==>")
		local init = torch.range(1,torch.pow(512,2),1):reshape(512,512)
		local zoom = 0.7
		--init = image.lena()
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
end
function displayImage(inputs,targets,predictions,idx)
		local class = "Class = " .. targets[1][1] .. ". Prediction = ".. predictions[1]
		-- Display rotated images
		-- Middle Slice
		image.display{image = inputs[1][1][{{idx},{},{params.sliceSize[1]/2 +1}}]:reshape(params.sliceSize[1],params.sliceSize[1]), win = imgZ, legend = class}
		image.display{image = inputs[1][2][{{idx},{},{params.sliceSize[2]/2 +1}}]:reshape(params.sliceSize[2],params.sliceSize[2]), win = imgY, legend = class}
		image.display{image = inputs[1][3][{{idx},{},{params.sliceSize[3]/2 +1}}]:reshape(params.sliceSize[3],params.sliceSize[3]), win = imgX, legend = class}

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

function train(inputs,targets)

	if i == nil then 
		print("==> Initalizing training")
		i = 1 
		epochLosses = {}
		batchLosses = {}
		batchLossesMA = {}
		accuraccies = {}
		if model then parameters,gradParameters = model:getParameters() end
		lrChangeThresh = 0.7
		timer = torch.Timer()
	end
	
	if params.cuda == 1 then
		targets = targets:cuda()
	end
		
	function feval(x)
		if x~= parameters then parameters:copy(x) end
		gradParameters:zero()
		predictions = model:forward(inputs[1])
		loss = criterion:forward(predictions,targets)
		dLoss_d0 = criterion:backward(predictions,targets)
		if params.log == 1 then logger:add{['loss'] = loss } end
		model:backward(inputs[1], dLoss_d0)

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
	local ma = 15 
	if i > ma then 
		accMa = accuracciesT[{{-ma,-1}}]:mean()
		print(string.format("Iteration %d accuracy= %f. MA loss of last 20 batches == > %f. MA accuracy ==> %f. Overall accuracy ==> %f ",
		i, accuracy, batchLossesT[{{-ma,-1}}]:mean(), accMa,accuracciesT:mean()))
	end

	--Plot
	if i %  params.displayFreq*3 == 0 then
		gnuplot.figure(1)
		gnuplot.plot({"Train loss",t,batchLossesT})
	end


	if i % 1000 == 0 then
		print("==> Saving weights for ".. modelPath)
		torch.save(modelPath,model)
	end

	if i % 200 == 0 then
		-- Learning rate change
		print("==> Dropping lr from ",params.lr)
		params.lr = params.lr/params.lrW
		print("==> to",params.lr)

	end
	
	displayImageInit()
	if params.display == 1 and displayTrue ~= nil and i % params.displayFreq == 0 then 
		displayImage(inputs,targets,predictions,1)
	end

	xlua.progress(i,params.iterations)
	i = i + 1
	collectgarbage()
end

function test(inputs,targets)
	if i == nil then 
		print("==> Initalizing training")
		i = 1 
		epochLosses = {}
		batchLosses = {}
		batchLossesMA = {}
		accuraccies = {}
	end

	if params.cuda == 1 then
		targets = targets:cuda()
	end

	predictions = model:forward(inputs[1])
	loss = criterion:forward(predictions,targets)

	-- Performance metrics
	accuracy = binaryAccuracy(targets,predictions,params.cuda)
	loss = criterion:forward(predictions,targets)

	accuraccies[#accuraccies + 1] = accuracy
	batchLosses[#batchLosses + 1] = loss 
	accuracciesT = torch.Tensor(accuraccies)
	batchLossesT = torch.Tensor(batchLosses)
	local t = torch.range(1,batchLossesT:size()[1])
	local ma = 15
	if i > ma then 
		accMa = accuracciesT[{{-ma,-1}}]:mean()
		print(string.format("Iteration %d accuracy= %f. MA loss of last 20 batches == > %f. MA accuracy ==> %f. Overall accuracy ==> %f ",
		i, accuracy, batchLossesT[{{-ma,-1}}]:mean(), accMa,accuracciesT:mean()))
	end

	--Plot
	if i % 50 == 0 then
		gnuplot.figure(1)
		gnuplot.plot({"Test loss",t,batchLossesT})
	end

	displayImageInit()
	if params.display == 1 and displayTrue ~= nil and i % 50 == 0 then 
		displayImage(inputs,targets,predictions,1)
	end

	xlua.progress(i,params.iterations)
	i = i + 1
	collectgarbage()
end
 
inputs = {}
targets = {}
if params.run == 1 then 
	if params.test == 1 then params.iterations = 3000 end 
	for i = 1, params.iterations do 
		donkeys:addjob(function()
					x,y = loadData.getBatch(C0,C1,params.batchSize,params.sliceSize,params.clipMin,
					params.clipMax,params.angleMax,params.scalingFactor,params.scalingFactorVar,
					params.test,params.para)
					return x,y
				end,
				function(x,y)
					if params.test == 1 then
						test(x,y)
					else 
						train(x,y)
					end
				end
				)
	end
	donkeys:synchronize()
end

