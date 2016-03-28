require "gnuplot"

BinaryConfusionMatrix = {}
BinaryConfusionMatrix.__index = BinaryConfusionMatrix

function BinaryConfusionMatrix.new(threshold,rocInterval)
	assert(threshold and type(threshold) == 'number',"Threshold must be a number")
	local self = {}
	self.cm = torch.zeros(2,2)
	self.threshold = threshold 
	self.rocInterval = rocInterval
	self.results = {}
	return setmetatable(self,BinaryConfusionMatrix)
end

function BinaryConfusionMatrix:allocate(matrix,prediction,target,threshold)
	if target == 0 and prediction < threshold then 
		matrix[1][1] = matrix[1][1] + 1
	elseif target == 0 and prediction > threshold then
		matrix[2][1] = matrix[2][1] + 1
	elseif target == 1 and prediction > threshold then
		matrix[2][2] = matrix[2][2] + 1
	elseif target == 1 and prediction < threshold then
		matrix[1][2] = matrix[1][2] + 1
	else
		print(prediction,target)
		print("Could not add to confusion matrix, please check args.")
	end
end

function BinaryConfusionMatrix:add(prediction,target)
	assert(prediction and type(prediction) == 'number', "Prediction must be a number")
	assert(target and type(target) == 'number',"Target must be a number")
	self.results[#self.results+1] = {prediction,target}
	self:allocate(self.cm,prediction,target,self.threshold) -- add to main Confusion matrix
end

function BinaryConfusionMatrix:roc()
	tpr = {} -- True positive rate
	fpr = {} -- False positive rate
	for threshold = 0.99, 0.01, -self.rocInterval do
		local newCm = torch.zeros(2,2) -- Make new confusion matrix
		for k,v in pairs(self.results) do 
			self:allocate(newCm,v[1],v[2],threshold) -- allocate based on new threshold
		end
		-- calculate tpr/fpr and store
		tpr[#tpr+1] = (newCm[2][2])/(newCm[1][2] + newCm[2][2]) -- TP/(FN + TP)
		fpr[#fpr+1] = (newCm[2][1])/(newCm[1][1] + newCm[2][1]) -- FP/(TN + FP)
	end
	tpr = torch.Tensor{tpr}:squeeze()
	fpr = torch.Tensor{fpr}:squeeze()
	gnuplot.plot(fpr,tpr)
	gnuplot.axis{0,1,0,1}
	gnuplot.xlabel("False positive rate")
	gnuplot.ylabel("True positive rate")
	gnuplot.plotflush()
	gnuplot.grid(true)
	gnuplot.title("ROC")
end

function BinaryConfusionMatrix:performance()
	self.accuracy = torch.diag(self.cm):sum()/self.cm:sum()
	self.precision = (self.cm[2][2])/(self.cm[2][1] + self.cm[2][2])
	self.recall = (self.cm[2][2])/(self.cm[1][2] + self.cm[2][2])
	print(string.format("Accuracy, precision, recall ==>  %f, %f, %f.", self.accuracy,self.precision,self.recall))
end

--Example
function rocExample()
	cm = BinaryConfusionMatrix.new(0.6,0.005)
	time = torch.Timer()
	for i = 1, 1000 do
		target = torch.bernoulli()
		if target == 1 then
			prediction = torch.uniform(0.5,1)
		else
			prediction = torch.uniform(0,0.7)
		end 
		cm:add(prediction,target)
	end
	print(time:time().real)
	cm:roc()
end
