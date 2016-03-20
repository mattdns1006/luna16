BinaryConfusionMatrix = {}
BinaryConfusionMatrix.__index = BinaryConfusionMatrix

function BinaryConfusionMatrix.new(threshold)
	assert(threshold and type(threshold) == 'number',"Threshold must be a number")
	local self = {}
	self.cm = torch.zeros(2,2)
	self.threshold = threshold 
	return setmetatable(self,BinaryConfusionMatrix)
end

function BinaryConfusionMatrix:add(prediction,target)
	assert(prediction and type(prediction) == 'number', "Prediction must be a number")
	assert(target and type(target) == 'number',"Target must be a number")

	if target == 0 and prediction < self.threshold then 
		self.cm[1][1] = self.cm[1][1] + 1
	elseif target == 0 and prediction > self.threshold then
		self.cm[2][1] = self.cm[2][1] + 1
	elseif target == 1 and prediction > self.threshold then
		self.cm[2][2] = self.cm[2][2] + 1
	elseif target == 1 and prediction < self.threshold then
		self.cm[1][2] = self.cm[1][2] + 1
	else
		print("Could not add to cm")
	end
end

function BinaryConfusionMatrix:performance()
	self.accuracy = torch.diag(self.cm):sum()/self.cm:sum()
	self.precision = (self.cm[2][2])/(self.cm[2][1] + self.cm[2][2])
	self.recall = (self.cm[2][2])/(self.cm[1][2] + self.cm[2][2])
	print(string.format("Accuracy, precision, recall ==>  %f, %f, %f.", self.accuracy,self.precision,self.recall))
end


