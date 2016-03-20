require "nn"

MovingAverage = {}
MovingAverage.__index = MovingAverage
function MovingAverage.new(ma)
	local self = {}
	self.conv = nn.TemporalConvolution(1,1,ma,1)
	self.conv.weight = torch.repeatTensor(torch.Tensor{1/ma},ma):reshape(1,ma)
	self.conv.bias = torch.Tensor{0}
	return setmetatable(self,MovingAverage)
end

function MovingAverage:forward(x)
	assert(x:dim()==1,"Input vector needs to be of dimension 1")
	return self.conv:forward(x:reshape(x:size()[1],1))
end


