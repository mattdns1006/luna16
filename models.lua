modelLayers = require "modelLayers"

models = {}

function models.model1()
	local nFiltersInc = 10
	local nFilters = {1,nFiltersInc,nFiltersInc*2,nFiltersInc*3,nFiltersInc*4}
	local filterSizeConv = {3,3,3,3,3}
	local strideConv = {1,1,1,1,1}
	local paddingConv = {1,1,1,1,1}

	local sizeMP = {3,3,3,3,3}
	local strideMP = {2,2,2,2,2}
	local paddingMP = {0,0,0,0,0}

	local model = nn.Sequential()

	local layerNu = 1
	for i = 1,5 do 
		modelLayers.add3DConv(model, layerNu, nFilters, filterSizeConv, strideConv, paddingConv)
		modelLayers.addMP(model, layerNu, nFilters, filterSizeConv, strideConv, paddingConv)

		layerNu = layerNu + 1
	end
	
	return model
end

