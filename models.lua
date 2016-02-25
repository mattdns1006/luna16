modelLayers = require "modelLayers"

models = {}

-- 64^3
function models.model1()
	nFiltersInc = 10
	nFilters = {1,nFiltersInc,nFiltersInc*2,nFiltersInc*3,nFiltersInc*4}
	filterSizeConv = {3,3,3,3,3}
	strideConv = {1,1,1,1,1}
	paddingConv = {1,1,1,1,1}

	sizeMP = {3,3,3,3,3}
	strideMP = {2,2,2,2,2}
	paddingMP = {1,1,1,1,1}

	model = nn.Sequential()

	layerNu = 2 
	--modelLayers.addBN(model, 1, 1)
	for i = 1,4 do 

		modelLayers.add3DConv(model, layerNu, nFilters, filterSizeConv, strideConv, paddingConv)
		model:add(nn.ReLU())
		modelLayers.addMP(model, layerNu, sizeMP, strideMP, paddingMP)
		modelLayers.addBN(model, layerNu, nFilters)
		layerNu = layerNu + 1
	end
	lastLayerNeurons = 40*3*3*3
	model:add(nn.View(lastLayerNeurons))
	model:add(nn.Linear(lastLayerNeurons,1))
	model:add(nn.Sigmoid())
	
	return model
end

--52^3
function models.model2()
	nFiltersInc = 20
	nFilters = {1,nFiltersInc,nFiltersInc*2,nFiltersInc*3,nFiltersInc*4}
	filterSizeConv = {3,3,3,3,3}
	strideConv = {1,1,1,1,1}
	paddingConv = {1,1,1,1,1}

	sizeMP = {3,3,3,3,3}
	strideMP = {2,2,2,2,2}
	paddingMP = {1,1,1,1,1}

	model = nn.Sequential()

	layerNu = 2 
	--modelLayers.addBN(model, 1, 1)
	for i = 1,4 do 

		modelLayers.add3DConv(model, layerNu, nFilters, filterSizeConv, strideConv, paddingConv)
		--model:add(nn.ReLU())
		model:add(nn.Tanh())
		modelLayers.addMP(model, layerNu, sizeMP, strideMP, paddingMP)
		modelLayers.addBN(model, layerNu, nFilters)
		layerNu = layerNu + 1
	end
	lastLayerNeurons = 640 
	model:add(nn.View(lastLayerNeurons))
	model:add(nn.Linear(lastLayerNeurons,1))
	model:add(nn.Sigmoid())
	
	return model
end

return models
