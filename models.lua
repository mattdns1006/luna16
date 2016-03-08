modelLayers = require "modelLayers"

models = {}

-- 64^3
function models.model1()
	nFiltersInc = 64 
	nFilters = {1,nFiltersInc,nFiltersInc*2,nFiltersInc*3,nFiltersInc*4}
	filterSizeConv = {5,5,5,5,5}
	strideConv = {1,1,1,1,1}
	paddingConv = {2,2,2,2,2}

	sizeMP = {3,3,3,3,3}
	strideMP = {2,2,2,2,2}
	paddingMP = {1,1,1,1,1}

	model = nn.Sequential()

	layerNu = 2 
	modelLayers.addBN(model, 1, 1)
	--model:add(nn.VolumetricConvolution(1,1,2,2,2,1,1,1,0,0,0))
	for i = 1,3 do 

		modelLayers.add3DConv(model, layerNu, nFilters, filterSizeConv, strideConv, paddingConv)
		--model:add(nn.ReLU())
		--model:add(nn.PReLU())
		model:add(nn.Tanh())
		modelLayers.addMP(model, layerNu, sizeMP, strideMP, paddingMP)
		modelLayers.addBN(model, layerNu, nFilters)
		layerNu = layerNu + 1
	end
	lastLayerNeurons = 192*4*4*4
	model:add(nn.View(lastLayerNeurons))
	--model:add(nn.Linear(lastLayerNeurons,lastLayerNeurons))
	model:add(nn.Linear(lastLayerNeurons,1))
	model:add(nn.Sigmoid())
	
	return model
end

--52^3
function models.model2()
	nFiltersInc = 32
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
	lastLayerNeurons = 1024
	model:add(nn.View(lastLayerNeurons))
	model:add(nn.Linear(lastLayerNeurons,1))
	model:add(nn.Sigmoid())
	
	return model
end

function models.model3()

	nFiltersInc = 64 
	Filters = {1,nFiltersInc,nFiltersInc*2,nFiltersInc*3,nFiltersInc*4}
	filterSizeConv = {3,3,3,3,3}
	strideConv = {1,1,1,1,1}
	paddingConv = {1,1,1,1,1}

	filterSizeConv = {5,5,5,5,5}
	strideConv = {1,1,1,1,1}
	paddingConv = {2,2,2,2,2}

	sizeMP = {3,3,3,3,3}
	strideMP = {2,2,2,2,2}
	addingMP = {1,1,1,1,1}

	model = nn.Sequential()
	c = nn.ConcatTable()
	modelLayers.add3DConv(c, layerNu, nFilters, filterSizeConv, strideConv, paddingConv)
	modelLayers.add3DConv(c, layerNu, nFilters, filterSizeConv, strideConv, paddingConv)

	p1 = nn.ParallelTable()
	p1:add(nn.Tanh())
	p1:add(nn.Tanh())
	model:add(p1)
	
	p2 = nn.ParallelTable()
	modelLayers.addMP(model, layerNu, sizeMP, strideMP, paddingMP)
	modelLayers.addMP(model, layerNu, sizeMP, strideMP, paddingMP)
	model:add(p2)
	
	model:add(nn.JoinTable(2))
	return model

end
return models
