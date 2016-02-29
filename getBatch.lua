function getBatch(data,batchSize,xBatchTensor,yBatchTensor,sliceSize,clipMin,clipMax,angleMax)

	indices = torch.randperm(#data)[{{1,batchSize}}]
	batch = {}
	local j = 1
	for i=1, batchSize do
		batch[j] = data[indices[i]]
		j = j + 1
	end

	-- For each image file in a batch. 1/ Load it 2/ Rotate it /3 Place it in batch tensor
	--local xBatchTensor = torch.Tensor(batchSize,1,params.sliceSize,params.sliceSize,params.sliceSize)
	--local yBatchTensor = torch.Tensor(batchSize,1)

	for k,v in ipairs(batch) do 

		obs = Candidate:new(batch,k)
		--x = obs:loadImg(clipMin,clipMax,sliceSize)[{{1,sliceSize},{1,sliceSize},{1,sliceSize}}]:reshape(1,sliceSize,sliceSize,sliceSize)
		x = rotation3d(obs, angleMax, sliceSize, clipMin, clipMax,1):reshape(1,sliceSize,sliceSize,sliceSize)
		y = obs.Class
		xBatchTensor[k] = x 
		yBatchTensor[k] = y 
	end

	collectgarbage()

	--return xBatchTensor, yBatchTensor, batch
end

--[[
function getBatch(data,batchSize)

	indices = torch.randperm(#data)[{{1,batchSize}}]
	batch = {}
	local j = 1
	for i=1, batchSize do
		batch[j] = data[indices[i] -]-
		j = j + 1
	end

	-- For each image file in a batch. 1/ Load it 2/ Rotate it /3 Place it in batch tensor
	local xBatchTensor = torch.Tensor(batchSize,1,params.sliceSize,params.sliceSize,params.sliceSize)
	local yBatchTensor = torch.Tensor(batchSize,1)
	if params.cuda == 1 then 
		xBatchTensor, yBatchTensor = xBatchTensor:cuda(), yBatchTensor:cuda()
	end


	for k,v in ipairs(batch) do 
		obs = Candidate:new(batch,k)
		x = rotation3d(obs, angleMax, params.sliceSize, params.clipMin, params.clipMax,1):reshape(1,params.sliceSize,params.sliceSize,params.sliceSize)
		y = obs.Class
		xBatchTensor[k] = x 
		yBatchTensor[k] = y 
	end

	return xBatchTensor, yBatchTensor, batch
end
]]--
