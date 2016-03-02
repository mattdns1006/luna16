function getBatch(data,batchSize,xBatchTensor,yBatchTensor,sliceSize,clipMin,clipMax,angleMax,scalingFactor)

	local indices = torch.randperm(#data)[{{1,batchSize}}]
	local batch = {}
	local j = 1
	for i=1, batchSize do
		batch[j] = data[indices[i]]
		j = j + 1
	end

	-- For each image file in a batch. 1/ Load it 2/ Rotate it /3 Place it in batch tensor

	for k,v in ipairs(batch) do 

		local obs = Candidate:new(batch,k)
		--x = obs:loadImg(clipMin,clipMax,sliceSize)[{{1,sliceSize},{1,sliceSize},{1,sliceSize}}]:reshape(1,sliceSize,sliceSize,sliceSize)
		local x = rotation3d(obs, angleMax, sliceSize, clipMin, clipMax,scalingFactor):reshape(1,sliceSize,sliceSize,sliceSize)
		local y = obs.Class
		xBatchTensor[k] = x 
		yBatchTensor[k] = y 
	end

	collectgarbage()

end

