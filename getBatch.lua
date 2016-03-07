function getBatch(data,batchSize,xBatchTensor,yBatchTensor,sliceSize,clipMin,clipMax,angleMax,scalingFactor)

	-- For each image file in a batch. 1/ Load it 2/ Rotate it /3 Place it in batch tensor

	for i=1, batchSize do
		if data.finishedScan == true then
			data:getNewScan()
		else
			data:getNextCandidate()
		end
		local x = rotation3d(data, angleMax, sliceSize, clipMin, clipMax, scalingFactor):reshape(1,sliceSize,sliceSize,sliceSize)
		local y = data.Class
		xBatchTensor[i] = x 
		yBatchTensor[i] = y 
	end

	collectgarbage()

end

