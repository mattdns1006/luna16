-------------------------------------------- Loading data ---------------------------------------------------

local loadData = {}

loadData.Init = function()
	dofile("imageCandidates.lua")
	dofile("3dInterpolation3.lua")

	if params.kFold == 1 then 
		if params.test == 1 then
			trainTest = "Test"
		else
			trainTest = "Train"
		end
		print("==> k fold cross validation leaving subset "..params.fold.." out for testing")
		local C0Path = "CSVFILES/subset"..params.fold.."/candidatesClass0"..trainTest..".csv"
		local C1Path = "CSVFILES/subset"..params.fold.."/candidatesClass1"..trainTest..".csv"

		print("==> "..trainTest.."ing on csv files; "..C0Path..", "..C1Path..".")
		C0 = Data:new(C0Path,params.clipMin,params.clipMax,params.sliceSize[1])
		C1 = Data:new(C1Path,params.clipMin,params.clipMax,params.sliceSize[1])
	end
	C0:getNewScan()
	C1:getNewScan()
end

loadData.getBatch = function(data1,data2,batchSize,sliceSize,clipMin,clipMax,angleMax,scalingFactor,scalingFactorVar,test,para)
		--Make empty table to loop into
		X = {}
		y = torch.Tensor(batchSize,1)
		for i=1, batchSize do
			if torch.uniform() < 0.5 then 
				data = data1 
			else 
				data = data2 
			end
			
			if data.finishedScan == true then
				data:getNewScan()
			else
				data:getNextCandidate()
			end

			if para > 1 then

				x = {}
				X[i] = x
				for i =1, para do
					--[[
					print("scalingFactor",scalingFactor[i])
					print("scalingFactorVar",scalingFactorVar[i])
					print("angleMax",angleMax[i])
					]]--
				   x[i] = rotation3d(data, angleMax[i], sliceSize[i], clipMin, clipMax, scalingFactor[i] ,
				   	  scalingFactorVar[i], test):reshape(1,1,sliceSize[i],sliceSize[i],sliceSize[i]):cuda()
				 end
			else 
				X[i] = rotation3d(data, angleMax, sliceSize, clipMin, clipMax, scalingFactor, 
				scalingFactorVar, test):reshape(1,1,sliceSize,sliceSize,sliceSize)
			end
			y[i] = data.Class

		end
		collectgarbage()
		return X,y
end

return loadData
