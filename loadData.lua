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
		if params.fullTest == 1 then 
			--print("==> Proper test set regarding competition i.e. imbalanced test set")
			--C0Path ="CSVFILES/subset"..params.fold.."/candidatesTest.csv" 
			--C1Path ="CSVFILES/subset"..params.fold.."/candidatesTest.csv" 
			C0Path ="CSVFILES/subset"..params.fold.."/candidatesTestSmall.csv" 
			C1Path ="CSVFILES/subset"..params.fold.."/candidatesTestSmall.csv" 
			print("==> Full testing on csv files; "..C0Path..", "..C1Path..".")
		else 
			print("==> k fold cross validation training on folds "..params.fold)
			C0Path = "CSVFILES/subset"..params.fold.."/candidatesClass0"..trainTest..".csv"
			C1Path = "CSVFILES/subset"..params.fold.."/candidatesClass1"..trainTest..".csv"
			print("==> "..trainTest.."ing on csv files; "..C0Path..", "..C1Path..".")
		end


		local maxSlice = 0 
		for _, v in pairs(params.sliceSize) do
			if v > maxSlice then
				maxSlice = v
			end
		end
		C0 = Data:new(C0Path,params.clipMin,params.clipMax,maxSlice,params.tid,params.nThreads)
		C1 = Data:new(C1Path,params.clipMin,params.clipMax,maxSlice,params.tid,params.nThreads)
	end
	C0:getNewScan()
	C1:getNewScan()
end

loadData.getBatch = function(data1,data2,batchSize,sliceSize,clipMin,clipMax,angleMax,scalingFactor,scalingFactorVar,test,para,fullTest)
		--Make empty table to loop into
		X = {}
		y = torch.Tensor(batchSize,1)
		relaventInfo = nil
		for i=1, batchSize do
			if fullTest == 1 then qunif = 1 else qunif = 0.5 end
			if torch.uniform() < qunif then 
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
				relaventInfo = data.relaventInfo
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
		return X,y,relaventInfo
end

return loadData
