dofile("readCsv.lua")
require "cunn"
require "cutorch"

--candidateCsv = csvToTable("CSVFILES/candidatesCleaned.csv")
--trainCSV = csvToTable("CSVFILES/CandidatesTrain8.csv")
--trainCSV = csvToTable("CSVFILES/CandidatesTrain8.csv")

Candidate = {}
Candidate.__index = Candidate

function Candidate:new(csv,obs)
	local fileInfo = csv[obs]:split(",")	
	return setmetatable({ AllFileInfo = fileInfo,
			      imgPath = fileInfo[1],
			      Class = fileInfo[2],
			      xSpacing = tonumber(fileInfo[3]),
			      ySpacing = tonumber(fileInfo[4]),
			      zSpacing = tonumber(fileInfo[5]),
			      x = tonumber(fileInfo[6]),
			      y = tonumber(fileInfo[7]),
			      z = tonumber(fileInfo[8])
				},Candidate)
end

function Candidate:loadImg(clipMin,clipMax,sliceSize)

	--torch.setdefaulttensortype("torch.ShortTensor")
	local img = torch.ShortStorage(self.imgPath)

	--img = torch.Tensor(img):double()
	img = torch.ShortTensor(img):double()
	--torch.setdefaulttensortype("torch.DoubleTensor")

	img = img:view(-1,512,512)

	-- Clip image to keep only ROI
	img[img:lt(clipMin)] = clipMin
	img[img:gt(clipMax)] = clipMax

	-- Remove mean
	img = img - img:mean()

	-- Function to check if nodule coords are near edge
	function checkCoords(coord, coordMax, sliceSize)
		returnCoord = coord
		if coord <= sliceSize then 
			returnCoord =  sliceSize + 1

		elseif sliceSize >= (coordMax - coord) then 
			returnCoord = (coordMax - sliceSize - 1)
		end
		return returnCoord
	end

	self.z = checkCoords(self.z,img:size()[1],sliceSize)
	self.y = checkCoords(self.y,img:size()[2],sliceSize)
	self.x = checkCoords(self.x,img:size()[3],sliceSize)

	return img
end

