dofile("readCsv.lua")
require "cunn"
require "cutorch"
require "xlua"
require "image"

function tableLength(T)
  local count = 0
    for _ in pairs(T) do count = count + 1 end
      return count
end

Data = {}
Data.__index = Data 

function Data:new(path,clipMn,clipMx,sliceSze)
	local csv = csvToTable(path)

	-- Get all unique values
	local csvUnique= {}
	for k,v in pairs(csv) do
		local fileName = v:split(",")[1]
		if csvUnique[fileName] == nil then 
			csvUnique[fileName] = 1
		else 
		   csvUnique[fileName] = csvUnique[fileName] + 1 
		end
	end

	--Shuffle the data - so different threads look at different scans at each stage.
	

	local currentScanNum = 1

	return setmetatable({allCandidates = csv,
			     allScans = csvUnique,
		     	     nScans = tableLength(csvUnique),
			     clipMin = clipMn,
			     clipMax = clipMx,
			     sliceSize = sliceSze,
			     currentScanNumber = currentScanNum
		     		},Data)
end

-- function that generates random scan with list of all candidates in that scan and appends it as an attribute as well as the clipped image
function Data:getNewScan() 
	--Draw random number 
	--local obsNumber = torch.random(self.nScans)

	--Get a random observation from the CSV
	local i = 1 
	self.scanName = {}
	for k,_ in pairs(self.allScans) do 	
		if i == self.currentScanNumber then
			self.scanName = k	
			break
		else
			i = i + 1
		end
	end
	--print("==> New scan number ",self.currentScanNumber, " out of ", self.nScans)
	--print(self.scanName)

	-- Get all candidates that belong to that scan
	self.scanCandidates = {}
	for k, v in pairs(self.allCandidates) do
		if self.scanName == v:split(",")[1] then
			self.scanCandidates[#self.scanCandidates + 1] = v 
		end
	end
	local fileInfo = self.scanCandidates[1]:split(",")

	-- Scan specific attributes

	self.seriesuid = fileInfo[1]
	self.imgPath = fileInfo[6]
	self.xSpacing = tonumber(fileInfo[7])
	self.ySpacing = tonumber(fileInfo[8])
	self.zSpacing = tonumber(fileInfo[9])
	self.scanName = scanName
	self.currentCandidate = 1

	-- Load image
	self.img = torch.ShortStorage(self.imgPath)
	self.img = torch.ShortTensor(self.img):double()
	self.img = self.img:view(-1,512,512)
	self.img:clamp(self.clipMin,self.clipMax) -- Clip image to keep only ROI

	-- Initialize candidate (nodule) img at first candidate
	self.coordX = tonumber(fileInfo[2])
	self.coordY = tonumber(fileInfo[3])
	self.coordZ = tonumber(fileInfo[4])
	self.Class = tonumber(fileInfo[5])
	self.x = tonumber(fileInfo[10]) 
	self.y = tonumber(fileInfo[11])
	self.z = tonumber(fileInfo[12])
	self.relaventInfo = {self.seriesuid,self.coordX,self.coordY,self.coordZ,self.Class}


	-- Boolean to note whether we have finished going through all training examples in a scan
	self.finishedScan = false 
end

function Data:getNextCandidate()

	local fileInfo = self.scanCandidates[self.currentCandidate]:split(",")

	self.coordX = tonumber(fileInfo[2])
	self.coordY = tonumber(fileInfo[3])
	self.coordZ = tonumber(fileInfo[4])
	self.Class = tonumber(fileInfo[5])
	self.x = tonumber(fileInfo[10]) 
	self.y = tonumber(fileInfo[11])
	self.z = tonumber(fileInfo[12])
	self.relaventInfo = {self.seriesuid,self.coordX,self.coordY,self.coordZ,self.Class}

	-- Function to check if nodule coords are near edge
	function checkCoords(coord, coordMax, sliceSize)
		returnCoord = coord
		if coord <= sliceSize/2 then 
			
			returnCoord =  sliceSize/2 + 1
			--print( "Coord " .. coord .. " is less than half slice size of .. " .. sliceSize/2 .. " changing to " .. returnCoord)

		elseif sliceSize/2 >= (coordMax - coord) then 
			returnCoord = (coordMax - sliceSize/2 - 1)
			--print ("Coord " .. coord .. " is greater than half slice size of .. " .. sliceSize/2 .. " changing to " .. returnCoord)
		end
		return returnCoord
	end

	self.z = checkCoords(self.z,self.img:size()[1],self.sliceSize)
	self.y = checkCoords(self.y,self.img:size()[2],self.sliceSize)
	self.x = checkCoords(self.x,self.img:size()[3],self.sliceSize)

	self.currentCandidate = self.currentCandidate + 1
	if self.currentCandidate >= #self.scanCandidates then 
		self.finishedScan = true 
		
		if self.currentScanNumber >= self.nScans then
			print("Been through all the scans, starting from scan number one.")
			self.currentScanNumber = 1
		else
			self.currentScanNumber = self.currentScanNumber + 1
		end
	end -- If we are at the end of the batch we flag in order to prompt a recall of getNewScan

end

function imageCandidatesEg()

	eg = "CSVFILES/subset40/candidatesTest.csv"
	data = Data:new(eg,-1200,1200,42)

	for i = 1, 18 do 
		local i = 1
		data:getNewScan()
		while data.finishedScan ~= true do 
			data:getNextCandidate()
			i = i + 1
			print(data.relaventInfo)
		end
		print("n candidates = " , i)
	end
end





