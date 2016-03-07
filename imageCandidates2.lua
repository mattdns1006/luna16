dofile("readCsv.lua")
require "cunn"
require "cutorch"
require "xlua"
require "image"

Csv = {}
Csv.__index = Csv

function Csv:new(path,clipMn,clipMx,sliceSze)
	local csv = csvToTable(path)

	-- Get all unique values
	local csvUnique= {}
	local numberOfScans = 0
	for k,v in pairs(csv) do
		local fileName = v:split(",")[1]
		if csvUnique[fileName] == nil then 
			csvUnique[fileName] = 1
			numberOfScans = numberOfScans + 1
		else 
		   csvUnique[fileName] = csvUnique[fileName] + 1 
		end
	end
	return setmetatable({allCandidates = csv,
			     allScans = csvUnique,
		     	     nScans = numberOfScans,
			     clipMin = clipMn,
			     clipMax = clipMx,
			     sliceSize = sliceSze
		     		},Csv)
end

-- function that generates random scan with list of all candidates in that scan and appends it as an attribute as well as the clipped image
function Csv:getNewScan() 
	--Draw random number 
	local obsNumber = torch.random(self.nScans)

	--Get a random observation from the CSV
	local i = 0
	self.scanName = {}
	for k,_ in pairs(self.allScans) do 	
		if i == obsNumber then
			self.scanName = k	
			break
		else
			i = i + 1
		end
	end

	-- Get all candidates that belong to that scan
	self.scanCandidates = {}
	for k, v in pairs(self.allCandidates) do
		if self.scanName == v:split(",")[1] then
			self.scanCandidates[#self.scanCandidates + 1] = v 
		end
	end
	local fileInfo = self.scanCandidates[1]:split(",")

	-- Scan specific attributes
	self.imgPath = fileInfo[1]
	self.xSpacing = tonumber(fileInfo[3])
	self.ySpacing = tonumber(fileInfo[4])
	self.zSpacing = tonumber(fileInfo[5])
	self.scanName = scanName
	self.currentCandidate = 1

	-- Load image
	self.img = torch.ShortStorage(self.imgPath)
	self.img = torch.ShortTensor(self.img):double()
	self.img = self.img:view(-1,512,512)
	self.img:clamp(self.clipMin,self.clipMax) -- Clip image to keep only ROI

	-- Initialize candidate (nodule) img at first candidate
	self.Class = tonumber(fileInfo[2])
	self.x = tonumber(fileInfo[6]) 
	self.y = tonumber(fileInfo[7])
	self.z = tonumber(fileInfo[8])

	-- Boolean to note whether we have finished going through all training examples in a scan
	self.finishedScan = false 
end

function Csv:getNextCandidate()

	local fileInfo = self.scanCandidates[self.currentCandidate]:split(",")
	self.Class = tonumber(fileInfo[2])
	self.x = tonumber(fileInfo[6]) 
	self.y = tonumber(fileInfo[7])
	self.z = tonumber(fileInfo[8])

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
	if self.currentCandidate >= #self.scanCandidates then self.finishedScan = true end -- If we are at the end of the batch we flag in order to prompt a recall of getNewScan

end

--[[
--example
C0 = Csv:new("CSVFILES/candidatesClass0Test.csv",-1000,1000,48)
C0:getNewScan()
C0:getNextCandidate()
]]--
