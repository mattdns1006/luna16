dofile("readCsv.lua")

csv = csvToTable("CSVFILES/annotationsNVC.csv") 

annotationImg = {}
function annotationImg.new(obs)
	local self = {}
	self.AllfileInfo = csv[obs]:split(",")

	self.diameter = tonumber(self.AllfileInfo[5])
	self.imgPath = self.AllfileInfo[6]:gsub(".mhd",".raw")
	self.imgMetaPath = self.imgPath:gsub(".raw",".mhd")

	self.noduleCoords = {}
	self.noduleCoords["x"], self.noduleCoords["y"], self.noduleCoords["z"] = tonumber(self.AllfileInfo[7]), tonumber(self.AllfileInfo[8]) , tonumber(self.AllfileInfo[9])
	
	-- Get spacing info
	local f = io.open(self.imgMetaPath,"r")
	local meta = f:read("*all")
	local spacing = meta:split("\n")[10] -- assumes 10th value in table
	spacing = spacing:split(" ")
	spacingTable = {}
	spacingTable["x"] = tonumber(spacing[3])
	spacingTable["y"] = tonumber(spacing[4])
	spacingTable["z"] = tonumber(spacing[5])
	f:close()
	self.spacing = torch.diag(torch.Tensor{1/spacingTable["z"],1/spacingTable["y"],1/spacingTable["x"]})

	---- METHODS ----

	-- Function to load original image and center around nodule given sliceSize
	function self.loadImg(sliceSize,clipMin,clipMax)
		torch.setdefaulttensortype("torch.ShortTensor")
		local img = torch.ShortStorage(self.imgPath)
		img = torch.Tensor(img):double()
		img = img:view(-1,512,512)

		torch.setdefaulttensortype("torch.DoubleTensor")

		-- Clip image to keep only ROI
		img[img:lt(clipMin)] = clipMin
		img[img:gt(clipMax)] = clipMax

		z,y,x = self.noduleCoords["z"], self.noduleCoords["y"], self.noduleCoords["x"] 
		imgSub = img:sub(z-sliceSize,z+sliceSize-1,y-sliceSize,y+sliceSize-1,x-sliceSize,x+sliceSize-1)
		return img,imgSub
	end
		
	return self
end



--eg = annotationImg.new(10)


