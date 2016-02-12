require "image"
require "paths"
require "cunn"
dofile("readCsv.lua")

-- ALl raw data is short
torch.setdefaulttensortype("torch.ShortTensor")

--Load file info
function getFileInfo(obs)
	local annotation = getAnnotationXYZ(obs)
	local filePath = annotation["path"]
	--print("==> Loading file path")
	--print(annotation)
	return annotation, filePath
end 

--Meta data
function getMeta(filePath)
	local f = io.open(filePath:gsub(".raw",".mhd"),"r")
	local meta = f:read("*all")
	--print("==> Meta data")
	local spacing = meta:split("\n")[10] -- assumes 10th value in table
	spacing = spacing:split(" ")
	local spacingTable = {}
	spacingTable["x"] = tonumber(spacing[3])
	spacingTable["y"] = tonumber(spacing[4])
	spacingTable["z"] = tonumber(spacing[5])
	f:close()
	return meta, spacingTable 
end

--Raw image
function getImg(filePath)
	local img = torch.ShortStorage(filePath)
	img = torch.Tensor(img):double() -- convert to double for reshape
	--img = img:view(-1,512,512)
	return img
end

obs = 1
for i = 1,100 do
	obs = obs + 1
	annotation, filePath = getFileInfo(obs)
	metaData,spacing = getMeta(filePath)
	print(spacing)
	--img = getImg(filePath)
end

--Saving
--[[
torch.setdefaulttensortype("torch.DoubleTensor")
imgSubset = img[{{1,196},{200,400},{100,300}}]
torch.save("lung3Dexample200.dat",imgSubset)
--torch.save("lung3Dexamplefull.dat",img)
]]--










