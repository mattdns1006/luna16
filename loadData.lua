require "image"
require "paths"
require "cunn"
dofile("readCsv.lua")


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

--Load file info
function getFileInfo(obs)
	local annotation = getAnnotationXYZ(obs)
	local filePath = annotation["path"]
	local meta, spacingTable = getMeta(filePath)
	--print("==> Loading file path")
	--print(annotation)
	fileInfo = {}
	fileInfo["annotation"] = annotation
	fileInfo["spacing"] = spacingTable
	fileInfo["meta"] = meta
	return fileInfo 
end 

--Raw image
function getImg(filePath)
	-- ALl raw data is short
	torch.setdefaulttensortype("torch.ShortTensor")
	local img = torch.ShortStorage(filePath)
	img = torch.Tensor(img):double() -- convert to double for reshape
	img = img:view(-1,512,512)
	return img
end














