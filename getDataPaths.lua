require "image"
require "paths"
require "cunn"

path = "subset0/"
imgPathsRaw = {}
imgPathsMhd = {}

local obs = 1
for file in paths.iterfiles(path) do
	if file:sub(-3,-1) == "raw" then 
		imgPathsRaw[obs] = path..file
		imgPathsMhd[obs] = path.. file:gsub("raw","mhd")
		obs = obs + 1
	end
end

require "nn"
normalize = nn.VolumetricBatchNormalization(1)

filePath = imgPathsRaw[10] 
print("==> Loading file path")
print(filePath)

--Meta data
f = io.open(filePath:gsub(".raw",".mhd"),"r")
meta = f:read("*all")
print(meta)
f:close()

--Raw image
torch.setdefaulttensortype("torch.ShortTensor")
img= torch.ShortStorage(filePath)
img = torch.Tensor(img):double()
img = img:view(-1,512,512)

--Saving
torch.setdefaulttensortype("torch.DoubleTensor")
imgSubset = img[{{1,196},{200,400},{100,300}}]
torch.save("lung3Dexample200.dat",imgSubset)
--torch.save("lung3Dexamplefull.dat",img)











