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
f = io.open(filePath:gsub(".raw",".mhd"),"r")
meta = f:read("*all")
f:close()

torch.setdefaulttensortype("torch.ShortTensor")
img= torch.ShortStorage(filePath)
img = torch.Tensor(img):double()
--img = torch.Tensor(f1):view(1,1,-1,512,512):double()
--img = normalize:forward(img)
img = img:view(-1,512,512)

-- Try view
obs = 30
torch.setdefaulttensortype("torch.DoubleTensor")
--image.display(img[obs])
imgSubset = img[{{1,100},{1,100},{1,100}}]
torch.save("lung3Dexample100.dat",imgSubset)
torch.save("lung3Dexamplefull.dat",img)











