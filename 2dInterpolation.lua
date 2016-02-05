require "torch"
require "image"

img = image.lena()
img = img:mean(1) -- make non rgb (sort of grayscale but not quite)

--Sizes
xSize,ySize= img:size()[2],img:size()[3]
xSize,ySize =  6,4 

function rotationMatrix()
	--Returns a rotation matrix
	local rotMatrix = torch.qr(torch.randn(4):reshape(2,2))
	rotMatrix = rotMatrix:cat(torch.zeros(2))
	return rotMatrix
end

function interpolate()
	-- Takes new coordinates, evaluates the coordinates at 
end

rotMatrix = rotationMatrix()

--Coords
x,y =  torch.linspace(1,xSize,xSize), torch.linspace(1,ySize,ySize)
xx = x:repeatTensor(ySize)
yy = xx:index(1,xx:sort():long())
ones = torch.ones(xSize*ySize)
xx, yy, ones = xx:reshape(1,xx:size()[1]),yy:reshape(1,yy:size()[1]),ones:reshape(1,ones:size()[1])
coords = torch.cat({xx,yy,ones},1)
xCoords = x:repeatTensor(ySize):reshape(ySize,xSize)
yCoords = y:repeatTensor(xSize):reshape(xSize,ySize):transpose(1,2)

newCoords = rotMatrix*coords -- new coordinates over which to interpolate









 
