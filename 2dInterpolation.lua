require "torch"
require "image"

img = image.lena()
img = img:mean(1) -- make non rgb (sort of grayscale but not quite)

--Sizes
xSize,ySize= img:size()[2],img:size()[3]
totSize = xSize*ySize

-- Toy example
--[[
xSize,ySize =  12,18 
totSize = xSize*ySize
img = torch.linspace(1,totSize,totSize):reshape(ySize,xSize)
--]]

function rotationMatrix()
	--Returns a rotation matrix
	local rotMatrix = torch.qr(torch.randn(4):reshape(2,2))
	rotMatrix = rotMatrix:cat(torch.zeros(2))
	return rotMatrix
end

function getNearestCoords(coordinates)
end



rotMatrix = rotationMatrix()
angle = 0.1*math.pi 
rotMatrix = torch.Tensor{torch.cos(angle),-torch.sin(angle),0,
			torch.sin(angle),torch.cos(angle),0}:reshape(2,3)

--Coords
x,y =  torch.linspace(1,xSize,xSize), torch.linspace(1,ySize,ySize)
xx = x:repeatTensor(ySize)
yy = xx:index(1,xx:sort():long())
minX,minY = torch.min(x),torch.min(y)
minXY = torch.max(torch.Tensor{minX,minY})
maxX, maxY = torch.max(x), torch.max(y)
maxXY = torch.min(torch.Tensor{maxX,maxY})

ones = torch.ones(totSize)
xx, yy, ones = xx:reshape(1,xx:size()[1]),yy:reshape(1,yy:size()[1]),ones:reshape(1,ones:size()[1])
coords = torch.cat({xx,yy,ones},1)
xCoords = x:repeatTensor(ySize):reshape(ySize,xSize)
yCoords = y:repeatTensor(xSize):reshape(xSize,ySize):transpose(1,2)

--newCoords = rotMatrix*coords -- new coordinates over which to interpolate
newCoords = coords:transpose(1,2)*rotMatrix:transpose(1,2) -- new coordinates over which to interpolate

-- Get four nearest coordinates which are whole numbers
newCoordsFloor = newCoords:clone()
oneZero = torch.ones(totSize):cat(torch.zeros(totSize)):reshape(2,totSize)
zeroOne = torch.zeros(totSize):cat(torch.ones(totSize)):reshape(2,totSize)
xy = newCoordsFloor:apply(torch.floor)
x1y = newCoordsFloor + oneZero
xy1 = newCoordsFloor + zeroOne 
x1y1 = xy + 1 

-- If new coordinates are outside of the original coordinate system then assign nearest max/min value
xy[xy:lt(minXY)] = minXY
x1y[x1y:lt(minXY)] = minXY
xy1[xy1:lt(minXY)] = minXY
x1y1[x1y1:lt(minXY)] = minXY

xy[xy:gt(maxXY)] = maxXY
x1y[x1y:gt(maxXY)] = maxXY
xy1[xy1:gt(maxXY)] = maxXY
x1y1[x1y1:gt(maxXY)] = maxXY

-- Substract the new coordinates from the four corners
ij = newCoords - xy -- difference between coordinates and their lower right corner
i1j1 = x1y1 - newCoords -- difference between coordinates and their lower right corner
i = ij[{{},{1}}]
j = ij[{{},{2}}]
i1 = i1j1[{{},{1}}]
j1 = i1j1[{{},{2}}]

img = img:reshape(512,512)

-- Evaluates img value using a set of coordinates
function Fxy(image,coords)
	local fxy = {}
	local coordsLen = coords:size()[1]
	for i=1,coordsLen do 
		fxy[i] = image[coords[i][1]][coords[i][2]]
	end
	return torch.Tensor{fxy}:reshape(coordsLen,1)
end

-- Get corner values
fxy = Fxy(img,xy)
fx1y = Fxy(img,x1y)
fxy1 = Fxy(img,xy1)
fx1y1 = Fxy(img,x1y1)

function interpolate(fxy,fx1y,fxy1,fx1y1,i,j,i1,j1)
	-- Takes new coordinates, evaluates the coordinates at 
	fxy = torch.cmul(j1,fxy)
	fx1y = torch.cmul(j,fx1y)
	fxy1 = torch.cmul(j1,fxy1)
	fx1y1 = torch.cmul(j,fx1y1)
	interpolated = torch.cmul(i1,(fxy+fx1y)) + torch.cmul(i,(fxy1+fx1y1))
	return interpolated
	
end

newImg = interpolate(fxy,fx1y,fxy1,fx1y1,i,j,i1,j1)
newImg = newImg:reshape(512,512)



























 
