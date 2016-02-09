require "torch"
require "image"
require "cunn"
require "cutorch"
require "image"

function rotationMatrix()
	--Returns a 4D rotation matrix
	local rotMatrix = torch.qr(torch.randn(9):reshape(3,3))
	rotMatrix = rotMatrix:cat(torch.zeros(3))
	return rotMatrix
end

rotMatrix = rotationMatrix()
print("==> Rotation matrix")
print(rotMatrix)

-- Toy example
--xSize,ySize,zSize =  12,10,6
xSize,ySize,zSize = 4,4,4 
totSize = xSize*ySize*zSize
img = torch.linspace(1,totSize,totSize):reshape(ySize,xSize,zSize)
--]]


--Coords
x,y,z = torch.linspace(1,xSize,xSize), torch.linspace(1,ySize,ySize), torch.linspace(1,zSize,zSize)
xx = x:repeatTensor(ySize*zSize)
yy = y:repeatTensor(xSize):sort():long():repeatTensor(zSize):double()
zz = xx:index(1,xx:sort():long())
ones = torch.ones(totSize)

coords = torch.cat({xx:reshape(totSize,1),yy:reshape(totSize,1),zz:reshape(totSize,1),ones:reshape(totSize,1)},2)

--Rotated coords
newCoords = coords*rotMatrix:transpose(1,2)

minMax = torch.min(torch.Tensor{xSize,ySize,zSize})
newCoords[newCoords:lt(2)] = 2
newCoords[newCoords:gt(minMax-1)] = minMax - 1


-- Need all 8 corners of the cube in which newCoords[i,j,k] lies
-- ozo means onesZerosOnes
zzz = torch.zeros(totSize,3)
ooo = torch.ones(totSize,3)

function fillzo(zzz_ooo,z_o,column)
	zzz_ooo_clone = zzz_ooo:clone()
	zzz_ooo_clone:select(2,column):fill(z_o)
	return zzz_ooo_clone
end
ozz = fillzo(zzz,1,1)
zoz = fillzo(zzz,1,2)
zzo = fillzo(zzz,1,3)
zoo = fillzo(ooo,0,1)
ozo = fillzo(ooo,0,2)
ooz = fillzo(ooo,0,3)

xyz = newCoords:clone():floor()
x1y1z1 = newCoords:clone():ceil()
x1yz = xyz + ozz
xy1z = xyz + zoz
xyz1 = xyz + zzo
x1y1z = xyz + ooz
x1yz1 = xyz + ozo
xy1z1 = xyz + zoo

-- Subtract the new coordinates from the 8 corners to get our distances ijk which are our weights
ijk = newCoords - xyz
i,j,k = ijk[{{},{1}}], ijk[{{},{2}}], ijk[{{},{3}}]
i1j1k1 = newCoords - xyz -- (1-i)(1-j)(1-k)
i1,j1,k1 = ones - ijk[{{},{1}}], ones - ijk[{{},{2}}], ones - ijk[{{},{3}}]

function flattenIndices(sp_indices, shape)
	local sp_indices = sp_indices - 1
	local n_elem, n_dim = sp_indices:size(1), sp_indices:size(2)
	local flat_ind = torch.LongTensor(n_elem):fill(1)

	local mult = 1
	for d = n_dim, 1, -1 do
		flat_ind:add(sp_indices[{{}, d}] * mult)
		mult = mult * shape[d]
	end
	return flat_ind
end

function getElements(tensor, sp_indices)
	local sp_indices = sp_indices:long()
	local flat_indices = flattenIndices(sp_indices, tensor:size()) 
	local flat_tensor = tensor:view(-1)
	return flat_tensor:index(1, flat_indices)
end

fxyz = getElements(img,xyz)
fx1yz = getElements(img,x1yz)
fxy1z = getElements(img,xy1z)
fxyz1 = getElements(img,xyz1)
fx1y1z = getElements(img,x1y1z)
fx1yz1 = getElements(img,x1yz1)
fxy1z1 = getElements(img,xy1z1)
fx1y1z1 = getElements(img,x1y1z1)


imgInterpolate = i1:cmul(j1):cmul(k1):cmul(fxyz) +  
	      i:cmul(j1):cmul(k1):cmul(fx1yz) + 
	      i1:cmul(j):cmul(k1):cmul(fxy1z) + 
	      i1:cmul(j1):cmul(k):cmul(fxyz1) + 
	      i:cmul(j):cmul(k1):cmul(fx1y1z) + 
	      i:cmul(j1):cmul(k):cmul(fx1yz1) + 
	      i1:cmul(j):cmul(k):cmul(fxy1z1) + 
	      i:cmul(j):cmul(k):cmul(fx1y1z1)  



 
