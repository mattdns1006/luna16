require "torch"
require "image"
require "cunn"
require "cutorch"
require "image"
dofile("loadData.lua")
dofile("image.lua")

function rotationMatrix(angle, sliceSize)
	--Returns a 3D rotation matrix
	--local rotMatrix, r  = torch.qr(torch.randn(9):reshape(3,3))
	--rotMatrix = rotMatrix:cat(torch.zeros(3):fill(0))
	local rotMatrix = torch.Tensor{1,0,0,0,0,torch.cos(angle),-torch.sin(angle),0,0,torch.sin(angle),torch.cos(angle),0}:reshape(3,4)
	
	--local rotMatrix = torch.Tensor{1,0,0,-sliceSize,0,torch.cos(angle),-torch.sin(angle),-sliceSize,0,torch.sin(angle),torch.cos(angle),-sliceSize}:reshape(3,4)
	return rotMatrix
end

function rotation3d(img, angleMax, spacing, sliceSize)

	-- Get dimensions and clone to make contiguous 

	xSize,ySize,zSize = img:size()[1], img:size()[2], img:size()[3]
	totSize = xSize*ySize*zSize
	newTensor = torch.Tensor(xSize,ySize,zSize)
	img = newTensor:copy(img)

	--local angle = torch.uniform(0,2*math.pi)
	angle = torch.uniform(0,angleMax)
	print("==> Angle", angle)
	rotMatrix = rotationMatrix(angle, spacing, sliceSize)
	print("==> RotationMatrix")
	print(rotMatrix)

	--Coords
	x,y,z = torch.linspace(1,xSize,xSize), torch.linspace(1,ySize,ySize), torch.linspace(1,zSize,zSize) -- Old not centered coords
	zz = z:repeatTensor(ySize*xSize)
	yy = y:repeatTensor(zSize):sort():long():repeatTensor(xSize):double()
	xx = x:repeatTensor(zSize*ySize):sort():long():double()
	ones = torch.ones(totSize)

	coords = torch.cat({xx:reshape(totSize,1),yy:reshape(totSize,1),zz:reshape(totSize,1),ones:reshape(totSize,1)},2)
	-- Translate coords to be about the origin i.e. mean subtract
	translate = torch.ones(totSize,4):fill(-sliceSize)
	coords = coords + translate

	--Rotated coords
	newCoords = coords*rotMatrix:transpose(1,2)
	newCoords = newCoords*spacing -- Using spacing information to transform back to the "real world"

	-- Translate coords back to original coordinate system
	newCoords = newCoords + torch.ones(totSize,3):fill(sliceSize) 
	newCoords1 = newCoords:clone()
	minMax = torch.min(torch.Tensor{xSize,ySize,zSize})

	--newCoords[newCoords:lt(2)] = 2
	--newCoords[newCoords:gt(minMax-1)] = minMax - 1


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
	ijk = newCoords - xyz

	xyz[xyz:lt(1)] = 1
	xyz[{{},{1}}][xyz[{{},{1}}]:gt(xSize-1)] = xSize -1
	xyz[{{},{2}}][xyz[{{},{2}}]:gt(ySize-1)] = ySize -1
	xyz[{{},{3}}][xyz[{{},{3}}]:gt(zSize-1)] = zSize -1

	x1y1z1 = xyz + ooo
	x1yz = xyz + ozz
	xy1z = xyz + zoz
	xyz1 = xyz + zzo
	x1y1z = xyz + ooz
	x1yz1 = xyz + ozo
	xy1z1 = xyz + zoo

	-- Subtract the new coordinates from the 8 corners to get our distances ijk which are our weights
	i,j,k = ijk[{{},{1}}], ijk[{{},{2}}], ijk[{{},{3}}]
	i1j1k1 = ooo - ijk -- (1-i)(1-j)(1-k)
	i1,j1,k1 = i1j1k1[{{},{1}}], i1j1k1[{{},{2}}], i1j1k1[{{},{3}}] 

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

	function imgInterpolate()  
		Wfxyz =	  torch.cmul(i1,j1):cmul(k1)
		Wfx1yz =  torch.cmul(i,j1):cmul(k1)
		Wfxy1z =  torch.cmul(i1,j):cmul(k1)
		Wfxyz1 =  torch.cmul(i1,j1):cmul(k)
		Wfx1y1z = torch.cmul(i,j):cmul(k1)
		Wfx1yz1 = torch.cmul(i,j1):cmul(k)
		Wfxy1z1 = torch.cmul(i1,j):cmul(k)
		Wfx1y1z1 =torch.cmul(i,j):cmul(k)
		--weightedSum = Wfxyz + Wfx1yz + Wfxy1z + Wfxyz1 + Wfx1y1z + Wfx1yz1 + Wfx1y1z + Wfxy1z1 + Wfx1y1z1
		weightedSum = torch.cmul(Wfxyz,fxyz) + torch.cmul(Wfx1yz,fx1yz) + 
			      torch.cmul(Wfxy1z,fxy1z) + torch.cmul(Wfxyz1,fxyz1) +
			      torch.cmul(Wfx1y1z,fx1y1z) + torch.cmul(Wfx1yz1,fx1yz1) +
			      torch.cmul(Wfxy1z1,fxy1z1) + torch.cmul(Wfx1y1z1,fx1y1z1)
		return weightedSum
	end


	imgInterpolate = imgInterpolate():reshape(xSize,ySize,zSize)
	return imgInterpolate
end


-- Display Image
function displayImage()

	--Initialize displays
	if displayTrue==nil then
		zoom = 0.3
		init = image.lena()
		imgOriginal = image.display{image=init, zoom=zoom, offscreen=false}
		imgDis = image.display{image=init, zoom=zoom, offscreen=false}
		imgInterpolateDis= image.display{image=init, zoom=zoom, offscreen=false}
		displayTrue = "Display initialized"
	end

	loadImgTimer = torch.Timer()
	angleMax = 0.1
	sliceSize = 32 
	obs = annotationImg.new(torch.random(90))
	img, imgSub = obs.loadImg(sliceSize)
	spacing = obs.spacing
	
	print("Time elapsed for loading image of size "..sliceSize .. " = " .. loadImgTimer:time().real .. " seconds.")
	imgInterpolate = rotation3d(imgSub, angleMax, spacing, sliceSize)

	--Display images in predefined windows
	image.display{image = imgSub[sliceSize], win = imgDis}
	image.display{image = imgInterpolate[sliceSize], win = imgInterpolateDis}
end




