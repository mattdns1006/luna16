dofile("imageCandidates.lua")
require "image"
require "torch"

--[[
cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Options')
cmd:option('-display',0,'Display images?')
cmd:option('-displayZoom',0.4,'Image zoom size')
cmd:option('-runEg',0,'Run example')
cmd:option('-sliceSize',40,'Slicesize')
cmd:option('-clipMin',-800,'Clipmin')
cmd:option('-clipMax',500,'Clipmin')
cmd:option('-angleMax',0.8,'angleMax')
cmd:option('-scalingFactor',0.8,'Scaling Factor')
cmd:text()

parameters = cmd:parse(arg)
parameters.rundir = cmd:string('parameters', parameters, {dir=true})
]]--

function rotationMatrix(angle)  --Returns a 3D rotation matrix
	local rotMatrix = torch.Tensor{1,0,0,0,torch.cos(angle),-torch.sin(angle),0,torch.sin(angle),torch.cos(angle)}:reshape(3,3)
	return rotMatrix
end

function rotation3d(imgObject, angleMax, sliceSize, clipMin, clipMax, scalingFactor) -- Returns 3d interpolated image

	local imgOriginal = imgObject:loadImg(clipMin,clipMax,sliceSize)
	
	local sliceSize_2 = torch.floor(sliceSize/2)
	local xSize,ySize,zSize = sliceSize, sliceSize, sliceSize -- Making a cube hence all dimensions are the same (possibly change)
	local totSize = xSize*ySize*zSize

	--Coords
	local x,y,z = torch.linspace(1,xSize,xSize), torch.linspace(1,ySize,ySize), torch.linspace(1,zSize,zSize) -- Old not centered coords
	local zz = z:repeatTensor(ySize*xSize)
	local yy = y:repeatTensor(zSize):sort():long():repeatTensor(xSize):double()
	local xx = x:repeatTensor(zSize*ySize):sort():long():double()
	local coords = torch.cat({xx:reshape(totSize,1),yy:reshape(totSize,1),zz:reshape(totSize,1)},2)
-- Translate coords to be about the origin i.e. mean subtract
	local translate = torch.ones(totSize,3):fill(-sliceSize/2)
	coords:add(translate)

	--Timer
	--timeToLoad = torch.Timer()
	--print("Time taken here".. timeToLoad:time().real)

	-- Rotated coords
	-- Rotation matrix
	local angle = torch.uniform(-angleMax,angleMax)
	local rotMatrix = rotationMatrix(angle)

	-- Rotation
	local newCoords = coords*rotMatrix:transpose(1,2)

	--Spacing & scaling
	local spacing  = torch.diag(torch.Tensor{1/imgObject.zSpacing, 1/imgObject.ySpacing, 1/imgObject.xSpacing})
	local sf = torch.normal(scalingFactor,0.01)
	newCoords = newCoords*spacing*sf -- Using spacing information to transform back to the "real world"

	-- Translate coords back to original coordinate system where the centre is on the nodule
	local noduleZ, noduleY, noduleX  = imgObject.z, imgObject.y, imgObject.x
	local noduleTranslate = torch.ones(totSize,3)
	noduleTranslate = torch.ones(totSize,3)
	noduleTranslate[{{},{1}}]:fill(noduleZ)
	noduleTranslate[{{},{2}}]:fill(noduleY)
	noduleTranslate[{{},{3}}]:fill(noduleX)

	local newCoordsT = newCoords + noduleTranslate -- These coords are now in the original space
	local newCoords1 = newCoordsT:clone()

	-- Need all 8 corners of the cube in which newCoords[i,j,k] lies
	local zzz = torch.zeros(totSize,3)
	local ooo = torch.ones(totSize,3)

	local function fillzo(zzz_ooo,z_o,column)
		zzz_ooo_clone = zzz_ooo:clone()
		zzz_ooo_clone:select(2,column):fill(z_o)
		--return zzz_ooo_clone
		return zzz_ooo_clone
	end

	local ozz = fillzo(zzz,1,1)
	local zoz = fillzo(zzz,1,2)
	local zzo = fillzo(zzz,1,3)
	local zoo = fillzo(ooo,0,1)
	local ozo = fillzo(ooo,0,2)
	local ooz = fillzo(ooo,0,3)

	local xyz = newCoords1:clone()
	xyz:floor()
	local ijk = newCoords1 - xyz

	xyz[{{},{1}}]:clamp(1,imgOriginal:size()[1]-1)
	xyz[{{},{2}}]:clamp(1,imgOriginal:size()[2]-1)
	xyz[{{},{3}}]:clamp(1,imgOriginal:size()[3]-1)

	local x1y1z1 = xyz + ooo
	local x1yz = xyz + ozz
	local xy1z = xyz + zoz
	local xyz1 = xyz + zzo
	local x1y1z = xyz + ooz
	local x1yz1 = xyz + ozo
	local xy1z1 = xyz + zoo

	-- Subtract the new coordinates from the 8 corners to get our distances ijk which are our weights
	--
	local i,j,k = ijk[{{},{1}}], ijk[{{},{2}}], ijk[{{},{3}}]
	local i1j1k1 = ooo - ijk -- (1-i)(1-j)(1-k)
	local i1,j1,k1 = i1j1k1[{{},{1}}], i1j1k1[{{},{2}}], i1j1k1[{{},{3}}] 

	-- Functions to return f(x,y,z) given xyz
	function flattenIndices(sp_indices, shape)
		sp_indices = sp_indices - 1
		n_elem, n_dim = sp_indices:size(1), sp_indices:size(2)
		flat_ind = torch.LongTensor(n_elem):fill(1)

		mult = 1
		for d = n_dim, 1, -1 do
			flat_ind:add(sp_indices[{{}, d}] * mult)
			mult = mult * shape[d]
		end
		return flat_ind
	end

	function getElements(tensor, sp_indices)
		sp_indices = sp_indices:long()
		flat_indices = flattenIndices(sp_indices, tensor:size()) 
		flat_tensor = tensor:view(-1):double()
		--flat_tensor = tensor:view(-1)
		return flat_tensor:index(1, flat_indices)
	end

	 fxyz = getElements(imgOriginal,xyz)
	 fx1yz = getElements(imgOriginal,x1yz)
	 fxy1z = getElements(imgOriginal,xy1z)
	 fxyz1 = getElements(imgOriginal,xyz1)
	 fx1y1z = getElements(imgOriginal,x1y1z)
	 fx1yz1 = getElements(imgOriginal,x1yz1)
	 fxy1z1 = getElements(imgOriginal,xy1z1)
	 fx1y1z1 = getElements(imgOriginal,x1y1z1)


	 Wfxyz =   torch.cmul(i1,j1):cmul(k1)
	 Wfx1yz =  torch.cmul(i,j1):cmul(k1)
	 Wfxy1z =  torch.cmul(i1,j):cmul(k1)
	 Wfxyz1 =  torch.cmul(i1,j1):cmul(k)
	 Wfx1y1z = torch.cmul(i,j):cmul(k1)
	 Wfx1yz1 = torch.cmul(i,j1):cmul(k)
	 Wfxy1z1 = torch.cmul(i1,j):cmul(k)
	 Wfx1y1z1 =torch.cmul(i,j):cmul(k)
	 imgInterpolate = torch.cmul(Wfxyz,fxyz) + torch.cmul(Wfx1yz,fx1yz) + 
		      torch.cmul(Wfxy1z,fxy1z) + torch.cmul(Wfxyz1,fxyz1) +
		      torch.cmul(Wfx1y1z,fx1y1z) + torch.cmul(Wfx1yz1,fx1yz1) +
		      torch.cmul(Wfxy1z1,fxy1z1) + torch.cmul(Wfx1y1z1,fx1y1z1)

	imgInterpolate:resize(xSize,ySize,zSize)

	-- Mean remove and normalize between -1 +1 
	imgInterpolate:add(-imgInterpolate:mean())
	--[[
	local min = imgInterpolate:min()
	local range  = imgInterpolate:max()-min
	imgInterpolate:add(-min)
	imgInterpolate:mul(1.0/range)
	]]--

	return imgInterpolate
end

--Example/Tests
function eg3d()

	dofile("readCsv.lua")
	dofile("imageCandidates.lua")
	csv = csvToTable("CSVFILES/candidatesClass1.csv")
	local timeToInterpolate = {}
	local imgOriginalZoom = 1

	--Initialize displays
	if displayTrue==nil and parameters.display==1 then
		print("Initializing displays ==>")
		local init = image.lena()
		imgOrigX = image.display{image=init, zoom=imgOriginalZoom, offscreen=false}
		local displayZoom = parameters.displayZoom
		imgSubZ = image.display{image=init, zoom=displayZoom, offscreen=false}
		imgSubY = image.display{image=init, zoom=displayZoom, offscreen=false}
		imgSubX = image.display{image=init, zoom=displayZoom, offscreen=false}
		imgInterpolateDisZ = image.display{image=init, zoom=displayZoom, offscreen=false}
		imgInterpolateDisY = image.display{image=init, zoom=displayZoom, offscreen=false}
		imgInterpolateDisX = image.display{image=init, zoom=displayZoom, offscreen=false}
		displayTrue = "Display initialized"
	end

	

	for i=1,#csv do

		local observationNumber = torch.random(#csv)
		local obs = Candidate:new(csv,observationNumber)

		local n = 10
		local timeToInterpolateBatch = {}
		for j = 1,n do 
			local timerInterpolate = torch.Timer()
			local imgInterpolated = rotation3d(obs, parameters.angleMax, parameters.sliceSize, parameters.clipMin, parameters.clipMax, parameters.scalingFactor)
			timeToInterpolateBatch[#timeToInterpolateBatch + 1] = timerInterpolate:time().real -- average interpolation time of size n
			timerInterpolate:reset()
			local imgSub = imgOriginal:sub(obs.z-parameters.sliceSize/2, obs.z + parameters.sliceSize/2 - 1, obs.y-parameters.sliceSize/2, obs.y + parameters.sliceSize/2 - 1,obs.x-parameters.sliceSize/2, obs.x + parameters.sliceSize/2 - 1)

			if parameters.display == 1 then 
				image.display{image = imgOriginal[obs.z], win = imgOrigX}

				image.display{image = imgSub[1+parameters.sliceSize/2], win = imgSubZ}
				image.display{image = imgSub[{{},{1+parameters.sliceSize/2}}]:reshape(parameters.sliceSize,parameters.sliceSize), win = imgSubY}
				image.display{image = imgSub[{{},{},{1+parameters.sliceSize/2}}]:reshape(parameters.sliceSize,parameters.sliceSize), win = imgSubX}

				image.display{image = imgInterpolated[1+parameters.sliceSize/2]:double(), win = imgInterpolateDisZ}
				image.display{image = imgInterpolated[{{},{1+parameters.sliceSize/2}}]:reshape(parameters.sliceSize,parameters.sliceSize), win = imgInterpolateDisY}
				image.display{image = imgInterpolated[{{},{},{1+parameters.sliceSize/2}}]:reshape(parameters.sliceSize,parameters.sliceSize), win = imgInterpolateDisX}
			end
		end
		timeToInterpolate[#timeToInterpolate + 1] = torch.Tensor{timeToInterpolateBatch}:mean() -- Mean batch time
		print(timeToInterpolate)

		collectgarbage()
	end
end
--if parameters.runEg == 1 then eg3d() end

