require "torch"
require "nn"

modelLayers = {}

--   VolumetricConvolution(nInputPlane, nOutputPlane, kT, kW, kH [, dT, dW, dH, padT, padW, padH])
function modelLayers.add3DConv(model,layerNu,nFilters,filterSize,stride,padding,nFiltersSame)
	if nFiltersSame == 1 then -- Are we doing back to back convolutions? I.e. keeping nFeatures same
		nFiltersPrevious = nFilters[layerNu]
	else
		nFiltersPrevious = nFilters[layerNu-1]
	end
	model:add(nn.VolumetricConvolution(
					  --nFilters[layerNu-1],
					  nFiltersPrevious,
					  nFilters[layerNu],
					  filterSize[layerNu],
					  filterSize[layerNu],
					  filterSize[layerNu],
					  stride[layerNu],
					  stride[layerNu],
					  stride[layerNu],
					  padding[layerNu],
					  padding[layerNu],
					  padding[layerNu]))
end

--   VolumetricMaxPooling(kT, kW, kH [, dT, dW, dH, padT, padW, padH])
function modelLayers.addMP(model,layerNu,size,stride,padding)
	model:add(nn.VolumetricMaxPooling(
					size[layerNu],
					size[layerNu],
					size[layerNu],
					stride[layerNu],
					stride[layerNu],
					stride[layerNu],
					padding[layerNu],
					padding[layerNu],
					padding[layerNu]
					))
end
					 
function modelLayers.addBN(model,layerNu,nfilters)
	model:add(nn.VolumetricBatchNormalization(nFilters[layerNu]))
end

return modelLayers

