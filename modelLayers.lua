require "torch"
require "nn"

modelLayers = {}


--   VolumetricConvolution(nInputPlane, nOutputPlane, kT, kW, kH [, dT, dW, dH, padT, padW, padH])
function modelLayers.add3dconv(model,layerNu,nfilters,filtersize,stride,padding)
	model:add(nn.VolumetricConvolution(
					  nFilters[layerNu-1],
					  nFilters[layerNu-1],
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
	model:add(nn.VolumetricBatchNormalization(nFilters[layer]))
end

return modelLayers

