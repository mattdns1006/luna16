require "torch"
require "cunn"
require "cutorch"

img = torch.linspace(1,4*4*4,64):reshape(4,4,4)

-- Functions to return f(x,y,z) given xyz
local function flattenIndices(sp_indices, shape)
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

local function getElements(tensor, sp_indices)
	sp_indices = sp_indices:long()
	flat_indices = flattenIndices(sp_indices, tensor:size()) 
	flat_tensor = tensor:view(-1)
	return flat_tensor:index(1, flat_indices)
end
