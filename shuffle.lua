require "image"
require "paths"
require "cunn"
dofile("3dInterpolation3.lua")

shuffle = {}

--Function to get batch from array given a from and to index
function shuffle.getBatch(array,from,to)

    local batch = {}
    local obs = 1
    for i = from, to do
	    batch[obs] = array[i]
	    obs = obs + 1
    end
    return batch 
end

--Function to shuffle array 
function shuffle.shuffle(array)
    math.randomseed(os.time())

    for i = 1, #array*2 do
	 local a = math.random(#array)
	 local b = math.random(#array)
	 array[a],array[b] = array[b],array[a]
    end
    return array
end
    
-- Function to split array into two given proportion (eg. into train test)
function shuffle.split(array,splitProportion)

    local splitPoint = torch.floor(splitProportion*#array)

    -- First array
    local first = {}
    for i=1, splitPoint do
        first[i] = array[i]
    end

    -- Second array
    local second = {}
    local obs = 1
    for i = splitPoint+1, #array do
        second[obs] = array[i]
        obs = obs + 1
    end

    return first, second -- Return the two arrays in order
end

return shuffle



















