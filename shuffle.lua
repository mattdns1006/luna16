require "image"
require "paths"
require "cunn"
dofile("3dInterpolation3.lua")

shuffle = {}

function shuffle.tableLength(T)
  local count = 0
    for _ in pairs(T) do count = count + 1 end
      return count
end

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
    newTable = {}
    nObs = shuffle.tableLength(array)
    for i = 1, nObs do
         nObs = shuffle.tableLength(array)
	 a = math.random(nObs)
	 j = 1
	 for k,v in pairs(array) do
		 if j == a then
			 newTable[k] = v
			 print("new")
			 print(newTable)
			 array[k] = nil -- remove from original array
			 print("old")
			 print(array)
		 else
			 j = j + 1
		 end
	end
    end
    return newTable 
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

function shuffle.table(table)
	shuffledTable = {}
	for k, v in pairs(table) do
	end
end

eg = {}
eg["hi"] = "hello"
eg["bi"] = "bye"
eg["cool"] = "cool"
eg["whats up"] = "hows it going"
eg["foo"] = "bar"

return shuffle



















