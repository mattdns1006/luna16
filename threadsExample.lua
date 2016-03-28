require "cunn"
require "csvigo"
csv = csvigo.load({path = "/home/msmith/luna16/CSVFILES/candidates.csv",mode = "large"})
smallCsv = {}
n = 10
for j = 1,n do
	smallCsv[j] = csv[j]
end


Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

nThreads = 3 
opt = {}
opt.csv = smallCsv
do
	local csv = smallCsv
	local options = opt -- make an upvalue to serialize over to donkey threads
	local nThreads = nThreads
	donkeys = Threads(
		nThreads,
		function()
			require 'torch'
		end,
		function(idx)
			print(#csv)
			print(nThreads)
			seq = torch.range(idx,#csv,nThreads)
			print(seq)
			local seed = idx
			tid = idx
			torch.manualSeed(seed)
			print("initialiization")
			print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
		end
		)
end

X = {}
Y = {}
donkeys:synchronize()
time = torch.Timer()
time2 = torch.Timer()
for i = 1, 50 do
	donkeys:addjob(function()
				x,y = torch.random(), torch.uniform()
				return x,y
			end,
			function(x,y)
				X[#X +1] = {x,tid}
				Y[#Y +1] = {y,tid}
			end
			)
	--print(X,y)
	--print(time:time().real)
	time:reset()
end
--print(time2:time().real)
