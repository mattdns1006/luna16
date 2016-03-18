require "cunn"
Threads = require 'threads'
Threads.serialization('threads.sharedserialize')
nThreads = 3 
opt = {}
do
	local options = opt -- make an upvalue to serialize over to donkey threads
	donkeys = Threads(
		nThreads,
		function()
			require 'torch'
		end,
		function(idx)
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
for i = 1, 5 do
	donkeys:addjob(function()
				x,y = torch.random(), torch.uniform()
				print(x,y)
				return x,y
			end,
			function(x,y)
				X = x
				Y = y
			end
			)
	print(X,y)
	--print(time:time().real)
	time:reset()
end
--print(time2:time().real)
