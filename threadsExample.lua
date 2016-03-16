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
			opt = options -- pass to all donkeys via upvalue
			--loadData = require "loadData"
			--loadData.Init()
			print("initialization")
		end
		)
end

X = {}
Y = {}
donkeys:synchronize()

time = torch.Timer()
time2 = torch.Timer()
for i = 1, 10 do
	donkeys:addjob(function()
				--x,y = loadData.getBatch(C0,C1,1,36,-1200,1200,0.4,0.3,0,params.para)
				x,y = torch.random(), torch.uniform()
				print(x,y)
				return x,y
			end,
			function(x,y)
				X = x
				Y = y
			end
			)
	print(X)
	print(time:time().real)
	time:reset()
end
print(time2:time().real)
