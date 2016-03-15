
require "cunn"
threads = require 'threads'
threads.serialization('threads.sharedserialize')
nthread = 4 
njob = 5 

Queue = require 'threads.queue'

donkeys = threads.Threads(
   nthread,
   function()
	   loadData = require "loadData"
	   loadData.Init()
   end
)

X = {}
y = {} 
for i=1, 10 do
	donkeys:addjob(function(idx)
				x,y = loadData.getBatch(C0,C1,1,36,-1200,1200,0.4,0.3,0,params.para)
				print("x",type(x))
				return x,y
	   		end,
		      function(x,y) 
			 X = x
			 y = y 
			end)
end
donkeys:synchronize()

