require "cunn"
Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local data = {}
local result = {}
local unpack = unpack and unpack or table.unpack

function data.new(nThreads)
	local self = {}
	self.threads = Threads(nThreads,
			   function()
				   loadData = require "loadData"
				   loadData.Init()
		           end
			)
	self.threads:synchronize()
	for i=1, nThreads do
		self.threads:addjob(self._getFromThreads,
				    self._pushResult)
	end
	return self
end

function data._getFromThreads()
	      x,y = loadData.getBatch(C0,C1,1,36,-1200,1200,0.4,0.3,0,params.para)
	      return x,y
end

function data._pushResult(...)
	local res = {...}
	if res == nil then 
		self.threads:synchronize()
	end
	result[1] = res
end

function data:getBatch()
	self.threads:addjob(self._getFromThreads, self._pushResult)
	self.threads:dojob()
	local res = result[1]
		result[1] = nil
	if torch.type(res) == 'table' then
		return unpack(res)
	end
		print(type(res))
	return res
end

d = data.new(2)




--[[
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
]]--

