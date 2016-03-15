require "cunn"
Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

DataThreads = {}
DataThreads.__index = DataThreads
local result = {}
local unpack = unpack and unpack or table.unpack

function DataThreads.new(nThreads,opt_)
	local self = {}
	opt_ = opt_ or {}
	self.threads = Threads(nThreads,
			   function()
				   print("background stuff")
				   --loadData = require "loadData"
				   --loadData.Init()
				   print("done")
		           end
			)
	self.threads:synchronize()
	for i=1, nThreads do
		self.threads:addjob(self._getFromThreads, self._pushResult)
	end
	return setmetatable(self,DataThreads)
end


function DataThreads._getFromThreads()
	      --x,y = loadData:getBatch(C0,C1,1,36,-1200,1200,0.4,0.3,0,params.para)
	      x,y = torch.uniform(),torch.uniform()
	      return x,y
end

function DataThreads._pushResult(...)
	local res = {...}
	if res == nil then 
		self.threads:synchronize()
	end
	result[1] = res
end


function DataThreads:getBatch()
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

d = DataThreads.new(4)


timer = torch.Timer()
overallTime = torch.Timer()
local n = 30
for i = 1, n do
	x,y = d:getBatch()
	print(timer:time().real)
	timer:reset()
end
print("Overall time for "..n.." = " .. overallTime:time().real)


--[[
loadData = require "loadData"
loadData.Init()
overallTime:reset()
for i = 1, n do
     x,y = loadData.getBatch(C0,C1,1,36,-1200,1200,0.4,0.3,0,params.para)
     print(timer:time().real)
     timer:reset()
end
print("Overall time for "..n.." = " .. overallTime:time().real)

]]--
