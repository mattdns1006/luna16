function binaryAccuracy(targets,predictions,cuda)
	local nObs = targets:size()[1]
	local tarOnes = torch.cmul(targets,predictions)
	local one = torch.ones(nObs)
	if cuda == 1 then 
		one = one:cuda()
	end
	local tarZeros = torch.cmul(one - targets, one -predictions)
	local accuracies = tarZeros + tarOnes
	return accuracies:mean()
end
