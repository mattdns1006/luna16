require "paths"
annotationPath = "CSVFILES/annotationsNVC.csv"

function csvToTable(path)
	local csvFile = io.open(path,"r")
	local header = csvFile:read()
	local data = {}

	local i = 0  
	for line in csvFile:lines('*l') do  
		i = i + 1
		data[i] = line 
	end
	return data
end

function getAnnotationXYZ(obs)
	local annotations = csvToTable(annotationPath)
	local annotation = annotations[obs]:split(",")

	local annotationTable = {}
	-- Get elements of an observation
	annotationTable["path"] = annotation[6] 
	local noduleCoords = {}
	noduleCoords["x"] = tonumber(annotation[7])
	noduleCoords["y"] = tonumber(annotation[8])
	noduleCoords["z"] = tonumber(annotation[9])
	annotationTable["noduleCoords"] = noduleCoords
	annotationTable["diameter"] = tonumber(annotation[5])

	return annotationTable 
end


