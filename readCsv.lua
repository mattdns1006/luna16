require "paths"
annotationPath = "CSVFILES/annotationsFullPaths.csv"

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
	annotationTable["x"] = tonumber(annotation[2])
	annotationTable["y"] = tonumber(annotation[3])
	annotationTable["z"] = tonumber(annotation[4])
	annotationTable["diameter"] = tonumber(annotation[5])

	return annotationTable 
end


