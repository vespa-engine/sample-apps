import json

tensor = []
for i in range(0,100):
	for j in range(0,384):
		cell = { 
			"address": { "x": str(j), "w": str(i) },
			 "value": 1.0 
		}
		tensor.append(cell)

weights = {
	"cells": tensor	
}
print(json.dumps(weights))
