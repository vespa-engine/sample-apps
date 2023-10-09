# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
#! /usr/bin/env python3

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
