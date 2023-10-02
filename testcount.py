# list =[[['M'],[100,300],[100,100],[1000,100],[1000,300],['z']]]
#remove first index and last index 
import numpy as np
import settings 
MOCK_LIST_POLYGON = settings.MOCK_LIST_POLYGON
#remove last index in list and inside list in list remove first index
list = [[i for i in l[:-1]][1:]  for l in MOCK_LIST_POLYGON ]
print(list)