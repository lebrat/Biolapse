import numpy as np
import re

def rgb_to_gray(rgb):
    return np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 

def stringSplitByNumbers(x):
    r = re.compile('(\d+)')
    l = r.split(x)
    return [int(y) if y.isdigit() else y for y in l]


