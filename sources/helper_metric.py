import numpy as np
import math

def MAE(s1, s2):
    if len(s1)!= len(s2):
        print("ERROR in MAE: S1 $\neq$ S2.")
        return -1
    
    res = 0
    for x,y in zip(s1,s2):
        res = res +  math.fabs(x-y)
    
    res = res / len(s1)

    return res


def MSE(s1, s2):
    if len(s1)!= len(s2):
        print("ERROR in MAE: S1 $\neq$ S2.")
        return -1
    
    res = 0
    for x,y in zip(s1,s2):
        res = res +  (x-y)*(x-y)
    
    res = res / len(s1)

    return res
