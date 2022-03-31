"""
@author: acp21ht
"""

import numpy as np
import numba
import matplotlib.pyplot as plt

class Environment:
    def __init__(self,shape=[1000,1000]):
        """
        Create the environment: default size = shape, shape[0] * shape[1]
        """
        self.shape = shape
    
    @numba.jit     
    def check_position(self, position):
        """
        returns whether the position is within the environment
        """
        position[:] = np.round(position)
        if position[0] < 0 or position[1] < 0: return False
        if position[0] > self.shape[0] - 1 or position[1] > self.shape[1] - 1 : return False
        
        return True
    
    
    def get_random_location(self):
        return np.random.randint([0,0], self.shape)