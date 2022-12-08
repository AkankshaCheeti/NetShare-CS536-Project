''' This module contains the class necessary to implement CountMinSketch

@author: Peter Xenopoulos
@website: www.peterxeno.com
'''

from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
import sys, configparser, json, random, copy, math, os, pickle
import argparse
import socket
import mmh3

random.seed(42)

class CountMinSketch(object):
    ''' Class for a CountMinSketch data structure
    '''
    def __init__(self, width, depth):
        ''' Method to initialize the data structure
        @param width int: Width of the table
        @param depth int: Depth of the table (num of hash func)
        '''
        self.width = width  #cols
        self.depth = depth  #rows
        self.table = np.zeros([depth, width])  # Create empty table
        self.seed = np.random.randint(width, size=depth) # np.random.randint(w, size = d) // create some seeds

    def increment(self, key):
        ''' Method to add a key to the CMS
        @param key str: A string to add to the CMS
        '''
        for i in range(0, self.depth):
            index = mmh3.hash(key, self.seed[i]) % self.width
            self.table[i, index] = self.table[i, index]+1

    def estimate(self, key):
        ''' Method to estimate if a key is in a CMS
        @param key str: A string to check
        '''
        min_est = self.width
        for i in range(0, self.depth):
            index = mmh3.hash(key, self.seed[i]) % self.width
            if self.table[i, index] < min_est:
                min_est = self.table[i, index]
        return min_est

    def merge(self, new_cms):
        ''' Method to combine two count min sketches
        @param new_cms CountMinSketch: Another CMS object
        '''
        return self.table + new_cms        
