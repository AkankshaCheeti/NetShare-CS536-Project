''' This module contains the class necessary to implement CountMinSketch

@author: Peter Xenopoulos
@website: www.peterxeno.com
'''

import numpy as np
import mmh3
import pandas as pd
import sys, configparser, json, random, copy, math, os, pickle
import socket
random.seed(42)

class CountMinSketch(object):
    ''' Class for a CountMinSketch data structure
    '''
    def __init__(self, width, depth, seeds):
        ''' Method to initialize the data structure
        @param width int: Width of the table
        @param depth int: Depth of the table (num of hash func)
        @param seeds list: Random seed list
        '''
        self.width = width  #cols
        self.depth = depth  #rows
        self.table = np.zeros([depth, width])  # Create empty table
        self.seed = seeds # np.random.randint(w, size = d) // create some seeds

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

if __name__ == '__main__':
    param_w = 10000
    param_d = 5
    seeds = np.random.randint(param_w, size = param_d)
    cms = CountMinSketch(10000, 5, seeds = seeds)

    # read raw data
    dataset = "eval/caida"
    raw_df = pd.read_csv(os.path.join(dataset, "raw.csv"))
    for dstip in raw_df.dstip:
        cms.increment(dstip.to_bytes(4, byteorder = 'big'))
        if dstip == raw_df.dstip[2]:
            ip = dstip


    print(ip)
    print(cms.estimate(ip.to_bytes(4, byteorder = 'big')))

    count = 0
    for dstip in raw_df.dstip:
        if dstip == raw_df.dstip[2]:
            count = count + 1
    print(count)
