"""
SipHash works well as a family of pairwise-independent hash functions for a CountMinSketch.
This implementation uses my Python cffi-bound version: https://github.com/zacharyvoase/python-csiphash
"""

import csiphash
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
import math, os
import argparse
import struct


def positions(n_rows, n_columns, event):
    def digest_for_row(row):
        digest = csiphash.siphash24(struct.pack('QQ', row, n_columns), event)
        return np.fromstring(digest, dtype=np.uint64, count=1)[0]
    digest_array = np.array(list(map(digest_for_row, range(n_rows))))
    bucket_array = np.true_divide(digest_array, np.uint64(2**64-1))
    column_array = np.trunc(n_columns * bucket_array).astype(np.uint64)
    # print(column_array)
    return column_array


class CountMinSketch(object):
    def __init__(self, d, w):
        self.d = d
        self.w = w
        self.array = np.zeros((d, w), dtype=np.uint64)

    @classmethod
    def from_error(cls, epsilon, delta):
        """Choose d and w based on desired error bound and probability."""
        d = int(math.ceil(math.log(1.0/delta)))
        w = int(math.ceil(math.e / epsilon))
        return cls(d, w)

    def increment(self, event):
        for i, position in enumerate(positions(self.d, self.w, event)):
            self.array[i][position] += np.uint64(1)

    def estimate(self, event):
        columns = positions(self.d, self.w, event)
        return min(self.array[row][col] for row, col in enumerate(columns))

