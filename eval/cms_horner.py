#
# Copyright 2020 MZ (mohammedzu@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# Rabbi Zidni Ilma
# "Simple Count Min Sketch in Pythonic Art"
# - Part II of the 2020 Weekend coding series on Probabilistic Data Structures.
# - Count Min Sketch is a space efficient, constant time Probabilistic data structure that
# -     guarantees zero under-counting and tunable probability over-counting of events in a stream.
# - This example use case illustrates counting the views for movies in a click stream.
# MZ: mohammedzu@gmail.com
# MZ: 20200119 - Initial
#

from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
import sys, configparser, json, random, copy, math, os, pickle
import argparse
import socket

import random
import math
from collections import Counter


class CountMinSketch:

    def __init__(self, w, d=3):
        self.size = w # self.nearest_prime(m)
        self.min_sketch = [[0] * self.size] * d
        self.hash_fns = d
        self.salts = random.sample(range(1, self.size), self.hash_fns)
        # print(f'Salts: {self.salts}')

    def __len__(self):
        return self.size * self.hash_fns

    def increment(self, key):
        for _i, hash_i in enumerate(self.get_hashes(key)):
            self.min_sketch[_i][hash_i] += 1
        return self

    def estimate(self, key):
        _all_count = [self.min_sketch[_i][hash_i]
                      for _i, hash_i in enumerate(self.get_hashes(key))]
        return min(_all_count) #, _all_count

    def get_hashes(self, key):
        return [self.hash_fn(key=key, size=self.size,
                             salt=self.salts[i])
                for i in range(self.hash_fns)]

    @classmethod
    def hash_fn(cls, key, size=1009, salt=1):
        """
        Simple hashing using Horner's rule
        """
        hx = int()
        for c in str(key):
            hx = ((hx * 31) + ord(c) + salt) % size
        return hx

    @classmethod
    def nearest_prime(cls, n):
        for num in range(n + 1 if n % 2 == 0 else n, (2 * n) + 1, 2):
            for divisor in range(3, math.ceil(math.sqrt(num)), 2):
                if num % divisor == 0:
                    break
            else:
                return num
        return n

