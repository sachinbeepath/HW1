#!/usr/bin/python

### import necessary packages

### PHYS 490
### Sachin Beepath 
### 20613204

import numpy as np
import matplotlib.pyplot as plt
import math
import json
import os
import sys

### read in files and load data
in_fname = sys.argv[1]
in_file = open(in_fname,'r')
data = in_file.readlines()

jfname = sys.argv[2]
jfile = open(jfname,'r')
jdata = json.load(jfile)
alpha = jdata['learning rate']
num_iter = jdata['num iter']

### Initialize arrays and load data into them
x = []
y = []

for i in data:
    line = i.split()
    z = []
    for j in range(len(line) - 1):
        z.append(float(line[j]))
    x.append(z)
    y.append(float(line[-1]))
x = np.array(x)
y = np.array(y)
### pad first column of array 
x = np.concatenate((np.ones((x.shape[0]))[:, np.newaxis], x), axis=1)

### calculate weight vector using analytic method
x_t = x.transpose()
inv = np.linalg.inv(x_t.dot(x))
invxt = inv.dot(x_t)
w = invxt.dot(y)

### calculate weight vector using GD method
w_j = np.ones(x.shape[1])
for i in range(num_iter+1):
    n = np.random.randint(0,x.shape[0])
    w_j += alpha * (y[n] - w_j.dot(x[n,:])) * x[n,:]
        
### write data to output file
out_fname =  (os.path.splitext(in_fname)[0]) + ".out"
with open(out_fname, 'w') as f:
    for item in w:
        f.write("%s\n" % '%.4f' %item)
    f.write("\n")
    for item in w_j:
        f.write("%s\n" % '%.4f' %item)

print("done")
