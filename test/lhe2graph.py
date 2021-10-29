#!/usr/bin/env python
import argparse
import h5py
import pylhe
import numpy as np

def getNodeFeature(p, *featureNames):
    return tuple([getattr(p, x) for x in featureNames])
getNodeFeatures = np.vectorize(getNodeFeature)

parser = argparse.ArgumentParser()
parser.add_argument('input', nargs='+', action='store', type=str, help='input file name')
parser.add_argument('-o', '--output', action='store', type=str, help='output directory name', required=True)
args = parser.parse_args()

if not args.output.endswith('.h5'): outPrefix, outSuffix = args.output+'/data', '.h5'
else: outPrefix, outSuffix = args.output.rsplit('.', 1)

fName = "events.lhe"
#fName = "pylhe-drell-yan-ll-lhe.gz"

lheInit = pylhe.readLHEInit(fName)
## Find out which is the unit weight
proc2Weight0 = {}
for procInfo in lheInit['procInfo']:
    procId = int(procInfo['procId'])
    proc2Weight0[procId] = procInfo['unitWeight']

lheEvents = pylhe.readLHEWithAttributes(fName)
#lheEvents = pylhe.readLHE(fName)
for event in lheEvents:
    ## Extract event weights and scale them
    procId = int(event.eventinfo.pid)
    weight0 = proc2Weight0[procId]

    weight = event.eventinfo.weight/weight0
    weights = [w/weight0 for w in event.weights.values()]

    ## Build particle decay tree
    n = int(event.eventinfo.nparticles)

    node_momentum = np.stack(getNodeFeatures(event.particles, 'px', 'py', 'pz', 'm'), axis=0)
    node_features = np.stack(getNodeFeatures(event.particles, 'id', 'status', 'spin'), axis=0)

    #edge_index = np.zeros([2,n], dtype=np.long)
    #for i, p in enumerate(event.particles):
    #    print(p.id)
