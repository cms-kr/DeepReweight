#!/usr/bin/env python
import argparse
import h5py
import pylhe
import numpy as np

def getNodeFeature(p, *featureNames):
    return tuple([getattr(p, x) for x in featureNames])
getNodeFeatures = np.vectorize(getNodeFeature)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', action='store', type=str, help='input file name', required=True)
parser.add_argument('-o', '--output', action='store', type=str, help='output file name', required=True)
args = parser.parse_args()

if not args.output.endswith('.h5'): outPrefix, outSuffix = args.output+'/data', '.h5'
else: outPrefix, outSuffix = args.output.rsplit('.', 1)

## Hold list of variables, by type
out_weight = []
out_weights = []
out_node_featureNamesF = ['px', 'py', 'pz', 'm']
out_node_featuresF = [[], [], [], []]
out_node_featureNamesI = ['id', 'status', 'spin']
out_node_featuresI = [[], [], []]
out_edgeIdxs1 = []
out_edgeIdxs2 = []
out_edgeByColorIdxs1 = []
out_edgeByColorIdxs2 = []

lheInit = pylhe.readLHEInit(args.input)
## Find out which is the unit weight
proc2Weight0 = {}
for procInfo in lheInit['procInfo']:
    procId = int(procInfo['procId'])
    proc2Weight0[procId] = procInfo['unitWeight']

lheEvents = pylhe.readLHEWithAttributes(args.input)
#lheEvents = pylhe.readLHE(args.input)
for event in lheEvents:
    ## Extract event weights and scale them
    procId = int(event.eventinfo.pid)
    weight0 = proc2Weight0[procId]

    weight = event.eventinfo.weight/weight0
    weights = [w/weight0 for w in event.weights.values()]

    out_weight.append(weight)
    out_weights.append(weights)

    ## Build particle decay tree
    n = int(event.eventinfo.nparticles)

    node_featuresF = getNodeFeatures(event.particles, *(out_node_featureNamesF))
    node_featuresI = getNodeFeatures(event.particles, *(out_node_featureNamesI))
    for i, values in enumerate(node_featuresF):
        out_node_featuresF[i].append(values)
    for i, values in enumerate(node_featuresI):
        out_node_featuresI[i].append(values)

    node_edges = getNodeFeatures(event.particles, 'mother1', 'mother2')
    #edge_index = np.zeros([2,n], dtype=np.long)
    #for i, p in enumerate(event.particles):
    #    print(p.id)

## Merge output objects
out_weight = np.array(out_weight)
out_weights = np.stack(out_weights)
#for i in range(len(out_node_featuresF)):
#    out_node_featuresF[i] = np.concatenate(out_node_featuresF[i])
#for i in range(len(out_node_featuresI)):
#    out_node_featuresI[i] = np.concatenate(out_node_featuresI[i])

## Save output
with h5py.File(args.output, 'w', libver='latest') as fout:
    dtypeFA = h5py.special_dtype(vlen=np.dtype('float64'))
    dtypeIA = h5py.special_dtype(vlen=np.dtype('uint32'))

    fout_events = fout.create_group("events")
    fout_events.create_dataset('weight', data=out_weight, dtype='f4')
    fout_events.create_dataset('weights', data=out_weights, dtype='f4')

    nEvents = len(out_weight)

    for name, features in zip(out_node_featureNamesF, out_node_featuresF):
        fout_events.create_dataset(name, (nEvents,), dtype=dtypeFA)
        fout_events[name][...] = features
    for name, features in zip(out_node_featureNamesI, out_node_featuresI):
        fout_events.create_dataset(name, (nEvents,), dtype=dtypeIA)
        fout_events[name][...] = features

    fout_graphs = fout.create_group('graphs')
    fout_graphs.create_dataset('edgeIdxs1', (nEvents,), dtype=dtypeIA)
    fout_graphs.create_dataset('edgeIdxs2', (nEvents,), dtype=dtypeIA)
    fout_graphs['edgeIdxs1'][...] = out_edgeIdxs1
    fout_graphs['edgeIdxs2'][...] = out_edgeIdxs2

    fout_graphs.create_dataset('edgeByColorIdxs1', (nEvents,), dtype=dtypeIA)
    fout_graphs.create_dataset('edgeByColorIdxs2', (nEvents,), dtype=dtypeIA)
    fout_graphs['edgeByColorIdxs1'][...] = out_edgeByColorIdxs1
    fout_graphs['edgeByColorIdxs2'][...] = out_edgeByColorIdxs2
