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
parser.add_argument('-v', '--verbose', action='store_true', help='show debug messages', default=False)
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
out_edge1 = []
out_edge2 = []
out_edgeColor1 = []
out_edgeColor2 = []

lheInit = pylhe.readLHEInit(args.input)
## Find out which is the unit weight
proc2Weight0 = {}
for procInfo in lheInit['procInfo']:
    procId = int(procInfo['procId'])
    proc2Weight0[procId] = procInfo['unitWeight']

lheEvents = pylhe.readLHEWithAttributes(args.input)
#lheEvents = pylhe.readLHE(args.input)
iEvent = 0
for event in lheEvents:
    iEvent += 1
    if args.verbose: print("Processing %d'th events" % (iEvent), end='\r')
    ## Extract event weights and scale them
    procId = int(event.eventinfo.pid)
    weight0 = proc2Weight0[procId]

    weight = event.eventinfo.weight/weight0
    weights = [w/weight0 for w in event.weights.values()]

    out_weight.append(weight)
    out_weights.append(weights)

    ## Extract particle feature variables
    n = int(event.eventinfo.nparticles)

    node_featuresF = getNodeFeatures(event.particles, *(out_node_featureNamesF))
    node_featuresI = getNodeFeatures(event.particles, *(out_node_featureNamesI))
    for i, values in enumerate(node_featuresF):
        out_node_featuresF[i].append(values)
    for i, values in enumerate(node_featuresI):
        out_node_featuresI[i].append(values)

    ## Build particle decay tree
    mothers1, mothers2 = getNodeFeatures(event.particles, 'mother1', 'mother2')

    out_edge1 = []
    out_edge2 = []
    for i, (m1, m2) in enumerate(zip(mothers1, mothers2)):
        m1, m2 = int(m1), int(m2)
        if m1 == 0 or m2 == 0: continue
        for m in range(m1, m2+1):
            out_edge1.append(i)
            out_edge2.append(m-1)
    out_edge1 = np.array(out_edge1, dtype=np.int32)
    out_edge2 = np.array(out_edge2, dtype=np.int32)

    ## Build color connection
    colors1, colors2 = getNodeFeatures(event.particles, 'color1', 'color2')

    out_edgeColor1 = []
    out_edgeColor2 = []
    for i, color1 in enumerate(colors1):
        if color1 == 0: continue
        for j in np.where(colors2 == color1)[0]:
            out_edgeColor1.append(i)
            out_edgeColor2.append(j)
if args.verbose: print("\nProcessing done.")

if args.verbose: print("Merging output and saving into", args.output)
## Merge output objects
out_weight = np.array(out_weight)
out_weights = np.stack(out_weights)

## Save output
with h5py.File(args.output, 'w', libver='latest') as fout:
    dtypeFA = h5py.special_dtype(vlen=np.dtype('float64'))
    dtypeIA = h5py.special_dtype(vlen=np.dtype('int32'))

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
    fout_graphs.create_dataset('edge1', (nEvents,), dtype=dtypeIA)
    fout_graphs.create_dataset('edge2', (nEvents,), dtype=dtypeIA)
    fout_graphs['edge1'][...] = out_edge1
    fout_graphs['edge2'][...] = out_edge2

    fout_graphs.create_dataset('edgeColor1', (nEvents,), dtype=dtypeIA)
    fout_graphs.create_dataset('edgeColor2', (nEvents,), dtype=dtypeIA)
    fout_graphs['edgeColor1'][...] = out_edgeColor1
    fout_graphs['edgeColor2'][...] = out_edgeColor2
if args.verbose: print("done.")
