#!/usr/bin/env python
import h5py
import torch
from torch.utils.data import Dataset
from bisect import bisect_right
from glob import glob
import pandas as pd
import numpy as np

from torch_geometric.data import InMemoryDataset as PyGDataset, Data as PyGData
import math

class LHEGraphDataset(PyGDataset):
    def __init__(self, **kwargs):
        super(LHEGraphDataset, self).__init__()
        self.isLoaded = False
        self.fNames = []
        self.sampleInfo = pd.DataFrame()

        self.featNames = ["px", "py", "pz", "m"]
        self.nFeats = len(self.featNames)
        self.maxEventsList = np.zeros(0)

        self.edgeType = 0
        if 'edgeType' in kwargs:
            edgeTypeMap = {
                'none':0, 'all':1,
                'decay':2, 'color':3,
            }
            self.edgeType = edgeTypeMap[kwargs['edgeType']]

    def addSample(self, procName, fNamePattern, scale=1.):
        for fName in glob(fNamePattern):
            if not fName.endswith(".h5"): continue
            fileIdx = len(self.fNames)
            self.fNames.append(fName)

            info = {
                'procName':procName, 'scale':scale, 'nEvent':0,
                'label':0, ## default label for the classification problem, to be filled later
                'fileName':fName, 'fileIdx':fileIdx,
                #'sumW' : 0, ## Note: this sumW is total sumW over all files of the same process names
            }
            self.sampleInfo = self.sampleInfo.append(info, ignore_index=True)

    def setProcessLabel(self, procName, label):
        self.sampleInfo.loc[self.sampleInfo.procName==procName, 'label'] = label

    def load(self):
        if self.isLoaded: return

        self.files = []
        self.weightList = []
        self.featsList = []
        self.edge1List = []
        self.edge2List = []

        for i, fName in enumerate(self.sampleInfo['fileName']):
            f = h5py.File(fName, 'r', libver='latest', swmr=True)
            self.files.append(f)

            ## set weight
            weight = f['events/weight']
            nEvent = weight.shape[0]
            self.sampleInfo.loc[i, 'nEvent'] = nEvent
            self.weightList.append(weight)

            self.featsList.append([f['events'][x] for x in self.featNames])

            if self.edgeType == 2:
                self.edge1List.append(f['graphs/edge1'])
                self.edge2List.append(f['graphs/edge2'])
            if self.edgeType == 3:
                self.edge1List.append(f['graphs/edgeColor1'])
                self.edge2List.append(f['graphs/edgeColor2'])

        print(self.sampleInfo)

        ## Compute cumulative sums of nEvent, to be used for the file indexing
        self.maxEventsList = np.concatenate(([0.], np.cumsum(self.sampleInfo['nEvent'])))

        self.isLoaded = True

    def get(self, idx):
        if not self.isLoaded: self.load()

        fileIdx = bisect_right(self.maxEventsList, idx)-1
        offset = self.maxEventsList[fileIdx]
        idx = idx-int(offset)

        weight = self.weightList[fileIdx][idx]

        feats = [self.featsList[fileIdx][i][idx] for i in range(self.nFeats)]
        feats = torch.tensor(np.array(feats).T).float()

        if self.edgeType == 0:
            edge = torch.LongTensor([]).view(2,-1)
        elif self.edgeType == 1:
            ## FIXME: to be implemented for the fully connected graph
            edge = torch.LongTensor([]).view(2,-1)
        elif self.edgeType >= 2:
            edge1 = self.edge1List[fileIdx][idx]
            edge2 = self.edge2List[fileIdx][idx]
            edge = torch.LongTensor(np.stack([edge1, edge2]))

        data = PyGData(x=feats, edge_index=edge)
        data.ww = torch.tensor(weight)

        return data

    def len(self):
        return int(self.maxEventsList[-1])

