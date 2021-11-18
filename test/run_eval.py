#!/usr/bin/env python
import sys, os
sys.path.append("../python")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', action='store', type=str, required=True, help='Path to the input file')
parser.add_argument('-t', '--trained', action='store', type=str, required=True, help='Path to training output directory')
parser.add_argument('-c', '--config', action='store', type=str, help='Configration file with sample information')
parser.add_argument('--device', action='store', type=int, default=0, help='device name')
parser.add_argument('--batch', action='store', type=int, default=32, help='Batch size')
parser.add_argument('--edgeType', action='store', type=str, default='decay',
                                  choices=('none', 'all', 'decay', 'color'), help='graph edge type')
args = parser.parse_args()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv(args.trained+'/train.csv')
epochs = np.arange(1, len(df['trn_loss'])+1)
plt.plot(epochs, df['trn_loss'], label='Training')
plt.plot(epochs, df['val_loss'], label='Validation')
plt.xscale('log')
plt.yscale('log')
plt.grid(linestyle=':', which='both')
plt.xlabel('Epoch')
plt.ylabel('MSE loss')
plt.tight_layout()
plt.show()

### Load configuration from the yaml file
import yaml
config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)

if not os.path.exists(args.trained):
    print("Cannot find input directory")
    os.exit()

import torch
torch.set_num_threads(os.cpu_count())
if torch.cuda.is_available() and args.device >= 0: torch.cuda.set_device(args.device)

##### Define dataset instance #####
from dataset.LHEGraphDataset import *
from torch_geometric.data import DataLoader
dset = LHEGraphDataset(edgeType=args.edgeType)
#for sampleInfo in config['samples']:
#    if 'ignore' in sampleInfo and sampleInfo['ignore']: continue
#    name = sampleInfo['name']
#    dset.addSample(name, sampleInfo['path'], scale=sampleInfo['xsec']/sampleInfo['ngen'])
#    dset.setProcessLabel(name, sampleInfo['label'])
dset.addSample('tttt', args.input, scale=1)
dset.setProcessLabel('tttt', 'signal')
dset.load()

kwargs = {'num_workers':min(config['training']['nDataLoaders'], os.cpu_count()), 'pin_memory':False}
dataLoader = DataLoader(dset, batch_size=args.batch, shuffle=False, **kwargs)

##### Define model instance #####
from model.GCN import GCN
#model = GCN(dset.nFeats)
model = torch.load(os.path.join(args.trained, 'model.pth'))
model.load_state_dict(torch.load(os.path.join(args.trained, 'weight.pth')))
model.eval()
device = 'cpu'
if args.device >= 0 and torch.cuda.is_available():
    model = model.cuda()
    device = 'cuda:%d' % args.device

from tqdm import tqdm
scores = np.ones(len(dset))*(-1)
for i, data in enumerate(tqdm(dataLoader)):
    data = data.to(device)
    w = data.ww
    pred = model(data)
    scores[i*args.batch:i*args.batch+len(pred)] = pred.detach().to('cpu').reshape(-1)

plt.hist(scores, bins=125, range=(-1,1.5), histtype='step', density=True)
#plt.yscale('log')
plt.grid(linestyle=':', which='both')
plt.xlabel('Event Weight')
plt.ylabel('Arbitrary Unit')
plt.tight_layout()
plt.show()
