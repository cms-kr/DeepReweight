#!/usr/bin/env python
import os, sys
sys.path.append("../python")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', action='store', type=str, required=True, help='Path to output directory')

parser.add_argument('--config', action='store', type=str, help='Configration file with sample information')
parser.add_argument('--epoch', action='store', type=int, default=400,help='Number of epochs')
parser.add_argument('--batch', action='store', type=int, default=32, help='Batch size')
parser.add_argument('--lr', action='store', type=float, default=1e-4,help='Learning rate')
parser.add_argument('--seed', action='store', type=int, default=12345,help='random seed')
parser.add_argument('--noshuffle', action='store_true', default=False, help='do not shuffle dataset')

parser.add_argument('--device', action='store', type=int, default=0, help='device name')

parser.add_argument('--edgeType', action='store', type=str, default='decay',
                                  choices=('none', 'all', 'decay', 'color'), help='graph edge type')
args = parser.parse_args()

### Load configuration from the yaml file
import yaml
config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)
config['training']['learningRate'] = float(config['training']['learningRate'])
if args.seed: config['training']['randomSeed1'] = args.seed
if args.epoch: config['training']['epoch'] = args.epoch
if args.lr: config['training']['learningRate'] = args.lr
if args.noshuffle: config['training']['shuffle'] = not args.noshuffle

import torch
torch.set_num_threads(os.cpu_count())
if torch.cuda.is_available() and args.device >= 0: torch.cuda.set_device(args.device)
if not os.path.exists(args.output): os.makedirs(args.output)

##### Define dataset instance #####
from dataset.LHEGraphDataset import *
from torch_geometric.data import DataLoader
dset = LHEGraphDataset(edgeType=args.edgeType)
for sampleInfo in config['samples']:
    if 'ignore' in sampleInfo and sampleInfo['ignore']: continue
    name = sampleInfo['name']
    dset.addSample(name, sampleInfo['path'], scale=sampleInfo['xsec']/sampleInfo['ngen'])
    dset.setProcessLabel(name, sampleInfo['label'])
dset.load()

doShuffle = config['training']['shuffle']
lengths = [int(x*len(dset)) for x in config['training']['splitFractions']]
lengths.append(len(dset)-sum(lengths))
torch.manual_seed(config['training']['randomSeed1'])
if doShuffle:
    trnDset, valDset, testDset = torch.utils.data.random_split(dset, lengths)
else:
    lengths[2] = np.arange(lengths[1]+1, lengths[2]+1)
    lengths[1] = np.arange(lengths[0]+1, lengths[1]+1)
    lengths[0] = np.arange(2, lengths[0]+1)
    trnDset, valDset, testDset = torch.utils.data.Subset(dset, lengths)

kwargs = {'num_workers':min(config['training']['nDataLoaders'], os.cpu_count()), 'pin_memory':False}
trnLoader = DataLoader(trnDset, batch_size=args.batch, shuffle=doShuffle, **kwargs)
valLoader = DataLoader(valDset, batch_size=args.batch, shuffle=False, **kwargs)
torch.manual_seed(torch.initial_seed())

##### Define model instance #####
from model.GCN import GCN
model = GCN(dset.nFeats)
torch.save(model, os.path.join(args.output, 'model.pth'))
device = 'cpu'
if args.device >= 0 and torch.cuda.is_available():
    model = model.cuda()
    device = 'cuda:%d' % args.device

##### Define optimizer instance #####
import torch.optim as optim
optm = optim.Adam(model.parameters(), lr=config['training']['learningRate'])
crit = torch.nn.MSELoss(size_average=None, reduction='sum')

##### Start training #####
with open(args.output+'/summary.txt', 'w') as fout:
    fout.write(str(args))
    fout.write('\n\n')
    fout.write(str(model))
    fout.close()

#from sklearn.metrics import accuracy_score
import csv
from tqdm import tqdm
bestState, bestLoss = {}, 1e9
train = {'trn_loss':[], 'val_loss':[],}
nEpoch = config['training']['epoch']

for epoch in range(nEpoch):
    model.train()

    trn_loss = 0.
    nProcessed = 0
    optm.zero_grad()
    for i, data in enumerate(tqdm(trnLoader, desc='epoch %d/%d' % (epoch+1, nEpoch))):
        data = data.to(device)
        data.ww = data.ww.to(device)
        data.edge_index = data.edge_index.to(device)
        w = data.ww

        pred = model(data)

        loss = crit(pred.view(-1), w)
        loss.backward()

        optm.step()
        optm.zero_grad()

        ibatch = len(w)
        trn_loss += loss.item()
        nProcessed += ibatch

    trn_loss /= nProcessed 

    model.eval()
    val_loss = 0.
    nProcessed = 0
    for i, data in enumerate(tqdm(valLoader)):
        data = data.to(device)
        data.ww = data.ww.to(device)
        w = data.ww

        pred = model(data)
        
        loss = crit(pred.view(-1), w)

        ibatch = len(w)
        val_loss += loss.item()
        nProcessed += ibatch

    val_loss /= nProcessed

    if bestLoss > val_loss:
        bestState = model.to('cpu').state_dict()
        bestLoss = val_loss
        torch.save(bestState, os.path.join(args.output, 'weight.pth'))

        model.to(device)

    train['trn_loss'].append(trn_loss)
    train['val_loss'].append(val_loss)

    with open(os.path.join(args.output, 'train.csv'), 'w') as f:
        writer = csv.writer(f)
        keys = train.keys()
        writer.writerow(keys)
        for row in zip(*[train[key] for key in keys]):
            writer.writerow(row)

bestState = model.to('cpu').state_dict()
torch.save(bestState, os.path.join(args.output, 'weightFinal.pth'))

