# Implementation of the best system from "Neural Message Passing for Quantum Chemistry",
# Gilmer et al., 2017, https://arxiv.org/abs/1704.01212, applying GNN to the QM9 dataset.
# Oct-Nov, 2023 (v10)


import os
import torch
torch.manual_seed(42)

import torch.nn.functional as F
import torch_geometric.transforms as transforms
from torch_geometric.nn import NNConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn.aggr import Set2Set
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
from torch.optim.lr_scheduler import ConstantLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR
from torch.utils.tensorboard import SummaryWriter
from checkpoint import save_checkpoint, load_checkpoint
from add_missing_edges import AddMissingEdges


class GNN(torch.nn.Module):
    def __init__(self, num_edge_features, num_node_features, hidden_dim, T, dropout, M, num_targets):
        super().__init__()
        self.num_edge_features = num_edge_features # number of edge features in the data (5)
        self.num_node_features = num_node_features # number of node features in the data (11)
        self.hidden_dim = hidden_dim               # hidden dimension used for nodes
        self.T = T                                 # number of message passing layers / updates
        self.dropout = dropout                     # dropout probability during training
        self.M = M                                 # number of set2set processing steps
        self.num_targets = num_targets             # number of targets being predicted
        # nn1 converts edge features into matrices
        self.nn1 = torch.nn.Sequential(
            torch.nn.Linear(num_edge_features, hidden_dim),     # mostly embedding
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),            # pure learning
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim*hidden_dim), # projection
        )
        # message computation from connected edges and neighbouring nodes
        self.nnconv = NNConv(hidden_dim, hidden_dim, self.nn1, aggr='add', root_weight=False, bias=False)
        # message passing update rnn
        self.rnn = torch.nn.GRUCell(hidden_dim, hidden_dim)
        # dropout regularization layer
        self.drp = torch.nn.Dropout(p=dropout)
        # projection layer from (hT,x) concatenation to a new larger h
        self.lin = torch.nn.Linear(hidden_dim + num_node_features, 2*hidden_dim)
        # set2set readout function produces one vector per molecule
        self.s2s = Set2Set(2*hidden_dim, M, num_layers=2, dropout=dropout)
        # nn2 converts set2set outputs into target predictions
        self.nn2 = torch.nn.Sequential(
            torch.nn.Linear(4*hidden_dim, 8*hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(8*hidden_dim, num_targets),
        )
        #
    def forward(self, data):
        x, edge_index, edge_attr, index, batch_size = data.x, data.edge_index, data.edge_attr, data.batch, len(data.name)
        # x           shape (num_nodes, num_node_features)
        # edge_index  shape (2, num_edges)
        # edge_attr   shape (num_edges, num_edge_features)
        # index       shape (nnodes) assigns node to molecule
        # batch_size  number of molecules in this batch
        #
        # initialize h with x, zero padding the feature dimension up to hidden_dim
        assert(x.shape[1] <= self.hidden_dim)
        h = x.new_zeros((x.shape[0], self.hidden_dim))                     # shape (num_nodes, hidden_dim)
        h[:,:x.shape[1]] = x
        # apply message passing and GRU updates T times
        for _ in range(self.T):
            # the NNConv layer computes m_v^(t+1) (eqn 1 in the paper)
            m = self.nnconv(h, edge_index, edge_attr, size=None)           # shape (num_nodes, hidden_dim)
            # a GRU is used for the update function (eqn 2 in the paper)
            h = self.rnn(m, h)                                             # shape (num_nodes, hidden_dim)
            # apply dropout
            h = self.drp(h)                                                # shape (num_nodes, hidden_dim)
        # concatenate hT and the input x
        h = torch.cat((h, x), dim=1)                                       # shape (num_nodes, hidden_dim + num_node_features)
        # apply a linear projection
        h = self.lin(h)                                                    # shape (num_nodes, 2*hidden_dim)
        # apply set2set to produce one vector per molecule
        q_star = self.s2s(h, index, ptr=None, dim_size=batch_size, dim=-2) # shape (batch_size, 4*hidden_dim)
        # apply a neural network to predict the correct number of targets
        h = self.nn2(q_star)                                               # shape (batch_size, num_targets)
        return h # shape (batch_size, num_targets)


# Hyperparameters

hp={
    'hidden_dim' : 200,  # The hidden dimension used for nodes in the model; the paper used values between 43-200
    'T'          : 3,    # The number of message passing / update layers; the paper used values between 3 and 8
    'dropout'    : 0.5,  # The dropout probability to use during training; not mentioned in paper
    'M'          : 6,    # The number of set2set computation steps; the paper used values between 1 and 12
    'batch_size' : 16,   # The number of molecules in each batch; the paper used 20
    'num_epochs' : 540,  # The number of training epochs; the paper used 540
    'ilr'        : 1e-4, # The initial learning rate; the paper used values between 1e-5 and 5e-4
    'lr_hold'    : 135,  # The number of epochs for which the learning rate is held constant; .1 to .9 * num_epochs
    'lr_decay_f' : .01,  # The factor to which the learning rate linearly falls after the hold; .01 - 1
    'target_idx' : 'j',  # Either 'j' for joint training (all 12 targets) or the individual target index
}

# Device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data

# See https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.datasets.QM9.html
# The dataset consists of about 130,000 molecules with 19 targets.

transform = transforms.Compose([
    AddMissingEdges(),                # adds edges between all unbonded atoms, with zeros for attributes (ie. no bonds)
    transforms.Distance(norm=False),  # adds pairwise node distances to the edge attributes
])

dataset           = QM9(
    root='/data/pytorch_data/QM9',                # location where the data will be saved
    transform=transform)
dataset           = dataset.shuffle()
valid_dataset     = dataset[:1000]                # 1k;  would be 10k for a full-dataset train,   as [:10000]
test_dataset      = dataset[10000:20000]          # 10k
train_dataset     = dataset[20000:31000]          # 11k; would be ~110k for a full-dataset train, as [20000:]
num_edge_features = dataset[0].edge_attr.shape[1] # 5;  1-hot bond-type for 4 dimensions, plus (float) distance
num_node_features = dataset[0].x.shape[1]         # 11; 1-hot H,C,N,O,F; (int) Atomic #, (bin) Aromatic, sp, sp2, sp3; (int) #Hs

# compute means and stdevs of the training data targets
ydata  = torch.cat([train_dataset.get(i).y for i in range(len(train_dataset))], dim=0).to(device)
ymeans = ydata.mean(dim=0)
ystds  = ydata.std(dim=0)

# this version of QM9 contains 19 targets, 12 of which match the first 12 in the NMP4QC paper
if hp['target_idx'] == 'j':
    target_indices = [0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11] # in paper order, see target_details below
    num_targets = 12
else:
    target_indices = [hp['target_idx'],]
    num_targets = 1

# loss function
def loss_fn(preds, batchy):
    # the model is trained to predict normalized targets (mean 0, variance 1)
    normy   = (batchy - ymeans) / ystds
    targets = normy[:, target_indices]
    loss    = F.mse_loss(preds, targets)
    return loss

# training setup
model        = GNN(num_edge_features, num_node_features, hp['hidden_dim'], hp['T'], hp['dropout'], hp['M'], num_targets)
model        = model.to(device)
optimizer    = torch.optim.Adam(model.parameters(), lr=hp['ilr'])
scheduler1   = ConstantLR(optimizer, factor=1.0, total_iters=hp['lr_hold']-1)
scheduler2   = LinearLR(optimizer, start_factor=1.0, end_factor=hp['lr_decay_f'], total_iters=hp['num_epochs']-hp['lr_hold'])
scheduler    = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[hp['lr_hold']-1])
train_loader = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=hp['batch_size'], shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=hp['batch_size'], shuffle=False)

# Run Tensorboard with 'tensorboard --logdir=runs'
# Point browser at http://localhost:6006/
writer = SummaryWriter('runs')

# Training runs can take some time.
# Checkpoints are therefore saved every epoch so training runs can be stopped.
# Restart training from the last checkpoint if available, otherwise start from scratch.

if os.path.isfile('ckpt_last'):
    epoch, steps, best_valid_loss = load_checkpoint('ckpt_last', hp, model, optimizer, scheduler)
    start_epoch = epoch + 1
else:
    start_epoch, steps, best_valid_loss = 0, 0, float('inf')

# training loop

for epoch in range(start_epoch, hp['num_epochs']):
    if epoch == start_epoch:
        print(f'Training from epoch {start_epoch}...')
    _ = model.train()
    train_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        steps += 1
    train_loss /= len(train_loader)
    writer.add_scalar('Train Loss', train_loss, steps)
    # valid
    _ = model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for batch in valid_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = loss_fn(out, batch.y)
            valid_loss += loss.item()
    valid_loss /= len(valid_loader)
    writer.add_scalar('Valid Loss', valid_loss, steps)    
    print(f'epoch {epoch:2d} steps {steps:4d} '
          f'Loss : train {train_loss:.4f} valid {valid_loss:.4f}')
    scheduler.step()
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        save_checkpoint('ckpt_best', hp, model, optimizer, scheduler, epoch, steps, best_valid_loss)
    save_checkpoint('ckpt_last', hp, model, optimizer, scheduler, epoch, steps, best_valid_loss)


# Test data.
print('Testing...')
# Load the best ckpt from the training run.
epoch, steps, best_valid_loss = load_checkpoint('ckpt_best', hp, model, optimizer, scheduler)

_ = model.eval()
test_loss = 0.0
mae = torch.zeros(num_targets).to(device)
nm  = 0
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        # compute single-figure normalized loss, as in training
        loss = loss_fn(out, batch.y)
        test_loss += loss.item()
        # also compute unnormalized summed absolute error for each target
        targets  = batch.y[:, target_indices]
        unormout = (out * ystds[target_indices]) + ymeans[target_indices]
        sae = F.l1_loss(unormout, targets, reduction='none').sum(dim=0)
        mae += sae
        nm  += out.shape[0]

test_loss /= len(test_loader)
mae /= nm
print(f'epoch {epoch:2d} steps {steps:4d} Loss : test {test_loss:.4f}')

# Target details, compiled from the pytorch_geometric QM9 page and the NMP4QC paper.
# Entries are target_index : [symbol, description, unit, chemical accuracy]
# Chemical accuracy is the accuracy required to make realistic chemical predictions.

target_details={
     0: ['mu', 'Dipole moment', 'D', .1],
     1: ['alpha', 'Isotropic polarizability', 'a0^3', .1],
     2: ['HOMO', 'Highest occupied molecular orbital energy', 'eV', .043],
     3: ['LUMO', 'Lowest unoccupied molecular orbital energy', 'eV', .043],
     4: ['gap', 'Gap between HOMO and LUMO', 'eV', .043],
     5: ['R2', 'Electronic spatial extent', 'a0^2', 1.2],
     6: ['ZPVE', 'Zero point vibrational energy', 'eV', .0012],
    12: ['U0atom', 'Atomization energy at 0K', 'eV', .043],
    13: ['Uatom', 'Atomization energy at 298.15K', 'eV', .043],
    14: ['Hatom', 'Atomization enthalpy at 298.15K', 'eV', .043],
    15: ['Gatom', 'Atomization free energy at 298.15K', 'eV', .043],
    11: ['Cv', 'Heat capacity at 298.15K', 'cal/molK', .050],
}

# Print out the target details together with the Mean Absolute Error of the model predictions
# on test data and the ratio MAE/ChemicalAccuracy as presented in the NMP4QC paper.

for e, ti in enumerate(target_indices):
    if e == 0:
        print(f'Symbol  Description                                  Unit      ChemAcc    MAE        Ratio')
    t = target_details[ti]
    print(f'{t[0]:7} {t[1]:44} {t[2]:9} {t[3]:<7} {mae[e]:9.4f}  {mae[e]/t[3]:9.4f}')

