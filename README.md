# GNN
<b>Implementation</b>

My code in <a href=https://github.com/redonovan/GNN/blob/main/gnn.py>gnn.py</a> is implemented in PyTorch 1.11.0 directly from the paper <a href=https://arxiv.org/abs/1704.01212>Neural Message Passing for Quantum Chemistry, Gilmer et al., 2017</a>.  It uses a graph neural network, implemented using the <a href=https://pytorch-geometric.readthedocs.io/en/stable/index.html>PyTorch Geometric</a> library, to predict the chemical properties of molecules in the QM9 dataset.  The implementation is of the best performing network in the paper, using Edge Networks for message passing, a GRU for an update function, and a set2set readout function based on the paper <a href=https://arxiv.org/abs/1511.06391>Order Matters: Sequence to sequence for sets, Vinyals et al., 2015</a>.  

Gilmer et al. used a random hyperparameter search with 50 trials for each model and target combination, each training for 540 epochs.  Since a single 540 epoch trial takes ~64 hours on my laptop, I instead ran an abbreviated 10 trial hyperparameter search using a cutdown 11k molecule training dataset, each trial training for only 10 epochs.  My final hyperparameters are given in the code.  I performed only one 540 epoch train on the full training dataset of ~110k molecules, predicting all 12 targets at once (a 'joint' train).

My results are shown in the following table, where MAE = Mean Absolute Error, and Ratio is MAE/ChemAcc, where the Chemical Accuracy is the accuracy required to make realistic chemical predictions.

| Symbol | Description                                 | Unit     | ChemAcc |   MAE       Ratio |
| :--    | :-------------------------------------------|:---------|:--------|------------------:|
| mu     | Dipole moment                               | D        | 0.1     |   0.6690     6.6899 |
| alpha  | Isotropic polarizability                    | a0^3     | 0.1     |   2.9664    29.6642 |
| HOMO   | Highest occupied molecular orbital energy   | eV       | 0.043   |   0.2000     4.6519 |
| LUMO   | Lowest unoccupied molecular orbital energy  | eV       | 0.043   |   0.2333     5.4250 |
| gap    | Gap between HOMO and LUMO                   | eV       | 0.043   |   0.2968     6.9024 |
| R2     | Electronic spatial extent                   | a0^2     | 1.2     | 106.3771    88.6476 |
| ZPVE   | Zero point vibrational energy               | eV       | 0.0012  |   0.1793   149.4524 |
| U0atom | Atomization energy at 0K                    | eV       | 0.043   | 355.8621  8275.8633 |
| Uatom  | Atomization energy at 298.15K               | eV       | 0.043   | 356.8731  8299.3740 |
| Hatom  | Atomization enthalpy at 298.15K             | eV       | 0.043   | 360.7728  8390.0645 |
| Gatom  | Atomization free energy at 298.15K          | eV       | 0.043   | 355.9486  8277.8740 |
| Cv     | Heat capacity at 298.15K                    | cal/molK | 0.05    |   1.3688    27.3755 |

These results are slightly less good than those in Table 2 in the paper, which used a separate model for each target rather than a joint train, a more exhaustive hyperparameter search, and, it would appear, Acceptor and Donor atom features, for which I do not have data.  The paper also predicted a 13th target, Omega, for which I also do not have data.
