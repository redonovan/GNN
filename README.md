# GNN
<b>Implementation</b>

My code in <a href=https://github.com/redonovan/GNN/blob/main/gnn.py>gnn.py</a> is implemented in PyTorch 1.11.0 directly from the paper <a href=https://arxiv.org/abs/1704.01212>Neural Message Passing for Quantum Chemistry, Gilmer et al., 2017</a>.  It uses a graph neural network, implemented using the <a href=https://pytorch-geometric.readthedocs.io/en/stable/index.html>PyTorch Geometric</a> library, to predict the chemical properties of molecules in the QM9 dataset.  The implementation is of the best performing network in the paper, using Edge Networks for message passing, a GRU for an update function, and a set2set readout function based on the paper <a href=https://arxiv.org/abs/1511.06391>Order Matters: Sequence to sequence for sets, Vinyals et al., 2015</a>.  

Gilmer et al. predicted each target individually, using a random hyperparameter search with 50 trials for each target, with each trial training for 540 epochs.  Unfortunately a single 540 epoch train on the full ~110k molecule training dataset would take approximately 329 hours on my laptop, which is prohibitive.  Since I don't have Google's compute resources or budget, I instead predicted all 12 targets simultaneously (a 'joint' train) and used an abbreviated hyperparameter search using a much smaller dataset and fewer epochs.  My final hyperparameters are given in my code and were used to perform one 'joint' train on an 11k molecule subset of the training dataset for 540 epochs.

My results are shown in the following table, where MAE = Mean Absolute Error, and Ratio is MAE/ChemAcc, where the Chemical Accuracy is the accuracy required to make realistic chemical predictions.

| Symbol | Description                                 | Unit     | ChemAcc |   MAE   |   Ratio |
| :--    | :-------------------------------------------|:---------|:--------|--------:|--------:|
| mu     | Dipole moment                               | D        | 0.1     |  0.2313 |  2.3132 |
| alpha  | Isotropic polarizability                    | a0^3     | 0.1     |  0.4741 |  4.7409 |
| HOMO   | Highest occupied molecular orbital energy   | eV       | 0.043   |  0.1229 |  2.8571 |
| LUMO   | Lowest unoccupied molecular orbital energy  | eV       | 0.043   |  0.1175 |  2.7331 |
| gap    | Gap between HOMO and LUMO                   | eV       | 0.043   |  0.1684 |  3.9172 |
| R2     | Electronic spatial extent                   | a0^2     | 1.2     | 10.0326 |  8.3605 |
| ZPVE   | Zero point vibrational energy               | eV       | 0.0012  |  0.0160 | 13.3651 |
| U0atom | Atomization energy at 0K                    | eV       | 0.043   |  0.1988 |  4.6226 |
| Uatom  | Atomization energy at 298.15K               | eV       | 0.043   |  0.1998 |  4.6460 |
| Hatom  | Atomization enthalpy at 298.15K             | eV       | 0.043   |  0.1999 |  4.6500 |
| Gatom  | Atomization free energy at 298.15K          | eV       | 0.043   |  0.1936 |  4.5021 |
| Cv     | Heat capacity at 298.15K                    | cal/molK | 0.05    |  0.1858 |  3.7165 |

These results are most appropriately compared to the N=11k column in Table 6 in the paper.

My results are less good than those in Table 6; possible reasons for this include:

1. The paper used a separate model for each target which can improve results by up to 40%.
2. The paper used Acceptor and Donor atom features, which are not in the PyTorch Geometric version of QM9.
3. The paper used a more thorough hyperparameter search.
4. The paper does not give details of nn1, nn2 and lin (see my code) and my versions may not be optimal.
5. The paper does not mention regularization, and my choices may not be optimal. 
6. Mistakes or omissions I may have made.

The paper also predicted a 13th target, Omega, which is not in the PyTorch Geometric version of QM9.

TensorBoard <a href=https://github.com/redonovan/GNN/blob/main/TensorBoardTrain.png>training</a> and <a href=https://github.com/redonovan/GNN/blob/main/TensorBoardValid.png>validation</a> loss plots are available here; training took 33 hours over 3 nights.
