# A Multi-Agent Reinforcement Learning Approach to the Coevolution of Signalling Behaviour

This repository contains the code for replication of the analysis in the paper *Maynard Smith Revisited: A Multi-Agent Reinforcement Learning Approach to the Coevolution of Signalling Behaviour.*

The code is used for an implementation of the Sir Philip Sidney game proposed by Maynard Smith. The game involves two players: *Beneficiary* (B) and *Donor* (D). B can choose to *signal* whether or not they are in need, and D can choose to *give* or *keep* an indivisible resource. 

Running experiments:
- Learning and discount rate, exploration length and decay can be set in lr_dr.py
- Game parameters are set in param.py
- signal_learning.py allows for individual runs
- signal_learning_runs.py can be used for multiple simultaneous runs and aggregated results 
