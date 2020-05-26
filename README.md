# FederatedLearningSimulation
An implementation of some FL methods

1. First, download the dataset [emnist](https://storage.googleapis.com/tff-datasets-public/fed_emnist_digitsonly.tar.bz2).
2. Run `python sample.py` to get the initial model weights and the clients training orders.
3. Run `*_test.py` where * is the method you want to test.
**Note: In my experiment, FedProx shows its advantage only in some cases that the data is highly heterogeneous and there are many stragglers in the federation. You can get it using the data [mnist](https://github.com/litian96/FedProx/tree/master/data/mnist) provided by FedProx's author.**