Neural Architecture Search (NAS) is a method for automatically designing and optimizing the architecture of a neural network. It involves using a search algorithm to explore the space of possible network architectures and selecting the best one based on some performance criteria. This notebook implements a NAS class for MLP which is used in the tcPINNs project.

Prerequisites/Dependencies: PyTorch, random, torch.autograd, torch.utils.data, torch.nn

Usage: Here is a demo to create a NAS object and perform search:

nas_demo = MLPNAS(10) #create a MLPNAS object that can at most produce 10 layers
nas_history = nas_demo.search_pinn(3,True) # search for best architecture with input size of 3
best_architecture,_ = get_best_architecture(nas_history)# get the best architeture from the searching history
best_model = nas_demo.model_generator.create_model(best_architecture) #create corresponding model
best_model.train()
...

Limitations: currently only applicable to MLP.But is scalable to more complex structures such as CNN,and RNN, etc

