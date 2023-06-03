from dig.fairgraph.method import run
from dig.fairgraph.dataset import POKEC, NBA
import torch


# # Load the dataset and split
# pokec = POKEC(dataset_sample='pockec_n')

# # Define model, loss, and evaluation


# # Train and evaluate
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# run_fairgraph = run()
# run_fairgraph.run(device,dataset=pokec,model='Graphair',epochs=2_000,batch_size=100,
#             lr=1e-4,weight_decay=1e-5)



# Load the dataset and split
nba = NBA()

# Define model, loss, and evaluation


# Train and evaluate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_fairgraph = run()
run_fairgraph.run(device,dataset=nba,model='Graphair',epochs=2_000,test_epochs=1_000,
            lr=1e-4,weight_decay=1e-5)