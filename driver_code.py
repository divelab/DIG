from dig.fairgraph.method import run
from dig.fairgraph.dataset import POKEC
import torch


# Load the dataset and split
pokec = POKEC()

# Define model, loss, and evaluation


# Train and evaluate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_fairgraph = run()
run_fairgraph.run(device,dataset=pokec,model='Graphair',epochs=10_000,batch_size=1_000,
            lr=1e-4,weight_decay=1e-5,save_dir='',log_dir='')