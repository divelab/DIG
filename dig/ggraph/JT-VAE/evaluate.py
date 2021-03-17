import os
import sys
import json
import argparse

sys.path.append('..')

from utils import metric_random_generation, metric_property_optimization, metric_constrained_optimization
from rdkit import Chem

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', required=True)
parser.add_argument('--samples_file', required=True)

args = parser.parse_args()
print(args)

with open(os.path.join("samples", args.samples_file), "r") as f:
    samples_smiles = f.read().splitlines()
    
sample_mols = list(map(Chem.MolFromSmiles, samples_smiles))

with open(os.path.join(os.pardir, "datasets", args.data_file), "r") as f:
    train_smiles = f.read().splitlines()[1:]
    

with open("results.json", "a") as f:
    f.write(json.dumps(metric_random_generation(sample_mols, train_smiles)))
    f.write(json.dumps(metric_property_optimization(sample_mols)))