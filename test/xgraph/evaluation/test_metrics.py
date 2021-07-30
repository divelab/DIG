from dig.xgraph.evaluation import XCollector, ExplanationProcessor, control_sparsity
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils.random import barabasi_albert_graph
from torch_geometric.data import Data
import torch


def test_metrics():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # --- Create a model ---
    model = GCNConv(in_channels=1, out_channels=2).to(device)

    # --- Set the Sparsity to 0.5 ---
    sparsity = 0.5

    # --- Create data collector and explanation processor ---
    x_collector = XCollector(sparsity)
    x_processor = ExplanationProcessor(model=model, device=device)

    # --- Given a 2-class classification with 10 explanation ---
    num_classes = 2
    for _ in range(10):
        # --- Create random ten-node BA graph ---
        x = torch.ones((10, 1), dtype=torch.float)
        edge_index = barabasi_albert_graph(10, 3)
        data = Data(x=x, edge_index=edge_index, y=torch.tensor([1.]))  # Assume that y is the ground-truth valuing 1

        # --- Create random explanation ---
        masks = [control_sparsity(torch.randn(edge_index.shape[1], device=device), sparsity) for _ in
                 range(num_classes)]

        # --- Process the explanation including data collection ---
        x_processor(data, masks, x_collector)

    # --- Get the evaluation metric results from the data collector ---
    print(f'Fidelity: {x_collector.fidelity:.4f}\n'
          f'Fidelity_inv: {x_collector.fidelity_inv:.4f}\n'
          f'Sparsity: {x_collector.sparsity:.4f}')

    assert x_collector.fidelity is not None
    assert x_collector.fidelity_inv is not None
    assert x_collector.sparsity is not None


if __name__ == '__main__':
    test_metrics()
