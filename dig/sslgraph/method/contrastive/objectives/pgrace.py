import torch
import torch.nn.functional as F


def projection(z: torch.Tensor, fc1, fc2) -> torch.Tensor:
    z = F.elu(fc1(z))
    return fc2(z)

def pGRACE_loss(zs, num_hidden, num_proj_hidden, batch_size=None, tau=0.5, mean:bool = True, **kwargs):
    global fc1, fc2
    fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
    fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
    assert len(zs) == 2
    h1 = projection(zs[0], fc1, fc2)
    h2 = projection(zs[1], fc1, fc2)

    if batch_size is None:
        l1 = semi_loss(h1, h2)
        l2 = semi_loss(h2, h1)
    else:
        l1 = batched_semi_loss(h1, h2, batch_size)
        l2 = batched_semi_loss(h2, h1, batch_size)
    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()

    return ret


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def semi_loss(z1: torch.Tensor, z2: torch.Tensor, tau:float):
    def f(x): return torch.exp(x / tau)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))

    return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))


def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
    # Space complexity: O(BN) (semi_loss: O(N^2))
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: torch.exp(x / self.tau)
    indices = torch.arange(0, num_nodes).to(device)
    losses = []

    for i in range(num_batches):
        mask = indices[i * batch_size:(i + 1) * batch_size]
        refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
        between_sim = f(self.sim(z1[mask], z2))  # [B, N]

        losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                    / (refl_sim.sum(1) + between_sim.sum(1)
                                    - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

    return torch.cat(losses)

