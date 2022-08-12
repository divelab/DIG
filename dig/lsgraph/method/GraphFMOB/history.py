from typing import Optional

import torch
from torch import Tensor


class History(torch.nn.Module):
    r"""A historical embedding storage module."""
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        pin_memory = device is None or str(device) == 'cpu'
        self.emb = torch.empty(num_embeddings, embedding_dim, device=device,
                               pin_memory=pin_memory)

        self._device = torch.device('cpu')

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.fill_(0)

    def _apply(self, fn):
        # Set the `_device` of the module without transfering `self.emb`.
        self._device = fn(torch.zeros(1)).device
        return self

    @torch.no_grad()
    def pull(self, n_id: Optional[Tensor] = None) -> Tensor:
        out = self.emb
        if n_id is not None:
            assert n_id.device == self.emb.device
            out = out.index_select(0, n_id)
        return out.to(device=self._device)

    @torch.no_grad()
    def push(self, x, n_id: Optional[Tensor] = None,
             offset: Optional[Tensor] = None, count: Optional[Tensor] = None):

        if n_id is None and x.size(0) != self.num_embeddings:
            raise ValueError

        elif n_id is None and x.size(0) == self.num_embeddings:
            self.emb.copy_(x)

        elif offset is None or count is None:
            assert n_id.device == self.emb.device
            self.emb[n_id] = x.to(self.emb.device)

        else:  # Push in chunks:
            src_o = 0
            x = x.to(self.emb.device)
            for dst_o, c, in zip(offset.tolist(), count.tolist()):
                self.emb[dst_o:dst_o + c] = x[src_o:src_o + c]
                src_o += c

    def forward(self, *args, **kwargs):
        """"""
        raise NotImplementedError

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_embeddings}, '
                f'{self.embedding_dim}, emb_device={self.emb.device}, '
                f'device={self._device})')
