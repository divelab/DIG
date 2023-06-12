import torch
import torch.nn as nn
from torch import Tensor
import pygmtools as pygm


class Sinkhorn(nn.Module):
    r"""
    Sinkhorn algorithm turns the input matrix into a bi-stochastic matrix.
    Sinkhorn algorithm firstly applies an ``exp`` function with temperature :math:`\tau`:
    .. math::
        \mathbf{S}_{i,j} = \exp \left(\frac{\mathbf{s}_{i,j}}{\tau}\right)
    And then turns the matrix into doubly-stochastic matrix by iterative row- and column-wise normalization:
    .. math::
        \mathbf{S} &= \mathbf{S} \oslash (\mathbf{1}_{n_2} \mathbf{1}_{n_2}^\top \cdot \mathbf{S}) \\
        \mathbf{S} &= \mathbf{S} \oslash (\mathbf{S} \cdot \mathbf{1}_{n_2} \mathbf{1}_{n_2}^\top)
    where :math:`\oslash` means element-wise division, :math:`\mathbf{1}_n` means a column-vector with length :math:`n`
    whose elements are all :math:`1`\ s.
    :param max_iter: maximum iterations (default: ``10``)
    :param tau: the hyper parameter :math:`\tau` controlling the temperature (default: ``1``)
    :param epsilon: a small number for numerical stability (default: ``1e-4``)
    :param log_forward: apply log-scale computation for better numerical stability (default: ``True``)
    :param batched_operation: apply batched_operation for better efficiency (but may cause issues for back-propagation,
     default: ``False``)
    .. note::
        ``tau`` is an important hyper parameter to be set for Sinkhorn algorithm. ``tau`` controls the distance between
        the predicted doubly-stochastic matrix, and the discrete permutation matrix computed by Hungarian algorithm (see
        :func:`~src.lap_solvers.hungarian.hungarian`). Given a small ``tau``, Sinkhorn performs more closely to
        Hungarian, at the cost of slower convergence speed and reduced numerical stability.
    .. note::
        We recommend setting ``log_forward=True`` because it is more numerically stable. It provides more precise
        gradient in back propagation and helps the model to converge better and faster.
    .. note::
        Setting ``batched_operation=True`` may be preferred when you are doing inference with this module and do not
        need the gradient.
    """
    def __init__(self, max_iter: int=10, tau: float=1., epsilon: float=1e-4,
                 log_forward: bool=True, batched_operation: bool=False):
        super(Sinkhorn, self).__init__()
        self.max_iter = max_iter
        self.tau = tau
        self.epsilon = epsilon
        self.log_forward = log_forward
        if not log_forward:
            print('Warning: Sinkhorn algorithm without log forward is deprecated because log_forward is more stable.')
        self.batched_operation = batched_operation # batched operation may cause instability in backward computation,
                                                   # but will boost computation.

    def forward(self, s: Tensor, nrows: Tensor=None, ncols: Tensor=None, dummy_row: bool=False) -> Tensor:
        r"""
        :param s: :math:`(b\times n_1 \times n_2)` input 3d tensor. :math:`b`: batch size
        :param nrows: :math:`(b)` number of objects in dim1
        :param ncols: :math:`(b)` number of objects in dim2
        :param dummy_row: whether to add dummy rows (rows whose elements are all 0) to pad the matrix to square matrix.
         default: ``False``
        :return: :math:`(b\times n_1 \times n_2)` the computed doubly-stochastic matrix
        .. note::
            We support batched instances with different number of nodes, therefore ``nrows`` and ``ncols`` are
            required to specify the exact number of objects of each dimension in the batch. If not specified, we assume
            the batched matrices are not padded.
        .. note::
            The original Sinkhorn algorithm only works for square matrices. To handle cases where the graphs to be
            matched have different number of nodes, it is a common practice to add dummy rows to construct a square
            matrix. After the row and column normalizations, the padded rows are discarded.
        .. note::
            We assume row number <= column number. If not, the input matrix will be transposed.
        """
        if self.log_forward:
            return self.forward_log(s, nrows, ncols, dummy_row)
        else:
            return self.forward_ori(s, nrows, ncols, dummy_row) # deprecated

    def forward_log(self, s, nrows=None, ncols=None, dummy_row=False):
        """Compute sinkhorn with row/column normalization in the log space."""
        return pygm.sinkhorn(s, n1=nrows, n2=ncols, dummy_row=dummy_row, max_iter=self.max_iter, tau=self.tau, batched_operation=self.batched_operation, backend='pytorch')

    def forward_ori(self, s, nrows=None, ncols=None, dummy_row=False):
        r"""
        Computing sinkhorn with row/column normalization.
        .. warning::
            This function is deprecated because :meth:`~src.lap_solvers.sinkhorn.Sinkhorn.forward_log` is more
            numerically stable.
        """
        if len(s.shape) == 2:
            s = s.unsqueeze(0)
            matrix_input = True
        elif len(s.shape) == 3:
            matrix_input = False
        else:
            raise ValueError('input data shape not understood.')

        batch_size = s.shape[0]

        #s = s.to(dtype=dtype)

        if nrows is None:
            nrows = [s.shape[1] for _ in range(batch_size)]
        if ncols is None:
            ncols = [s.shape[2] for _ in range(batch_size)]

        # tau scaling
        ret_s = torch.zeros_like(s)
        for b, n in enumerate(nrows):
            ret_s[b, 0:n, 0:ncols[b]] = \
                nn.functional.softmax(s[b, 0:n, 0:ncols[b]] / self.tau, dim=-1)
        s = ret_s

        # add dummy elements
        if dummy_row:
            dummy_shape = list(s.shape)
            dummy_shape[1] = s.shape[2] - s.shape[1]
            #s = torch.cat((s, torch.full(dummy_shape, self.epsilon * 10).to(s.device)), dim=1)
            #nrows = nrows + dummy_shape[1] # non in-place
            s = torch.cat((s, torch.full(dummy_shape, 0.).to(s.device)), dim=1)
            ori_nrows = nrows
            nrows = ncols
            for b in range(batch_size):
                s[b, ori_nrows[b]:nrows[b], :ncols[b]] = self.epsilon

        row_norm_ones = torch.zeros(batch_size, s.shape[1], s.shape[1], device=s.device, dtype=s.dtype)  # size: row x row
        col_norm_ones = torch.zeros(batch_size, s.shape[2], s.shape[2], device=s.device, dtype=s.dtype)  # size: col x col
        for b in range(batch_size):
            row_slice = slice(0, nrows[b])
            col_slice = slice(0, ncols[b])
            row_norm_ones[b, row_slice, row_slice] = 1
            col_norm_ones[b, col_slice, col_slice] = 1

        s += self.epsilon

        for i in range(self.max_iter):
            if i % 2 == 0:
                # column norm
                #ones = torch.ones(batch_size, s.shape[1], s.shape[1], device=s.device)
                sum = torch.sum(torch.mul(s.unsqueeze(3), col_norm_ones.unsqueeze(1)), dim=2)
            else:
                # row norm
                # ones = torch.ones(batch_size, s.shape[2], s.shape[2], device=s.device)
                sum = torch.sum(torch.mul(row_norm_ones.unsqueeze(3), s.unsqueeze(1)), dim=2)

            tmp = torch.zeros_like(s)
            for b in range(batch_size):
                row_slice = slice(0, nrows[b] if nrows is not None else s.shape[2])
                col_slice = slice(0, ncols[b] if ncols is not None else s.shape[1])
                tmp[b, row_slice, col_slice] = 1 / sum[b, row_slice, col_slice]
            s = s * tmp

        if dummy_row:
            if dummy_shape[1] > 0:
                s = s[:, :-dummy_shape[1]]
            for b in range(batch_size):
                s[b, ori_nrows[b]:nrows[b], :ncols[b]] = 0

        if matrix_input:
            s.squeeze_(0)

        return s

class GumbelSinkhorn(nn.Module):
    """
    Gumbel Sinkhorn Layer turns the input matrix into a bi-stochastic matrix.
    See details in `"Mena et al. Learning Latent Permutations with Gumbel-Sinkhorn Networks. ICLR 2018"
    <https://arxiv.org/abs/1802.08665>`_
    :param max_iter: maximum iterations (default: ``10``)
    :param tau: the hyper parameter :math:`\tau` controlling the temperature (default: ``1``)
    :param epsilon: a small number for numerical stability (default: ``1e-4``)
    :param batched_operation: apply batched_operation for better efficiency (but may cause issues for back-propagation,
     default: ``False``)
    .. note::
        This module only supports log-scale Sinkhorn operation.
    """
    def __init__(self, max_iter=10, tau=1., epsilon=1e-4, batched_operation=False):
        super(GumbelSinkhorn, self).__init__()
        self.sinkhorn = Sinkhorn(max_iter, tau, epsilon, batched_operation=batched_operation)

    def forward(self, s: Tensor, nrows: Tensor=None, ncols: Tensor=None,
                sample_num=5, dummy_row=False) -> Tensor:
        r"""
        :param s: :math:`(b\times n_1 \times n_2)` input 3d tensor. :math:`b`: batch size
        :param nrows: :math:`(b)` number of objects in dim1
        :param ncols: :math:`(b)` number of objects in dim2
        :param sample_num: number of samples
        :param dummy_row: whether to add dummy rows (rows whose elements are all 0) to pad the matrix to square matrix.
         default: ``False``
        :return: :math:`(b m\times n_1 \times n_2)` the computed doubly-stochastic matrix. :math:`m`: number of samples
         (``sample_num``)
        The samples are stacked at the fist dimension of the output tensor. You may reshape the output tensor ``s`` as:
        ::
            s = torch.reshape(s, (-1, sample_num, s.shape[1], s.shape[2]))
        .. note::
            We support batched instances with different number of nodes, therefore ``nrows`` and ``ncols`` are
            required to specify the exact number of objects of each dimension in the batch. If not specified, we assume
            the batched matrices are not padded.
        .. note::
            The original Sinkhorn algorithm only works for square matrices. To handle cases where the graphs to be
            matched have different number of nodes, it is a common practice to add dummy rows to construct a square
            matrix. After the row and column normalizations, the padded rows are discarded.
        .. note::
            We assume row number <= column number. If not, the input matrix will be transposed.
        """
        def sample_gumbel(t_like, eps=1e-20):
            """
            randomly sample standard gumbel variables
            """
            u = torch.empty_like(t_like).uniform_()
            return -torch.log(-torch.log(u + eps) + eps)

        s_rep = torch.repeat_interleave(s, sample_num, dim=0)
        s_rep = s_rep + sample_gumbel(s_rep)
        nrows_rep = torch.repeat_interleave(nrows, sample_num, dim=0)
        ncols_rep = torch.repeat_interleave(ncols, sample_num, dim=0)
        s_rep = self.sinkhorn(s_rep, nrows_rep, ncols_rep, dummy_row)
        #s_rep = torch.reshape(s_rep, (-1, sample_num, s_rep.shape[1], s_rep.shape[2]))
        return s_rep