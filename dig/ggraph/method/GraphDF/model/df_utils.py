"""
author: trentbrick and yannadani
Utils for the discrete layers. Taken from https://github.com/google/edward2/blob/2077d67ab8a5c73c39b8d43ccc8cd036dc0a8566/edward2/tensorflow/layers/utils.py 
Which is introduced and explained in the paper: https://arxiv.org/abs/1905.10347 
And modified for PyTorch. 
"""
import sys
import warnings
import torch
import torch.nn.functional as F
import numpy as np

def one_hot(inputs, vocab_size = None):
    """Returns one hot of data over each element of the inputs"""
    if vocab_size is None:
        vocab_size = inputs.max() + 1
    input_shape = inputs.shape
    inputs = inputs.flatten().unsqueeze(1).long()
    z = torch.zeros(len(inputs), vocab_size, device=inputs.device)
    z.scatter_(1, inputs, 1.)
    return z.view(*input_shape, vocab_size)

def one_hot_argmax(inputs, temperature=0.1, axis=-1):
    """Returns one-hot of argmax with backward pass set to softmax-temperature."""
    vocab_size = inputs.shape[-1]
    z = one_hot(torch.argmax(inputs, dim=axis), vocab_size) 
    soft = F.softmax(inputs / temperature, dim=axis)
    outputs = soft + (z - soft).detach()
    return outputs

def multiplicative_inverse(a, n):
    """Multiplicative inverse of a modulo n.
    Args:
        a: Tensor of shape [..., vocab_size]. It denotes an integer in the one-hot
        space.
        n: int Tensor of shape [...].
    Returns:
        Tensor of same shape and dtype as a.
    """

    vocab_size = a.shape[-1]
    sparse_a = torch.argmax(a, dim=-1)
    sparse_outputs = torch.tensor(py_multiplicative_inverse( sparse_a, n))
    z = one_hot(sparse_outputs, vocab_size)
    return z

def py_multiplicative_inverse(a, n):
    """Multiplicative inverse of a modulo n (in Python).
    Implements extended Euclidean algorithm.
    Args:
        a: int-like np.ndarray.
        n: int.
    Returns:
        Multiplicative inverse as an int32 np.ndarray with same shape as a.
    """
    batched_a = np.asarray(a, dtype=np.int32)
    n = np.asarray(n, dtype=np.int32)
    batched_inverse = []
    for a in np.nditer(batched_a):
        inverse = 0
        new_inverse = 1
        remainder = n
        new_remainder = a
        while new_remainder != 0:
            quotient = remainder // new_remainder
            (inverse, new_inverse) = (new_inverse, inverse - quotient * new_inverse)
            (remainder, new_remainder) = (new_remainder,
                                            remainder - quotient * new_remainder)
            
        if remainder > 1:
            raise ValueError(
                'Inverse for {} modulo {} does not exist.'.format(a, n))
        if inverse < 0:
            inverse += n
        batched_inverse.append(inverse)
    return np.asarray(batched_inverse, dtype=np.int32).reshape(batched_a.shape)


def one_hot_minus(inputs, shift):
    """Performs (inputs - shift) % vocab_size in the one-hot space.
    Args:
        inputs: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
        Tensor.
        shift: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
        Tensor specifying how much to shift the corresponding one-hot vector in
        inputs. Soft values perform a "weighted shift": for example,
        shift=[0.2, 0.3, 0.5] performs a linear combination of 0.2 * shifting by
        zero; 0.3 * shifting by one; and 0.5 * shifting by two.
    Returns:
        Tensor of same shape and dtype as inputs.
    """
    # TODO: Implement with circular conv1d.
    #inputs = torch.tensor(inputs)
    shift = shift.type( inputs.dtype)
    vocab_size = inputs.shape[-1]
    # Form a [..., vocab_size, vocab_size] matrix. Each batch element of
    # inputs will vector-matrix multiply the vocab_size x vocab_size matrix. This
    # "shifts" the inputs batch element by the corresponding shift batch element.
    shift_matrix = torch.stack([torch.roll(shift, i, dims=-1)
                            for i in range(vocab_size)], dim=-2)
    outputs = torch.einsum('...v,...uv->...u', inputs, shift_matrix)
    return outputs

def one_hot_add(inputs, shift):
    """Performs (inputs + shift) % vocab_size in the one-hot space.
    Args:
        inputs: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
        Tensor.
        shift: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
        Tensor specifying how much to shift the corresponding one-hot vector in
        inputs. Soft values perform a "weighted shift": for example,
        shift=[0.2, 0.3, 0.5] performs a linear combination of 0.2 * shifting by
        zero; 0.3 * shifting by one; and 0.5 * shifting by two.
    Returns:
        Tensor of same shape and dtype as inputs.
    """
    inputs = torch.stack((inputs, torch.zeros_like(inputs)), dim = -1)
    shift = torch.stack((shift, torch.zeros_like(shift)), dim = -1)
    if 'torch.fft' not in sys.modules:
        with warnings.catch_warnings(record=True) as w:
            inputs_fft = torch.fft(inputs, 1) #ignore last and first dimension to do batched fft
            shift_fft = torch.fft(shift, 1)
    else:
        inputs_fft = torch.view_as_real(torch.fft.fft(torch.view_as_complex(inputs)))
        shift_fft = torch.view_as_real(torch.fft.fft(torch.view_as_complex(shift)))
    result_fft_real = inputs_fft[...,0]*shift_fft[...,0] - inputs_fft[...,1]*shift_fft[...,1]
    result_fft_imag = inputs_fft[...,0]*shift_fft[...,1] + inputs_fft[...,1]*shift_fft[...,0]
    result_fft = torch.stack((result_fft_real,result_fft_imag), dim = -1)
    if 'torch.fft' not in sys.modules:
        with warnings.catch_warnings(record=True) as w:
            return torch.ifft(result_fft, 1)[...,0] #return only the real part
    else:
        return torch.view_as_real(torch.fft.ifft(torch.view_as_complex(result_fft)))[...,0]

def one_hot_multiply(inputs, scale):
    """Performs (inputs * scale) % vocab_size in the one-hot space.
    Args:
    inputs: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
        Tensor.
    scale: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
        Tensor specifying how much to scale the corresponding one-hot vector in
        inputs. Soft values perform a "weighted scale": for example,
        scale=[0.2, 0.3, 0.5] performs a linear combination of
        0.2 * scaling by zero; 0.3 * scaling by one; and 0.5 * scaling by two.
    Returns:
    Tensor of same shape and dtype as inputs.
    """
    # TODO: Implement with circular conv1d.
    #inputs = torch.tensor(inputs)
    scale = scale.type( inputs.dtype)
    batch_shape = list(inputs.shape[:-1])
    vocab_size = inputs.shape[-1]
    # Form a [..., vocab_size, vocab_size] tensor. The ith row of the
    # batched vocab_size x vocab_size matrix represents scaling inputs by i.
    to_perm = torch.arange(vocab_size).unsqueeze(1).repeat(1, vocab_size) * torch.arange(vocab_size).unsqueeze(0)
    permutation_matrix = one_hot(torch.fmod(to_perm,vocab_size))
    # Scale the inputs according to the permutation matrix of all possible scales.
    scaled_inputs = torch.einsum('...v,avu->...au', inputs, permutation_matrix)
    scaled_inputs = torch.cat( (torch.zeros(batch_shape + [1, vocab_size]),
                                scaled_inputs[..., 1:, :]), dim=-2)
    # Reduce rows of the scaled inputs by the scale values. This forms a
    # weighted linear combination of scaling by zero, scaling by one, and so on.
    outputs = torch.einsum('...v,...vu->...u', scale, scaled_inputs)
    return outputs