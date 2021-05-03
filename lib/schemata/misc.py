####################################
# -- (ADDED) Our helper functions --
####################################
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
from torch_scatter import scatter_add, scatter_max

def normalize_batch(input_batch):
    """
    Normalizes each images in the batch between 0 and 1
    :param input_batch: Tensor[N,C,H,W]
    :return: Tensor[N,C,H,W]
    """
    N, C, H, W = input_batch.size()
    # -- Reshape the batch to be able to calculate the min and max
    # per image in the batch
    reshaped_batch = input_batch.view(N, -1)
    # -- Calculate minimum and maximum per image
    min_per_img = reshaped_batch.min(dim=-1)[0][:, None, None, None]
    max_per_img = reshaped_batch.max(dim=-1)[0][:, None, None, None]
    # -- Normalize the images between 0 and 1
    return (input_batch - min_per_img) / (max_per_img - min_per_img)


def add_module_summary(module, writer, namespace):
    """
    Adds a histogram summary of the provided weights to the tensorboardX
    :param module: The provided module
    :param writer: tensorboardX summary writer
    :param namespace: the provided namespace which is used to divide the histogram sections
    :return: None
    """
    for module_name, module in module.named_modules():
        if isinstance(module, nn.Conv2d):
            writer.add_histogram(f"{namespace}/{module_name}/conv_weights", module.weight)
            if module.bias is not None:
                writer.add_histogram(f"{namespace}/{module_name}/conv_bias", module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            writer.add_histogram(f"{namespace}/{module_name}/bn_weights", module.weight)
            writer.add_histogram(f"{namespace}/{module_name}/bn_bias", module.bias)
        elif isinstance(module, nn.Linear):
            writer.add_histogram(f"{namespace}/{module_name}/dense_weights", module.weight)
            writer.add_histogram(f"{namespace}/{module_name}/dense_bias", module.bias)


def set_random_seed(rnd_seed):
    """
    Fix the random seed among different libraries
    :param rnd_seed: random seed (int)
    :return: None
    """
    # -- Print random seed
    print(f"Set random seed to: {rnd_seed}")

    # -- Set different libraries random seed
    random.seed(rnd_seed)
    np.random.seed(rnd_seed)
    torch.manual_seed(rnd_seed)

    # -- Set CUDA's random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rnd_seed)
        torch.cuda.manual_seed_all(rnd_seed)

    # -- Disable CUDNN
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def remove_params(state_dict, parameters):
    """
    Remove the provided paramteres' weights from the model
    :param state_dict: Checkpoints' state dictionary
    :param parameters: Parameters to be removed
    :return:
    """
    if len(parameters) == 0:
        return

    # -- (ADDED) added a visual separator for better readability
    print("\n==================================\n"
          "Removing unnecessary parameters...\n")

    for param in parameters:
        if param in state_dict:
            state_dict.pop(param)
            print("Successfully removed parameter:", param)
        else:
            print("Couldn't remove parameter:", param)

    # -- (ADDED) added a visual separator for better readability
    print("==================================\n")


def xavier_init(module, gain=1.0):
    """
    Takes a nn.Linear module and initializes it's weights
    with Xavier normal function (PyTorch <=0.4.0)
    :param module: The provided nn.Linear module
    :return: Initialized nn.Linear module
    """
    assert isinstance(module, nn.Linear)
    module.weight = torch.nn.init.xavier_normal_(module.weight, gain=gain)
    return module


def sparse_softmax(src: torch.Tensor, index: torch.LongTensor, dim: int = 0, num_nodes=None,
                   skip_torchscatter=False) -> torch.Tensor:
    r"""Computes a sparsely evaluated softmax. MaxB
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    if num_nodes is None:
        num_nodes = index.max() + 1
    # Torch scatter library might be trouble to install. This is a local implementation in case that was the case.
    if skip_torchscatter:
        zeros_vec = torch.zeros(src[:, 0].shape[0], index.max() + 1, device=index.device)
        b = torch.where(index[:, None] == torch.arange(index.max() + 1, device=index.device))
        zeros_vec[b] = 1
        masked_src = zeros_vec.repeat((src.shape[1], 1, 1)) * src.transpose(1, 0)[:, :, None]
        masked_src[masked_src == 0] = torch.finfo().min
        scattered_max = masked_src.max(1)[0].transpose(1, 0).index_select(index=index, dim=dim)
    else:
        scattered_max = scatter_max(src, index, dim=dim, dim_size=num_nodes)[0].index_select(index=index, dim=dim)

    src = src - scattered_max
    src = src.exp()

    if skip_torchscatter:
        sct_add = torch.zeros((index.max() + 1), 5, device=index.device)
        sct_added = sct_add.index_add_(dim, index, src).index_select(index=index, dim=dim) + 1e-16
    else:
        sct_added = (scatter_add(src, index, dim=dim, dim_size=num_nodes).index_select(index=index, dim=dim) + 1e-16)
    src = src / sct_added
    return src
