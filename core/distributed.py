import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

MAIN_RANK = 0


def is_enabled():
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return get_rank() == 0


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_enabled():
        return 1
    return dist.get_world_size()


def convert_model(model, **kwargs):
    if is_enabled():
        rank = get_rank()
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], output_device=rank, **kwargs)
    else:
        model.cuda()
    return model


def unwrap_model(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def build_dataloader(dataset, **kwargs):
    if is_enabled():
        assert 'sampler' not in kwargs, 'Sampler can not be used in distributed mode!'
        shuffle = kwargs.get('shuffle', False)
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        sampler.set_epoch(1)
        batch_size = kwargs.get('batch_size', 1)
        world_size = get_world_size()
        assert batch_size % world_size == 0, 'Batch size must be divisible by world size!'
        kwargs['batch_size'] = batch_size // world_size
        kwargs['sampler'] = sampler
        kwargs['shuffle'] = False
    loader = DataLoader(dataset, **kwargs)
    return loader


def multi_process_run(func, args, nprocs=1, join=True, host='localhost', port='12355'):
    if nprocs == 1:
        func(0, *args)
    else:
        os.environ['MASTER_ADDR'] = host
        os.environ['MASTER_PORT'] = port
        mp.spawn(func, args=args, nprocs=nprocs, join=join)


def init_process_group(backend, world_size=1, rank=0):
    if world_size == 1:
        return
    dist.init_process_group(backend, world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if dist.is_available() and dist.is_initialized():
        tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output
    else:
        return tensor


def reduce(tensor, dst=0, avg=True):
    if not is_enabled():
        return tensor
    dist.reduce(tensor, dst)
    tensor = tensor / get_world_size() if avg and is_main_process() else tensor
    return tensor


def reduce_dict(tensor_dict, dst=0, avg=True):
    if not is_enabled():
        return tensor_dict
    handle_dict = dict()
    for k, v in tensor_dict.items():
        reduce(v, dst)
        handle_dict[k] = v / get_world_size() if avg and is_main_process() else v
    return handle_dict


def all_reduce(tensor):
    if not is_enabled():
        return None
    return dist.all_reduce(tensor)


def all_reduce_dict(tensor_dict):
    if not is_enabled():
        return None
    handle_dict = dict()
    for k, v in tensor_dict.items():
        handle_dict[k] = all_reduce(v)
    return handle_dict


# def reduce(tensor, average=True):
#     world_size = dist.get_world_size()
#     if world_size < 2:
#         return tensor
#     with torch.no_grad():
#         dist.reduce(tensor, dst=0)
#         if dist.get_rank() == 0 and average:
#             # only main process gets accumulated, so only divide by
#             # world_size in this case
#             tensor = tensor / world_size
#     return tensor


# def reduce_dict(input_dict, average=True):
#     """
#     Reduce the values in the dictionary from all processes so that process with rank
#     0 has the reduced results.

#     Args:
#         input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
#         average (bool): whether to do average or sum

#     Returns:
#         a dict with the same keys as input_dict, after reduction.
#     """
#     if not is_enabled():
#         return input_dict
#     with torch.no_grad():
#         names = []
#         values = []
#         # sort the keys so that they are consistent across processes
#         for k in sorted(input_dict.keys()):
#             names.append(k)
#             values.append(input_dict[k])
#             dist.reduce(input_dict[k], dst=0)
#         if dist.get_rank() == 0 and average:
#             # only main process gets accumulated, so only divide by
#             # world_size in this case
#             values = [v / world_size for v in values]
#         reduced_dict = {k: v for k, v in zip(names, values)}
#     return reduced_dict


# def process_output(output_dict):
#     for k,v in output_dict.items():
#         if len(v.shape) == 0:
