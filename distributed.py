import torch
from torch import distributed
import socket
import os


def init_process_group(backend):

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()

    info = {}
    if rank == 0:
        host = socket.gethostname()
        address = socket.gethostbyname(host)
        info.update(MASTER_ADDR=address, MASTER_PORT='1234')

    info = comm.bcast(info, root=0)
    info.update(WORLD_SIZE=str(world_size), RANK=str(rank))
    os.environ.update(info)

    distributed.init_process_group(backend=backend)


def average_gradients(parameters):
    world_size = distributed.get_world_size()
    for parameter in parameters:
        if parameter.requires_grad:
            distributed.all_reduce(parameter.grad)
            parameter.grad /= world_size


def average_tensors(tensors):
    world_size = distributed.get_world_size()
    for tensor in tensors:
        distributed.all_reduce(tensor)
        tensor /= world_size


def broadcast_tensors(tensors, src_rank=0):
    for tensor in tensors:
        distributed.broadcast(tensor, src_rank)


def all_gather(tensor):
    tensors = [torch.empty_like(tensor) for rank in range(distributed.get_world_size())]
    distributed.all_gather(tensors, tensor)
    return tensors


def dprint(*args, rank=0):
    if distributed.get_rank() == rank:
        print(*args)
