import torch
from torch import multiprocessing
from torch import distributed
from torch import backends
from torch import cuda
from torch import utils
from torch import optim
from torch import nn
from torchvision import models
from torchvision import transforms
from torchvision import utils as vutils
from tensorboardX import SummaryWriter
from chainercv import evaluations
from collections import OrderedDict
from PIL import Image
from modules import *
from samplers import *
from distributed import *
from utils import *
import visualization
import numpy as np
import itertools
import functools
import importlib
import argparse
import datetime
import shutil
import random
import json
import time
import glob
import os


class MultiMNIST(utils.data.Dataset):

    def __init__(self, metafile, transform=None, target_transform=None):
        with open(metafile) as file:
            self.meta = list(json.load(file).items())
        self.dirname = os.path.dirname(metafile)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        filename, target = self.meta[index]
        image = Image.open(filename).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


class MNISTModel(nn.Module):

    def __init__(self, conv_params, attention_param, linear_params):
        super().__init__()
        self.network = nn.Sequential(OrderedDict(
            conv_blocks=nn.Sequential(*[
                nn.Sequential(OrderedDict(
                    conv=nn.Conv2d(**conv_param),
                    actv=nn.ReLU()
                )) for conv_param in conv_params
            ]),
            attention_network=AttentionNetwork(**attention_param),
            linear_blocks=nn.Sequential(*[
                nn.Sequential(
                    nn.Identity() if i else nn.ReLU(),
                    nn.Linear(**linear_param)
                ) for i, linear_param in enumerate(linear_params)
            ])
        ))

    def forward(self, input):
        output = self.conv_blocks(input)
        output, attention = self.attention_network(output)
        output = self.linear_blocks(output)
        return output, attention


def main(args):

    with open(args.config) as file:
        config = json.load(file)
        config.update(vars(args))
        config = apply_dict(Dict, config)

    # Multi-process single-GPU distributed training
    # See https://pytorch.org/docs/1.1.0/distributed.html
    # and https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel

    # On PyTorch, we should specify `MASTER_ADDR` and `MASTER_PORT` by environment variable.
    init_process_group(backend='nccl')  # For PyTorch
    # On Parrots, we don't have to specify them.
    # distributed.init_process_group(backend='nccl') # For Parrots

    # Force each process to run on a single device.
    cuda.set_device(distributed.get_rank() % cuda.device_count())

    # NOTE: Using fork method causes an error in a data loader.
    multiprocessing.set_start_method('spawn', force=True)

    backends.cudnn.enabled = True
    backends.cudnn.benchmark = False

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    cuda.manual_seed(config.seed)

    config.dataset.update(global_batch_size=config.dataset.local_batch_size * distributed.get_world_size())

    dprint(f'\n{"=" * 32} Configuration {"=" * 32}')
    dprint(json.dumps(config, indent=4))

    train_dataset = MultiMNIST(
        metafile=config.dataset.train.metafile,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3)
        ]),
        target_transform=lambda target: {key: torch.tensor(value) for key, value in target.items()}
    )
    val_dataset = MultiMNIST(
        metafile=config.dataset.val.metafile,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3)
        ]),
        target_transform=lambda target: {key: torch.tensor(value) for key, value in target.items()}
    )

    # Just run 1 iteration for debug.
    if config.debug:
        indices = range(config.dataset.global_batch_size)
        train_dataset = utils.data.Subset(train_dataset, indices)
        eval_datasets = Dict({name: utils.data.Subset(val_dataset, indices) for name, val_dataset in eval_datasets.items()})

    # Sampler for distributed training.
    # This guarantees that each process loads a different batch in each training step.
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_data_loader = utils.data.DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=config.dataset.local_batch_size,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        drop_last=False
    )
    val_data_loader = utils.data.DataLoader(
        dataset=val_dataset,
        sampler=val_sampler,
        batch_size=config.dataset.local_batch_size,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        drop_last=False
    )

    model = MNISTModel(
        conv_params=[
            Dict(in_channels=3, out_channels=32, stride=2),
            Dict(in_channels=32, out_channels=64, stride=2),
        ],
        attention_param=Dict(
            conv_param=[
                Dict(in_channels=64, out_channels=32, stride=2),
                Dict(in_channels=32, out_channels=16, stride=2),
            ],
            linear_params=[
                Dict(in_features=1024, out_features=64),
                Dict(in_features=64, out_features=1024),
            ],
            deconv_params=[
                Dict(in_channels=16, out_channels=8, stride=2),
                Dict(in_channels=8, out_channels=4, stride=2),
            ]
        ),
        linear_params=[
            Dict(in_features=256, out_features=1024),
            Dict(in_features=1024, out_features=10),
        ]
    )
    model.cuda()

    num_process_groups = distributed.get_world_size() // config.distributed.batch_norm_group_size
    process_groups = [distributed.new_group(ranks) for ranks in np.split(np.arange(distributed.get_world_size()), num_process_groups)]
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_groups[distributed.get_rank() // config.distributed.batch_norm_group_size])
    model = nn.parallel.DistributedDataParallel(model, [distributed.get_rank() % cuda.device_count()], broadcast_buffers=False)

    # Scale learning rate following the `global` batch size (`local batch size` * `world size`)
    config.optimizer.lr *= config.global_batch_size / config.global_batch_denom
    optimizer = optim.Adam(model.parameters(), **config.optimizer)

    epoch = -1
    step = -1
    if config.saving.resume_model:
        checkpoint = Dict(torch.load(config.saving.resume_model, map_location=lambda storage, location: storage.cuda()))
        model.load_state_dict(checkpoint.model_state_dict, strict=True)
        optimizer.load_state_dict(checkpoint.optimizer_state_dict, strict=True)
        epoch = checkpoint.epoch
        step = checkpoint.step

    writer = SummaryWriter('logs') if not distributed.get_rank() else None
    saver = Saver('ckpts') if not distributed.get_rank() else None

    stop_watch = StopWatch()
    ema_meter = Dict(EMAMeter())

    def train(data_loader):
        nonlocal step
        dprint(f'\n{"=" * 32} Training started {"=" * 32}')
        model.train()
        stop_watch.start()
        for step, (input, target) in enumerate(data_loader, step + 1):
            input = to_gpu(input, non_blocking=True)
            ema_meter.update(data_time=stop_watch.stop())
            stop_watch.start()
            logit, attention = model(input)
            loss = nn.functional.cross_entropy(logit, target)
            prediction = torch.argmax(logit, dim=-1)
            accuracy = torch.mean(prediction == target)
            ema_meter.update(forward_time=stop_watch.stop())
            stop_watch.start()
            optimizer.zero_grad()
            if torch.isnan(loss):
                dprint('NaN in the loss...')
            elif torch.isinf(loss):
                dprint('Inf in the loss...')
            else:
                loss.backward(retain_graph=False)
            if not isinstance(model, nn.parallel.DistributedDataParallel):
                average_gradients(model.parameters())
            optimizer.step()
            ema_meter.update(backward_time=stop_watch.stop())
            stop_watch.start()
            if not step % config.training.log_steps:
                average_tensors([loss, accuracy])
                if writer:
                    writer.add_scalar(f'loss/train', loss, step)
                    writer.add_scalar(f'accuracy/train', accuracy, step)
                    writer.add_image(f'attention/train', vutils.make_grid(visualization.linear_map(attention, attention.min(), attention.max(), 0, 1)), step)
                progress = step / (config.training.train_epochs * len(data_loader)) * 100
                eta_seconds = (config.training.train_epochs * len(data_loader) - step) * sum(ema_meter.values())
                eta_string = str(datetime.timedelta(seconds=eta_seconds))
                dprint(f'\n[training] epoch: {epoch} progress: {progress:.2f}% ETA: {eta_string} loss: {loss} accuracy: {accuracy}')
                dprint(f' '.join(f"{name}: {time:.4f} sec" for name, time in ema_meter.items()))
            distributed.barrier()
        stop_watch.stop()

    if config.train:
        stop_watch.start()
        broadcast_tensors(model.state_dict().values())
        for epoch in range(epoch + 1, config.training.train_epochs):
            train_sampler.set_epoch(epoch)
            train(train_data_loader)
            if saver:
                saver.save(
                    filename=f'epoch_{epoch}',
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    epoch=epoch,
                    step=step
                )
        dprint(f'\n{"=" * 32} Training finished {"=" * 32}')
        dprint(f'Elapsed time: {stop_watch.stop()} sec')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Jigsaw Puzzle')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(args)
