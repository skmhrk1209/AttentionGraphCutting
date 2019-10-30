import torch
import time
import os

class Saver(object):

    def __init__(self, dirname):
        self.dirname = dirname

    def save(self, filename, **kwargs):
        os.makedirs(self.dirname, exist_ok=True)
        torch.save(kwargs, os.path.join(self.dirname, filename))


class StopWatch(object):

    def __init__(self):
        self.stack = []

    def start(self):
        self.stack.append(time.time())

    def stop(self):
        return time.time() - self.stack.pop()


class EMAMeter(dict):

    def __init__(self, momentum=0.9):
        self.momentum = momentum

    def update(self, values1={}, **values2):
        values = {**values1, **values2}
        for key, value in values.items():
            if key in self:
                self[key] *= self.momentum
                self[key] += value * (1 - self.momentum)
            else:
                self[key] = value


class Dict(dict):

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError

    def __setattr__(self, name, value):
        if name in self:
            self[name] = value
        else:
            raise AttributeError

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError


def apply_dict(function, input):
    if isinstance(input, dict):
        return function({key: apply_dict(function, value) for key, value in input.items()})
    elif isinstance(input, list):
        return [apply_dict(function, value) for value in input]
    return input


def to_gpu(input, non_blocking):
    if isinstance(input, torch.Tensor):
        return input.cuda(non_blocking=non_blocking)
    elif isinstance(input, dict):
        return {key: to_gpu(value, non_blocking) for key, value in input.items()}
    elif isinstance(input, list):
        return [to_gpu(value, non_blocking) for value in input]
    return input


def to_cpu(input):
    if isinstance(input, torch.Tensor):
        return input.cpu()
    elif isinstance(input, dict):
        return {key: to_cpu(value) for key, value in input.items()}
    elif isinstance(input, list):
        return [to_cpu(value) for value in input]
    return input
