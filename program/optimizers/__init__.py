import os, sys, time, random, importlib

optimizer_dict = {}

def load_optimizer(optimizer_name):
    return optimizer_dict[optimizer_name].setup_optimizer

def register_optimizer(name):
    def register_optimizer(cls):
        if name in optimizer_dict:
            raise ValueError('Cannot register duplicate optimizer ({})'.format(name))
        optimizer_dict[name] = cls
        return cls
    return register_optimizer

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('program.optimizers.' + module)


