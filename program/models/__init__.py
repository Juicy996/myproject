import os, sys, time, random, importlib
import torch

# ---
# 模型有两种输出模式：rep+extra_output 和 ret_dict
# rep的格式为 batchsize，seqlen，hiddendim

model_dict = {}

def load_model(model_name):
    return model_dict[model_name].setup_model()

def register_model(name):
    def register_model(cls):
        if name in model_dict:
            raise ValueError('Cannot register duplicate model ({})'.format(name))
        model_dict[name] = cls
        return cls
    return register_model

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('program.models.' + module)

