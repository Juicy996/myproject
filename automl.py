"""
A deep MNIST classifier using convolutional layers.
This file is a modification of the official pytorch mnist example:
https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import os, json, argparse, logging, nni, torch, main
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nni.utils import merge_parameter
from torchvision import datasets, transforms

logger = logging.getLogger('mnist_AutoML')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("NNI.log")
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        # logger.debug(tuner_params)
        # logger.debug(torch.cuda.device_count())
        config_dict = {'hyper_params': {}, "template_config_set": {}, "dataset_config_set": {}, "model_config_set": {}, "criterion_config_set": {}}
        for k, v in tuner_params.items():
            if 'hyper_params' in k:
                config_dict['hyper_params'][k.split('#')[-1]] = v
            elif 'template_config_set' in k:
                stand = k.split('#')
                if len(stand) == 2:
                    config_dict['template_config_set'][k.split('#')[-1]] = v
                else:
                    template_name, template_key = stand[1:]
                    if not template_name in config_dict["template_config_set"]:
                        config_dict["template_config_set"][template_name] = {}
                    config_dict["template_config_set"][template_name][template_key] = v
            elif 'dataset_config_set' in k:
                dataset_name, dataset_key = k.split('#')[1:]
                if not dataset_name in config_dict["dataset_config_set"]:
                    config_dict["dataset_config_set"][dataset_name] = {}
                config_dict["dataset_config_set"][dataset_name][dataset_key] = v
            elif 'model_config_set' in k:
                model_name, model_key = k.split('#')[1:]
                if not model_name in config_dict["model_config_set"]:
                    config_dict["model_config_set"][model_name] = {}
                config_dict["model_config_set"][model_name][model_key] = v
            elif 'criterion_config_set' in k:
                criterion_name, criterion_key = k.split('#')[1:]
                if not criterion_name in config_dict["criterion_config_set"]:
                    config_dict["criterion_config_set"][criterion_name] = {}
                config_dict["criterion_config_set"][criterion_name][criterion_key] = v
        config_dict = json.dumps(config_dict, indent = 2)
        with open('program/configs/automl.json','w', encoding = 'utf-8') as f:
            f.write(config_dict)
        # logger.debug(config_dict)
        # print(params)
        main.main()
    except Exception as exception:
        # print(exception)
        logger.debug(exception)
        raise