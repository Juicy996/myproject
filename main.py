import os, sys, time, random, torch, config
import numpy as np
from program import trainer

def seed_torch(seed = 1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.enabled = False
    print(f'Random seed was set with {seed}.')

def main():
    args = config.load_params()

    if args.seed > 0: 
        seed = args.seed
        seed_torch(seed)
    else:
        seed = random.randint(1, 999999999)
        seed_torch(seed)
    
    config_file = 'program/configs/{}.json'.format(args.config_file)
    hp, template_config, dataset_config_set, model_config_set, criterion_config_set, \
        optimizer_config, trainer_config = config.load_configs(config_file, args.config_file)

    hp.seed = seed
    _trainer = trainer.trainer(hp, 
        template_config,
        dataset_config_set,
        model_config_set,
        criterion_config_set,
        optimizer_config,
        trainer_config,
        resume = args.resume,
        )
    _trainer()

if __name__ == '__main__': 
    main()










