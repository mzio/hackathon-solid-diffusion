import torch
import os
from os.path import join


def format_arg(arg):
    if '_' in arg:
        try:
            abbrev = ''.join([str(a[0]) for a in arg.split('_')])
        except:
            print(arg)
            abbrev = arg
    else:
        abbrev = arg[:3]
    return abbrev


def init_experiment(config, args, prefix='s4_simple-d=arima'):
    args.device = (torch.device('cuda:0') if torch.cuda.is_available()
                   else torch.device('cpu'))
    args.experiment_name = prefix
    dataset_name = 'arima'
    for k, v in config.dataset.items():
        if k not in ['_name_', 'val_gap', 'test_gap', 'seed', 'seasonal']:
            _k = format_arg(k)
            if isinstance(v, bool):
                v = int(v)
            args.experiment_name += f'-{_k}={v}'
            dataset_name += f'-{_k}={v}'
    for k, v in config.model.items():
        if k not in ['defaults', '_name_', 'prenorm', 
                     'transposed', 'pool', 'layer',
                     'residual', 'dropout', 'norm']:
            _k = format_arg(k)
            if isinstance(v, bool):
                v = int(v)
            args.experiment_name += f'-{_k}={v}'
    for k, v in config.model.layer.items():
        if k not in ['_name_', 'channels', 'bidirectional',
                     'postact', 'initializer', 'weight_norm', 
                     'dropout', 'dt_min', 'dt_max']:
            _k = format_arg(k)
            if isinstance(v, bool):
                v = int(v)
            args.experiment_name += f'-{_k}={v}'
            
    args.experiment_name += f'-std={int(args.no_standardize)}'  
    args.experiment_name += f'-ed={int(args.encoder_decoder)}' 
    args.experiment_name += f'-gtc={int(args.ground_truth_c)}'
    args.experiment_name += f'-s={args.seed}'
    args.best_train_metric = 1e10  # RMSE
    args.best_val_metric = 1e10  # RMSE
    
    # Dataset setup
    dataset_dir = join(args.checkpoint_path, dataset_name)
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    args.checkpoint_path = dataset_dir
    
    args.best_train_checkpoint_path = join(
        args.checkpoint_path,
        f'best_train_ckpt-{args.experiment_name}.pth')
    args.best_val_checkpoint_path = join(
        args.checkpoint_path,
        f'best_val_ckpt-{args.experiment_name}.pth')
    
    return args.experiment_name
            
    