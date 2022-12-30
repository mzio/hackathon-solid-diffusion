"""
Save results to a single CSV
- Make one per evaluation dataset 
"""

import os
from os.path import join

import numpy as np
import pandas as pd


def print_header(x, border='both'):
    print('-' * len(x))
    print(x)
    print('-' * len(x))
    
    
def print_args(args):
    attributes = [a for a in dir(args) if a[0] != '_']
    print('ARGPARSE ARGS')
    for ix, attr in enumerate(attributes):
        fancy = '└──' if ix == len(attributes) - 1 else '├──'
        print(f'{fancy} {attr}: {getattr(args, attr)}')
        
# Control how tqdm progress bar looks        
def type_of_script():
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'



def make_csv(args, save_dir='./logs', file_prefix='results', column_names=None):
    if column_names is None:
        column_names = ['dataset', 'horizon', 'lag', 'method', 'network_config', 
                        'replicate', 'seed', 'task_norm', 'scale', 'max_epochs', 'best_val_metric_epoch',
                        'train_mse_informer', 'train_mae_informer',
                        'val_mse_informer', 'val_mae_informer',
                        'test_mse_informer', 'test_mae_informer',
                        'train_mse_transformed', 'train_mae_transformed',
                        'val_mse_transformed', 'val_mae_transformed',
                        'test_mse_transformed', 'test_mae_transformed',
                        'train_mse', 'train_mae',
                        'val_mse', 'val_mae',
                        'test_mse', 'test_mae',
                        'hparam_lr', 'hparam_weight_decay', 
                        'hparam_model_dropout', 'hparam_batch_size',
                        'hparam_optimizer', 'hparam_scheduler']
    
    if args.dataset in ['etth', 'ettm']:
        dataset = args.dataset + str(args.variant)
    else:
        dataset = args.dataset
        
    if args.features != 'S':
        dataset += '-f=M'
    fpath = join(save_dir, f'{file_prefix}-d={dataset}.csv')
    
    if os.path.exists(fpath):
        print(f'Great! Logging path already exists!')
        print(f'Logging to {fpath}...')
        pass
    else:
        # Inefficient way to do it but consistent
        df = pd.DataFrame({k: [] for k in column_names})
        df.sort_index(axis=1).to_csv(fpath)
        print(f'Great! Created new logging path!')
        print(f'Logging to {fpath}...')
        
    return fpath, column_names


def save_results(split_metrics, column_names, args, fpath, method='SpaceTime'):
    save_dict = {}
    added_columns = {k: False for k in column_names}
    
    # Save metrics
    for split, metrics in split_metrics.items():
        for metric_name, metric in metrics.items():
            if f'{split}_{metric_name}' in column_names:
                try:
                    save_dict[f'{split}_{metric_name}'] = [metric.cpu().item()]
                except:
                    save_dict[f'{split}_{metric_name}'] = [metric]
                added_columns[f'{split}_{metric_name}'] = True
    
    # Save rest of experiment details
    save_dict['method'] = [method]
    save_dict['dataset'] = [args.dataset] if args.dataset not in ['etth', 'ettm1'] else [args.dataset + str(args.variant)]
    added_columns['method'] = True
    added_columns['dataset'] = True
    
    dict_args = vars(args)
    
    if args.loss != 'rmse':
        dict_args['network_config'] = f'{args.network_config}-loss={args.loss}'
    
    for column in column_names:
        if added_columns[column] is False:
            try:
                save_dict[column] = [dict_args[column]] if column[:6] != 'hparam' else [dict_args[column[7:]]]
            except Exception as e:
                print(e)
                
                if args.bash is False:
                    breakpoint()
                    
            try:
                if column == 'replicate':
                    save_dict[column] = [str(dict_args[column]) + '-mnorm'] if args.memory_norm == 1 else [str(dict_args[column])]
            except:
                pass
    print(save_dict)
    
    pd.DataFrame(save_dict).sort_index(axis=1).to_csv(fpath, mode='a', header=False)
    print(f'Saved results to {fpath}!')
        
    
    
                          
        
        
        


