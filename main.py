"""
python main.py --classes 0 4 --samples_per_class 3200 --batch_size 32 --d_kernel 8 --n_heads 2 --embedding_dim 2 --embedding_type learn_1d --n_positions 784 --criterion_weights 1 1 1000 1
"""
import os
import time
import random
import numpy as np
import argparse

from copy import deepcopy
from tqdm.auto import tqdm

import torch
import torchvision
from torchvision import transforms
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from diffusion.schedulers import DDPMScheduler
from model.ssd.network import SSD
from train import train, evaluate


def initialize_args():
    parser = argparse.ArgumentParser(description='Solid Diffusion')
    parser.add_argument('--wandb_entity', type=str, default='mzhang')
    
    parser.add_argument('--max_epochs', type=int, default=100)
    
    # Dataloader - assume MNIST for now
    parser.add_argument('--classes', nargs='+', default=None)
    parser.add_argument('--samples_per_class', type=int, default=320)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    
    # Diffusion
    parser.add_argument('--diffusion_scheduler', type=str, default='ddpm')
    parser.add_argument('--beta_start', type=float, default=1e-5)
    parser.add_argument('--beta_end', type=float, default=1e-2)
    parser.add_argument('--timesteps', type=float, default=1000)
    ## Ours
    parser.add_argument('--beta_weight_loss', action='store_true', default=False)
    parser.add_argument('--predict_sample', action='store_true', default=False)
                        
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--criterion_weights', nargs='+')
    
    # Model
    parser.add_argument('--d_kernel', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=2)  # Should be >= args.embedding_dim
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--embedding_dim', type=int, default=2)
    parser.add_argument('--embedding_type', type=str, default='learn_1d',
                        choices=['learn_1d', 'sinusoid_1d'])
    parser.add_argument('--n_positions', type=int, default=None,
                        help='MNIST: --n_positions 28 * 28')
    
    # Saving
    parser.add_argument('--data_dir', type=str, default='/dfs/scratch1/mzhang/projects/slice-and-dice-smol/datasets/data/')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--generation_dir', type=str, default='./images')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--replicate', type=int, default=0)
    parser.add_argument('--no_wandb', action='store_true', default=False)
    args = parser.parse_args()
    return args
    
    
def init_wandb(args):
    if not args.no_wandb:
        import wandb
        classes = '_'.join(args.classes)
        wandb.init(config={'lr': args.lr},
                   entity=args.wandb_entity,
                   name=args.run_name,
                   project=args.project_name,
                   dir=args.log_dir)
        wandb.config.update(args)
    else:
        wandb = None
    
    
def main():
    args = initialize_args()
    seed_everything(args.seed)
    classes = '_'.join(args.classes)
    criterion_weights = '_'.join(args.criterion_weights)
    diffusion_args = f'dsc={args.diffusion_scheduler}-beta=({args.beta_start}_{args.beta_end})-ts={args.timesteps}-bwl={int(args.beta_weight_loss)}-pds={int(args.predict_sample)}'
    args.criterion_weights = [float(w) for w in args.criterion_weights]
    args.project_name = f'ssd-d={classes}-spc={args.samples_per_class}'
    args.run_name     = f'kd={args.d_kernel}-nh={args.n_heads}-ps={args.patch_size}-ed={args.embedding_dim}-et={args.embedding_type}-cw={criterion_weights}-me={args.max_epochs}-op={args.optimizer}-lr={args.lr}-wd={args.weight_decay}-{diffusion_args}-sd={args.seed}'
        
    init_wandb(args)
    
    # LOAD DATA
    print('** LOAD DATA **')
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root=args.data_dir, train=True,  
                                               transform=transform, download=False)
    test_dataset  = torchvision.datasets.MNIST(root=args.data_dir, train=False, 
                                               transform=transform, download=False)
    train_class_indices = [np.where(train_dataset.targets == int(t))[0] for t in args.classes]
    test_class_indices  = [np.where(test_dataset.targets == int(t))[0] for t in args.classes]
    # print(f'Total training samples: {train_class_indices}')
    print(f'Random sample: {np.random.choice(train_dataset.targets[np.concatenate(train_class_indices)], size=20)}')
    
    # LOAD FORWARD DIFFUSION SAMPLER
    if args.diffusion_scheduler == 'ddpm':    
        scheduler = DDPMScheduler(args.beta_start, args.beta_end, args.timesteps)
    else:
        raise NotImplementedError
    sampling_timesteps = np.arange(args.timesteps)
    
    # dataloader_config = {
    #     'batch_size': args.batch_size,
    #     'num_workers': args.num_workers,
    # }
    
    # LOAD MODEL
    print('** LOAD MODEL **')
    embedding_config = initialize_embedding_config(args)
    encoder_config   = initialize_encoder_config(args)
    decoder_config   = initialize_decoder_config(args)
    ssd_config       = initialie_ssd_config(args)
    
    model = SSD(embedding_config=embedding_config,
                encoder_config=encoder_config,
                decoder_config=decoder_config,
                ssd_layer_config=ssd_config)
    print_args(args)
    
    # LOAD OPTIMIZER
    optim_config = {
        'lr': args.lr,   
        'weight_decay': args.weight_decay
    }
    optimizer = Adam(model.parameters(), **optim_config)

    # TRAINING
    train_config = {
        'dataloader': None,  # Fill-in / update below 
        'optimizer': optimizer, 
        'criterion': torch.nn.MSELoss(reduction='none'), 
        'criterion_weights': args.criterion_weights,  # [1., 1., 10., 10.], 
        'beta_weight_loss': args.beta_weight_loss,
        'device': torch.device('cuda:0') if torch.cuda.is_available() else False
    }
    
    if train_config['criterion_weights'] == [0, 0, 1, 0]:
        model.set_inference_only(mode=True)  # HACK
    component_names = ['SSD Encoder', 'SSD Decoder']
    for ix, component in enumerate([model.ssd_encoder, model.ssd_decoder]):
        for layer_ix in range(len(component)):
            print(f'{component_names[ix]}, kernel layer {layer_ix}, closed-loop only: {component[layer_ix].inference_only}, requires_grad: {component[layer_ix].kernel.requires_grad}')
    print(f'Model closed-loop only: {model.inference_only}')   
    
    pbar = tqdm(range(args.max_epochs))
    for epoch in pbar:
        # Speed up training heuristic
        if (epoch + 1) % 10 == 0 or epoch == 0:
            data_indices = np.concatenate([np.random.choice(c, size=args.samples_per_class, replace=True)
                                           for c in train_class_indices])
            dataloader = get_resampled_dataloader(train_dataset, data_indices,
                                                  batch_size=len(data_indices),
                                                  shuffle=False)
            assert len(dataloader) == 1
            samples, classes = next(iter(dataloader))
            
            # Get samples and time it
            start_time = time.time()
            print('** Running forward diffusion process **')
            noisy, means, noise, noise_var = scheduler.get_forward_samples(samples, sampling_timesteps)
            print(f"-- Epoch: {epoch} | Generating {len(sampling_timesteps) * len(samples)} samples took {time.time() - start_time:3f} seconds --")
            
            if args.predict_sample is True:  # need to flip them because we flip them later in train loop
                # dataloader = [
                #     (noisy[ix * args.batch_size: (ix + 1) * args.batch_size][..., :args.timesteps-1], 
                #      noisy[ix * args.batch_size: (ix + 1) * args.batch_size][..., 1:args.timesteps],
                #      noise_var[..., :args.timesteps-1])
                #     for ix in range(len(data_indices) // args.batch_size)
                # ]
                dataloader = [
                    (noisy[ix * args.batch_size: (ix + 1) * args.batch_size][..., 1:args.timesteps], 
                     noisy[ix * args.batch_size: (ix + 1) * args.batch_size][..., :args.timesteps-1],
                     noise_var[..., :args.timesteps-1])
                    for ix in range(len(data_indices) // args.batch_size)
                ]
                        
            else:
                dataloader = [
                    (noisy[ix * args.batch_size: (ix + 1) * args.batch_size], 
                     means[ix * args.batch_size: (ix + 1) * args.batch_size],
                     noise_var)
                    for ix in range(len(data_indices) // args.batch_size)
                ]
        else:
            shuffle_indices = np.arange(len(data_indices))
            # print(shuffle_indices)
            np.random.shuffle(shuffle_indices)
            # print(shuffle_indices)
            # hacky because randomness compounds
            _noisy = noisy[shuffle_indices]
            _means = means[shuffle_indices]
            # _noise_var = noise_var[shuffle_indices]
                        
            if args.predict_sample:
                dataloader = [
                    (_noisy[ix * args.batch_size: (ix + 1) * args.batch_size][..., 1:args.timesteps], 
                     _noisy[ix * args.batch_size: (ix + 1) * args.batch_size][..., :args.timesteps-1],
                     noise_var[..., :args.timesteps-1])
                    for ix in range(len(data_indices) // args.batch_size)
                ]
            else:
                dataloader = [
                    (_noisy[ix * args.batch_size: (ix + 1) * args.batch_size], 
                     _means[ix * args.batch_size: (ix + 1) * args.batch_size],
                     noise_var)
                    for ix in range(len(data_indices) // args.batch_size)
                ]
            
        train_config['dataloader'] = dataloader
        if epoch == 0:
            pbar.set_description(f'Epoch: {epoch}')

        else:
            train_loss_dict, n_batches_train = results_train
            # test_loss_dict,  n_batches_test  = results_test

            train_loss_desc = ' | '.join([f'{k}: {v / n_batches_train:.3f}' for k, v in train_loss_dict.items()])  
            # test_loss_desc  = ' | '.join([f'{k}: {v / n_batches_test:.3f}'  for k, v in test_loss_dict.items()])  
            pbar.set_description(f'Epoch: {epoch} | {train_loss_desc}')
            
        model, *results_train = train(model, **train_config)
        train_loss_dict, n_batches_train = results_train
        if (epoch == 0 or 
            train_loss_dict['y_ct'] < best_train_loss_dict['y_ct']):
            best_train_loss_dict = deepcopy(train_loss_dict)
            save_dict = {'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict()}
            for k, v in best_train_loss_dict.items():
                save_dict[k] = v
            torch.save(save_dict, os.path.join(
                args.checkpoint_dir, f'{args.project_name}-{args.run_name}.pt'))
            
    
def print_args(args):
    attributes = [a for a in dir(args) if a[:1] != '_']
    print('ARGPARSE ARGS')
    for ix, attr in enumerate(attributes):
        fancy = '└──' if ix == len(attributes) - 1 else '├──'
        print(f'{fancy} {attr}: {getattr(args, attr)}')
        
        
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def get_resampled_dataloader(dataset, sample_indices, **kwargs):
    dataset_rs = deepcopy(dataset)
    dataset_rs.data = dataset_rs.data[sample_indices]
    dataset_rs.targets = dataset_rs.targets[sample_indices]
    return DataLoader(dataset_rs, **kwargs)
    
    
def initialize_embedding_config(args):
    embedding_config = {
        'patch': {
            'patch_size': args.patch_size,
            'dilation': 1,
            'padding': 0,
            'stride': args.patch_size
        },
        'linear': {
            'input_dim': args.patch_size ** 2,
            'output_dim': (args.patch_size ** 2, args.embedding_dim),
            'identity_init': True,
            'identity_val': 1
        },
        'position': {
            'type': args.embedding_type,  
            'n_positions': args.n_positions,  # h x w; or None for inferring
            'embedding_dim': args.embedding_dim,
            'kwargs': {}  # Ignore for now
        },
        'bidirectional': False,
        'output_shape': 'bt(cnl)'
    }
    return embedding_config
    
    
def initialize_encoder_config(args):
    encoder_config = {
        'type': 'repeat',
        'kwargs': {
            'input_dim': 784, # Number of patches * 2 if bidirectinoal
            'output_dim': 784 * args.n_heads,  # Input dim * number of 1st layer heads
            'input_shape': 'bld'
        }
    }
    return encoder_config

def initialize_decoder_config(args):
    decoder_config = {
        'type': 'dense',
        'kwargs': {
            'input_dim': 784,   # Last SSD layer output dim
            'output_dim': 784,  # Number of patches
            'activation': 'gelu',
            'n_layers': 2,
            'n_activations': 1,
            'input_shape': 'bld',
        }
    }
    return decoder_config

def initialie_ssd_config(args):
    ssd_config = {
        '0': {
            'kernel': {
                'type': 'companion',
                'kwargs': {
                    'd_kernel': 4,
                    'n_heads': 784 * args.n_heads * 1,  # 4? (number patches) * (number heads) * (2 if bidirectional else 1)
                    'n_channels': 1,  # Ignore this for now
                    'skip_connection': True,
                    'closed_loop': False,
                    'train': True,
                }
            },
            'decoder': {
                'type': 'dense',
                'kwargs': {
                    'input_dim': args.n_heads * 784,
                    'output_dim': 784,
                    'activation': 'gelu',
                    'n_layers': 2,
                    'n_activations': 1,
                }
            },
            'skip_connection': False,
            'closed_loop': False  # redundant?
        },
        '1': {
            'kernel': {
                'type': 'companion',
                'kwargs': {
                    'd_kernel': 4,
                    'n_heads': 784,
                    'n_channels': 1,
                    'skip_connection': True,
                    'closed_loop': False,
                    'train': True,
                }
            },
            'decoder': {
                'type': 'dense',
                'kwargs': {
                    'input_dim': 784,
                    'output_dim': 784 // 2,
                    'activation': 'gelu',
                    'n_layers': 2,
                    'n_activations': 1,
                }
            },
            'skip_connection': False,
            'closed_loop': False  # redundant?
        },
        '2': {
            'kernel': {
                'type': 'shift',
                'kwargs': {
                    'd_kernel': 4,
                    'n_heads': 784 // 2,
                    'n_channels': 1,
                    'skip_connection': False,
                    'closed_loop': True,
                    'train': True,
                    'n_hidden_state': 1
                }
            },
            'decoder': {
                'type': 'dense',
                'kwargs': {
                    'input_dim': 784 // 2,
                    'output_dim': 784,
                    'activation': 'gelu',
                    'n_layers': 2,
                    'n_activations': 1,
                }
            },
            'skip_connection': False,
            'closed_loop': True  # redundant?
        }
    }
    return ssd_config
    
    
if __name__ == '__main__':
    main()
    
    