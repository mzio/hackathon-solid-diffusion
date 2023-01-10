"""
python main.py --classes 0 4 --samples_per_class 3200 --batch_size 32 --d_kernel 8 --d_kernel_decoder 2 --n_heads 2 --embedding_dim 2 --embedding_type learn_1d --n_positions 784 --criterion_weights 1 1 1000 1 --beta_weight_loss --seed 42 --no_wandb

python main.py --classes 0 4 --samples_per_class 3200 --batch_size 32 --d_kernel 16 --d_kernel_decoder 2 --n_heads 2 --embedding_dim 2 --embedding_type learn_1d --n_positions 784 --criterion_weights 1 1 1000 1 --beta_weight_loss --seed 42 --no_wandb

python main.py --classes 0 4 --samples_per_class 3200 --batch_size 32 --d_kernel 32 --d_kernel_decoder 2 --n_heads 2 --embedding_dim 2 --embedding_type learn_1d --n_positions 784 --criterion_weights 1 1 1000 1 --beta_weight_loss --seed 42 --no_wandb


python main.py --classes 0 4 --samples_per_class 3200 --batch_size 32 --d_kernel 8 --d_kernel_decoder 8 --n_heads 2 --embedding_dim 2 --embedding_type learn_1d --n_positions 784 --criterion_weights 1 1 1000 1 --beta_weight_loss --seed 42 --no_wandb

python main.py --classes 0 4 --samples_per_class 3200 --batch_size 32 --d_kernel 16 --d_kernel_decoder 8 --n_heads 2 --embedding_dim 2 --embedding_type learn_1d --n_positions 784 --criterion_weights 1 1 1000 1 --beta_weight_loss --seed 42 --no_wandb

python main.py --classes 0 4 --samples_per_class 3200 --batch_size 32 --d_kernel 32 --d_kernel_decoder 8 --n_heads 2 --embedding_dim 2 --embedding_type learn_1d --n_positions 784 --criterion_weights 1 1 1000 1 --beta_weight_loss --seed 42 --no_wandb


python main.py --classes 0 4 --samples_per_class 3200 --batch_size 32 --d_kernel 8 --d_kernel_decoder 16 --n_heads 2 --embedding_dim 2 --embedding_type learn_1d --n_positions 784 --criterion_weights 1 1 1000 1 --beta_weight_loss --seed 42 --no_wandb

python main.py --classes 0 4 --samples_per_class 3200 --batch_size 32 --d_kernel 16 --d_kernel_decoder 16 --n_heads 2 --embedding_dim 2 --embedding_type learn_1d --n_positions 784 --criterion_weights 1 1 1000 1 --beta_weight_loss --seed 42 --no_wandb

python main.py --classes 0 4 --samples_per_class 3200 --batch_size 32 --d_kernel 32 --d_kernel_decoder 16 --n_heads 2 --embedding_dim 2 --embedding_type learn_1d --n_positions 784 --criterion_weights 1 1 1000 1 --beta_weight_loss --seed 42 --no_wandb


python main.py --classes 0 1 2 3 4 5 6 7 8 9 --samples_per_class 320 --batch_size 32 --d_kernel 32 --d_kernel_decoder 8 --n_heads 2 --embedding_dim 2 --embedding_type learn_1d --n_positions 784 --criterion_weights 1 1 1000 1 --beta_weight_loss --seed 42 --no_wandb

python main.py --classes 0 1 2 3 4 5 6 7 8 9 --samples_per_class 320 --batch_size 32 --d_kernel 32 --d_kernel_decoder 2 --n_heads 2 --embedding_dim 2 --embedding_type learn_1d --n_positions 784 --criterion_weights 1 1 1000 1 --beta_weight_loss --seed 42 --no_wandb

python main.py --classes 0 1 2 3 4 5 6 7 8 9 --samples_per_class 320 --batch_size 32 --d_kernel 32 --d_kernel_decoder 16 --n_heads 2 --embedding_dim 2 --embedding_type learn_1d --n_positions 784 --criterion_weights 1 1 1000 1 --beta_weight_loss --seed 42 --no_wandb

python main.py --classes 0 1 2 3 4 5 6 7 8 9 --samples_per_class 320 --batch_size 32 --d_kernel 32 --d_kernel_decoder 32 --n_heads 2 --embedding_dim 2 --embedding_type learn_1d --n_positions 784 --criterion_weights 1 1 1000 1 --beta_weight_loss --seed 42 --no_wandb


python main.py --classes 0 1 2 3 4 5 6 7 8 9 --samples_per_class 320 --batch_size 32 --d_kernel 32 --d_kernel_decoder 8 --n_heads 2 --embedding_dim 2 --embedding_type learn_1d --n_positions 784 --beta_weight_loss --seed 42 --no_wandb --criterion_weights 0 0 1 0 --samples_per_class 32 --max_epochs 1000 --ssd_config mnist_1layer --d_kernel 2 --d_kernel_decoder 2 --criterion_weights 0 0 1 0 --max_epochs 1000

python main.py --classes 0 1 2 3 4 5 6 7 8 9 --samples_per_class 320 --batch_size 32 --d_kernel 32 --d_kernel_decoder 16 --n_heads 2 --embedding_dim 2 --embedding_type learn_1d --n_positions 784 --beta_weight_loss --seed 42 --no_wandb --criterion_weights 0 0 1 0 --samples_per_class 32 --max_epochs 1000 --ssd_config mnist_1layer --d_kernel 2 --d_kernel_decoder 2 --criterion_weights 0 0 1 0 --max_epochs 1000
"""
import os
import time
import random
import numpy as np

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

from setup import initialize_args, initialize_network_config
from utils.config import print_config
    
    
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
    args.run_name     = f'dk={args.d_kernel}-dkd={args.d_kernel_decoder}-nh={args.n_heads}-ps={args.patch_size}-ed={args.embedding_dim}-et={args.embedding_type}-cw={criterion_weights}-me={args.max_epochs}-op={args.optimizer}-lr={args.lr}-wd={args.weight_decay}-{diffusion_args}-sd={args.seed}'
        
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
    network_config, network_config_kwargs = initialize_network_config(args)
    model = SSD(**network_config_kwargs)
    print_config(network_config)
    print(model)
    
    # embedding_config = initialize_embedding_config(args)
    # encoder_config   = initialize_encoder_config(args)
    # decoder_config   = initialize_decoder_config(args)
    # ssd_config       = initialize_ssd_config(args)
    
    # model = SSD(embedding_config=embedding_config,
    #             encoder_config=encoder_config,
    #             decoder_config=decoder_config,
    #             ssd_layer_config=ssd_config)
    model.d_kernel = args.d_kernel  # HACKS
    model.d_kernel_decoder = args.d_kernel_decoder
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
        try:
            for layer_ix in range(len(component)):
                print(f'{component_names[ix]}, kernel layer {layer_ix}, closed-loop only: {component[layer_ix].inference_only}, requires_grad: {component[layer_ix].kernel.requires_grad}, d_kernel: {component[layer_ix].kernel.d_kernel}')
        except AttributeError:
            pass
            
    print(f'Model closed-loop only: {model.inference_only}')   
    
    pbar = tqdm(range(args.max_epochs))
    for epoch in pbar:
        # Speed up training heuristic
        if (epoch + 1) % 1 == 0 or epoch == 0:
            data_indices = np.concatenate([np.random.choice(c, size=args.samples_per_class, replace=True)
                                           for c in train_class_indices])
            dataloader = get_resampled_dataloader(train_dataset, data_indices,
                                                  batch_size=len(data_indices),
                                                  shuffle=False)
            assert len(dataloader) == 1
            samples, classes = next(iter(dataloader))
            
            # Get samples and time it
            start_time = time.time()
            # print('** Running forward diffusion process **')
            noisy, means, noise, noise_var = scheduler.get_forward_samples(samples, sampling_timesteps)
            # print(f"-- Epoch: {epoch} | Generating {len(sampling_timesteps) * len(samples)} samples took {time.time() - start_time:3f} seconds --")
            
            # Shuffle
            shuffle_indices = np.arange(len(data_indices))
            # print(shuffle_indices)
            np.random.shuffle(shuffle_indices)
            
            if args.predict_sample is True:  # need to flip them because we flip them later in train loop
                # dataloader = [
                #     (noisy[ix * args.batch_size: (ix + 1) * args.batch_size][..., :args.timesteps-1], 
                #      noisy[ix * args.batch_size: (ix + 1) * args.batch_size][..., 1:args.timesteps],
                #      noise_var[..., :args.timesteps-1])
                #     for ix in range(len(data_indices) // args.batch_size)
                # ]
                # dataloader = [
                #     (noisy[ix * args.batch_size: (ix + 1) * args.batch_size][..., 1:args.timesteps], 
                #      noisy[ix * args.batch_size: (ix + 1) * args.batch_size][..., :args.timesteps-1],
                #      noise_var[..., :args.timesteps-1])
                #     for ix in range(len(data_indices) // args.batch_size)
                # ]
                dataloader = [
                    ((noise * noise_var)[ix * args.batch_size: (ix + 1) * args.batch_size][..., 1:args.timesteps], 
                     means[ix * args.batch_size: (ix + 1) * args.batch_size][..., 1:args.timesteps],
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
    
    
if __name__ == '__main__':
    main()
    
    

    
    

    
    
