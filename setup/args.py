import argparse


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
    ## Base configs
    parser.add_argument('--embedding_config', type=str, default='mnist')
    parser.add_argument('--encoder_config', type=str, default='mnist_repeat')
    parser.add_argument('--decoder_config', type=str, default='mnist_dense')
    parser.add_argument('--ssd_config', type=str, default='mnist')
    
    ## Overriding args
    parser.add_argument('--d_kernel', type=int, default=4)
    parser.add_argument('--d_kernel_decoder', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=2)  # Should be >= args.embedding_dim
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--embedding_dim', type=int, default=2)
    parser.add_argument('--embedding_type', type=str, default='learn_1d',
                        choices=['learn_1d', 'sinusoid_1d'],
                        help='positional embedding type')
    parser.add_argument('--n_positions', type=int, default=None,
                        help='MNIST: --n_positions 28 * 28')
    parser.add_argument('--no_linear', action='store_true', default=False)
    parser.add_argument('--no_position', action='store_true', default=False)
    
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