from os.path import join
from omegaconf import OmegaConf


def initialize_network_config(args, config_dir='configs'):
    # Assume args has args.embedding_config, args.encoder_config, args.decoder_config, args.ssd_config
    
    embedding_conf = OmegaConf.load(join(config_dir, 'embedding', f'{args.embedding_config}.yaml'))
    encoder_conf   = OmegaConf.load(join(config_dir, 'encoder', f'{args.encoder_config}.yaml'))
    decoder_conf   = OmegaConf.load(join(config_dir, 'decoder', f'{args.decoder_config}.yaml'))
    ssd_conf       = OmegaConf.load(join(config_dir, 'ssd', f'{args.ssd_config}.yaml'))
    
    update_embedding_config(embedding_conf, args)
    update_encoder_config(encoder_conf, args)
    update_decoder_config(decoder_conf, args)
    update_ssd_config(ssd_conf, args)
    
    network_conf = OmegaConf.merge(embedding_conf, encoder_conf, decoder_conf, ssd_conf)
    network_conf_kwargs = {'embedding_config': embedding_conf, 
                           'encoder_config': encoder_conf,
                           'decoder_config': decoder_conf,
                           'ssd_layer_config': ssd_conf}
    return network_conf, network_conf_kwargs
    
    
def update_embedding_config(config, args):
    # Update patching
    config.patch.patch_size = args.patch_size
    config.patch.stride = args.patch_size  # default is no overlap
    
    # Update linear mapping
    if args.no_linear:
        config.linear = None
    else:
        config.linear.input_dim = args.patch_size ** 2
        config.linear.output_dim = (args.patch_size ** 2, 
                                    args.embedding_dim)
    # Update position embeddings
    if args.no_position:
        config.position = None
    else:
        config.position.type = args.embedding_type
        config.position.n_positions = args.n_positions
        config.position.embedding_dim = args.embedding_dim


def update_encoder_config(config, args):
    config.kwargs.output_dim = config.kwargs.output_dim * args.n_heads


def update_decoder_config(config, args):
    pass


def update_ssd_config(config, args):
    # Can refactor with YAML list
    for idx in config.layers.keys():
        if int(idx) == len(config.layers.items()) - 1:
            config.layers[idx].kernel.kwargs.d_kernel = args.d_kernel_decoder
        else:
            config.layers[idx].kernel.kwargs.d_kernel = args.d_kernel
        if int(idx) == 0:  # Updated
            config.layers[idx].kernel.kwargs.n_heads = config.layers[idx].kernel.kwargs.n_heads * args.n_heads
            config.layers[idx].decoder.kwargs.input_dim = config.layers[idx].decoder.kwargs.input_dim * args.n_heads

            