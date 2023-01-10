# General Setup
import os
import sys
import copy
import time

import random

from os.path import join
from copy import deepcopy

import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import opt_einsum as oe
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm

from diffusion.schedulers import DDPMScheduler, get_equal_steps
from diffusion.utils.visualize import visualize_diffusions, get_display_image, display_image


# HuggingFace Imports
from datasets import load_dataset
from diffusers import AutoencoderKL  # , UNet2DConditionModel, LMSDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer

auth_token = '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'


from omegaconf import OmegaConf
from utils.config import print_config

# SSD MODEL
from model.ssd.embeddings import PatchEmbedding
from model.ssd.embeddings.position import LearnedPositionEmbedding  # get_position_embedding
from model.ssd.encoders import *
from model.ssd.kernels.companion import CompanionKernel
from model.functional.krylov import krylov

from einops import rearrange, repeat


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def pil_to_tensor(pil_image):
    """
    Convert 4 channel PIL image to 3 channel tensor
    """
    to_tensor_transform = transforms.ToTensor()
    tensor_image = to_tensor_transform(pil_image)
    if (tensor_image.shape[0] == 4 and 
        (tensor_image[-1, ...] == torch.ones(tensor_image.shape[1:])).sum().item()
        == np.prod(tensor_image.shape[1:])):
        tensor_image = tensor_image[:-1, ...]
    return tensor_image


# TRAIN INITIAL
config = """
max_epochs: 100
data:
  image_size: 512
  train_batch_size: 4
  eval_batch_size: 4  # How many images to sample during eval
optimizer:
  name: Adam
  learning_rate: 1e-4
  weight_decay: 5e-5
logging:
  save_image_epochs: 10
  save_model_epochs: 20
seed: 42
no_gpu: false 
"""
config = OmegaConf.create(config)
config.device = ('cuda:0' 
                 if torch.cuda.is_available() and config.no_gpu is False
                 else 'cpu')

# DATA PREPROCESSING
config_autoencoder = """
autoencoder:
  mean: 0.5
  std: 0.5
"""
config_autoencoder = OmegaConf.create(config_autoencoder)
config = OmegaConf.merge(config, config_autoencoder)

image_transform = transforms.Compose(
    [
        transforms.Resize((config.data.image_size, 
                           config.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([config.autoencoder.mean],
                             [config.autoencoder.std])
    ]
)

def get_transform(samples, image_transform, 
                  tokenizer, text_encoder,
                  guided=True):
    # Process images
    images = [image_transform(image.convert("RGB")) for image in samples["image"]]

     # Process text captions
    with torch.no_grad():
        text_embeddings = []
        for text in samples['text']:
            text_input = tokenizer(text, padding='max_length', max_length=tokenizer.model_max_length,
                                   truncation=True, return_tensors='pt')
            text_embedding = text_encoder(text_input.input_ids)[0]

            if guided:
                max_length = text_input.input_ids.shape[-1]
                unconditional_input = tokenizer(
                    [""] * 1,  # config.train_batch_size, 
                    padding='max_length', 
                    max_length=max_length, return_tensors='pt'
                )
                unconditional_embedding = text_encoder(unconditional_input.input_ids)[0]
                text_embedding = torch.cat([unconditional_embedding, text_embedding])

            # print(text_embedding.shape) 
            # [D x clip_token_size x clip_embed_dim]
            # D is 2 if guided else 1
            text_embeddings.append(text_embedding)

    return {"image": images, 'text': samples['text'], 'text_embedding': text_embeddings}


# AUTOENCODER
def image_encode(images, autoencoder, return_stat, device):
    autoencoder.to(device)
    with torch.no_grad():
        latents = autoencoder.encode(images.to(device))
    autoencoder.cpu()
    
    # Autoencoder is a VAE
    if return_stat == 'mode':
        try:
            return 0.18215 * latents.mode() # or .mean or .sample
        except:
            return 0.18215 * latents.latent_dist.mode()
    elif return_stat == 'mean':
        try:
            return 0.18215 * latents.mean() # or .mean or .sample
        except:
            return 0.18215 * latents.latent_dist.mean()
    elif return_stat == 'sample':
        try:
            return 0.18215 * latents.sample() # or .mean or .sample
        except:
            return 0.18215 * latents.latent_dist.sample()
        
def tensor_to_image(tensors, image_mean, image_std):
    images = (tensors * image_std + image_mean).clamp(0, 1)
    images = images.detach().cpu() * 255
    return images

def image_decode(latents, autoencoder, image_mean, image_std, device):
    autoencoder.to(device)
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        latents = latents.to(device)
        images = autoencoder.decode(latents).sample
        latents = latents.cpu()
    autoencoder.cpu()
    
    images = (images * image_std + image_mean).clamp(0, 1)
    images = images.detach().cpu() * 255 
    return images

def image_decode_to_pil(latents, **kwargs):
    images = image_decode(latents, **kwargs)
    images = images.permute(0, 2, 3, 1).numpy().round().astype('uint8')
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def image_to_pil(images):
    images = images.permute(0, 2, 3, 1).numpy().round().astype('uint8')
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

# DIFFUSION
config_diffusion = """
diffusion_scheduler:
  name: DDPM
  kwargs:
    beta_start: 1e-5
    beta_end: 1e-2
    T: 1000
"""
config_diffusion = OmegaConf.create(config_diffusion)



# MODEL
# Unpatch input
def unpatch(x, patch_embedding, patch_pattern='b (t l) (c n)'): 
    b, _, d = x.shape 
    x = rearrange(x, f'{patch_pattern} -> b t c n l',
                  b=b, 
                  c=patch_embedding.n_channels,
                  n=patch_embedding.n_patches, 
                  l=patch_embedding.patch_len)
    t = x.shape[1]
    x = rearrange(x, 'b t c n l -> (b t) (c l) n')
    x = patch_embedding.fold(x)
    x = rearrange(x, '(b t) c h w -> b c h w t', b=b, t=t)
    return x


def patch_input(noise, patch_embedding, bidirectional, 
                    rearrange_pattern='b c n t l -> b (t l) (c n)'):
    # assume noise is B x C x H x W x T
    noise = patch_embedding(noise)
    if bidirectional:
        noise = rearrange([noise, torch.flip(noise, [4])],
                          'r b c n t l -> b c (n r) t l')
    return rearrange(noise, rearrange_pattern)


class ShiftKernel(CompanionKernel):
    """
    
    """
    def __init__(self, n_hidden_state, **kwargs):
        self.n_hidden_state = n_hidden_state
        self.target_length = None
        super().__init__(**kwargs)
        
    def init_weights(self):
        # A matrix ix fixed shift matrix
        A = torch.zeros(self.n_channels, self.n_heads, 
                        self.d_kernel, self.d_kernel)
        A[:, :, 1:, :-1] = torch.eye(self.d_kernel - 1)
        self.register("A", A, trainable=False, lr=None, wd=None)
        
        # B matrix is fixed
        b    =  torch.zeros(self.d_kernel).float()
        b[0] = 1.
        b    = repeat(b, 'd -> c h d', 
                      c=self.n_channels, h=self.n_heads).clone().contiguous()
        self.register("b", b, trainable=False, lr=None, wd=None)
        
        # C, K, D matrices are trainable by default
        trainable = True if self.requires_grad is True else False
        
        # Hack for now
        if self.closed_loop:
            k = torch.randn(*self._fp)
            self.register("k", k, trainable, lr=None, wd=None)
        else:
            c = torch.randn(*self._fp)
            self.register("c", c, trainable, lr=None, wd=None)
        
        # D matrix (skip connection) is trainable
        d = torch.randn(self.n_channels, self.n_heads)
        self.register("d", d, trainable, lr=None, wd=None)
    
    def convolve(self, u, A, l, b, c):
        """
        if self.closed_loop:
        - compute: \sum_{i=0}^{t - 1} c(A + bc)^{t - 1 - i}b * u[i]
        else:
        - compute: \sum_{i=0}^{t - 1} cA^{t - 1 - i}b * u[i]
        """
        f = krylov(l, A, b, c)
        y = self.fft_conv(u, f)
        y = rearrange(y, 'b c h l -> b (c h) l')
        return y
    
    def norm(self, x, ord=1):
        # x.shape = C x H x D
        x_norm = torch.linalg.norm(x, ord=ord, dim=2, keepdim=True)
        x = x / x_norm
        return x
    
    def forward(self, u):
        ch, h, d = self.b.shape
        b,  d, l = u.shape  # assume u is B x D x L
        
        # Inference
        if self.target_length is not None: l = self.target_length
        
        if self.closed_loop:  # Assume u already has the noise terms coded in
            # Hacks for stabilization right now
            k = self.norm(self.k.clamp(1 / 16, 1), ord=1)  # Not sure where 16 is coming from
            BK = oe.contract('c h i, c h j -> c h i j', self.b, k)
            
            # 1st computes cb, c(A + BK)b, ..., c(A + BK)^(l-1)b
            # -> f = krylov(l, self.norm(self.A + BK), b=self.b, c=self.k) 
            # Then compute convolution: 
            # -> y = f * u
            y = self.convolve(u, self.norm(self.A + BK), l, self.b, self.c)
        else:
            y = self.convolve(u, self.A, l, self.b, self.c)
            
        return y, None
    
    
# Update get_encoder method
def get_encoder(encoder_config):
    if encoder_config['type'] == 'repeat':
        encoder = RepeatEncoder
    elif encoder_config['type'] == 'dense':
        encoder = DenseEncoder
    elif encoder_config['type'] == 'convolution':
        encoder = ConvEncoder
    elif encoder_config['type'] == 'identity':
        encoder = Encoder  # identity
    else:
        print(encoder_config['type'])
        raise NotImplementedError
    return encoder(**encoder_config['kwargs'])


def get_decoder(decoder_config):
    # Same classes for both
    return get_encoder(decoder_config)



class SSDLayer(nn.Module):
    def __init__(self, 
                 kernel: dict,
                 decoder: dict, 
                 skip_connection: bool,
                 closed_loop: bool):
        super().__init__()
        self.kernel_args     = kernel
        self.decoder_args    = decoder
        self.skip_connection = skip_connection
        self.closed_loop     = closed_loop
        
        self.kernel  = self.get_kernel()
        self.decoder = self.get_decoder()
        
        self.inference_only = False
        
        # Bug with shape matching
        assert self.skip_connection is False
    
    def get_kernel(self):
        if self.kernel_args['type'] == 'companion':
            kernel_class = CompanionKernel
        elif self.kernel_args['type'] == 'shift':
            kernel_class = ShiftKernel
        else:
            # TODO: implement more kernels
            raise NotImplementedError(f'Error: {kernel_name} not implemented')
        return kernel_class(**self.kernel_args['kwargs'])
    
    def get_decoder(self):
        return get_decoder(self.decoder_args)      
        
    def forward(self, u):  # 
        # Assume input shape is (B, L, H)
        u = rearrange(u, 'b l h -> b h l')
        y, _ = self.kernel(u)  # could output multiple, so should modify this
        y = rearrange(y, 'b h l -> b l h')
        y = self.decoder(y)
        return y
    
    
class OurSequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input
    
    
class SSD(nn.Module):
    def __init__(self, config, noise_schedule, noise_stride=1):
        super().__init__()
        self.encoder_config = config.encoder
        self.decoder_config = config.decoder
        self.rollout_config = config.rollout
        
        self.noise_schedule = noise_schedule  # e.g., betas
        self.noise_stride   = noise_stride    # How many to time-steps to apply same beta
        
        self.encoder, self.decoder, self.rollout = self.init_layers()
        
    def init_layers(self):
        encoder = []
        decoder = []
        rollout = []
        
        for ix, layer_config in enumerate(self.encoder_config):
            encoder.append(SSDLayer(**layer_config.layer))
        for ix, layer_config in enumerate(self.decoder_config):
            decoder.append(SSDLayer(**layer_config.layer))
        for ix, layer_config in enumerate(self.rollout_config):
            rollout.append(SSDLayer(**layer_config.layer))
            
        encoder = nn.Sequential(*encoder)
        decoder = nn.Sequential(*decoder)
        rollout = nn.Sequential(*rollout)
        
        return encoder, decoder, rollout
        
    def set_inference_length(self, length: int):
        """
        Use during evaluation to rollout up to specified length
        """
        for ix in range(len(self.ssd_decoder)):
            self.rollout[ix].kernel.target_length = length
            
    def set_inference_only(self, mode: bool=False):
        """
        Use during evaluation to only go through rollout branch
        """
        self.inference_only = mode
        self.requires_grad  = not mode  # Not implemented
        
    def sample_noise(self, z, noise_schedule):
        # z is shape B x L x D
        noise = torch.randn_like(z)
        var   = repeat(noise_schedule, 'l -> (l r)', r=self.noise_stride)
        noise = oe.contract('b l d, l -> b l d', noise, var)
        return noise
        
    def compute_rollout(self, z):
        # z is shape B x L x D
        noise = self.sample_noise(z, self.noise_schedule).to(z.device)
        
        # Replace first noise_stride terms with first noise_stride terms of z
        mask = torch.zeros(z.shape).to(z.device).type(z.type)
        mask[:, :self.noise_stride, :] = 1.
        z = (z * mask) + (noise * (1. - mask))
        
        # Compute rollout with closed-loop SSM
        z = self.rollout(z)
        
        # Sample outputs as mu + sigma * noise
        # z += self.sample_noise(z, self.noise_schedule[1:], 0)
        z[:, :-1, :] += noise[:, 1:, :]
        return z
        
    def forward(self, u):
        # u is shape B x L x D
        z = self.encoder(u)
        # Compute closed-loop rollout
        z_rollout = self.compute_rollout(z)
        # rollout is a prediction for future samples, so keep first input sample
        z_rollout = torch.cat([z[:, :1, :], z_rollout[:, :-1, :]], dim=1)
        y_rollout = self.decoder(z_rollout)
        
        if not self.inference_only:
            # During training, can also compute outputs from available inputs
            y = self.decoder(z)
        else:
            y = None
            
        return y_rollout, y
    
    
class ModelEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed = PatchEmbedding(**config.patch.kwargs)
        # Hardcoded
        self.position_embed = LearnedPositionEmbedding(**config.position.kwargs)
        self.bidirectional = config.bidirectional
        
    def forward(self, x):
        x = self.patch_embed(x).unsqueeze(-1)  # hard-coded, shape is now B x C x N x T x L x D
        x += self.position_embed(x)
        if self.bidirectional:
            x_r = torch.flip(x, [4])
            x = rearrange([x, x_r], 'r b c n t l d -> b c (n r) t l d')
        x = rearrange(x, 'b c n t l d -> b (t l) (c n d)')
        return x
    
    
class StateSpaceDiffusionModel(nn.Module):
    def __init__(self,
                 embedding_config: dict,
                 encoder_config: dict,
                 decoder_config: dict,
                 ssd_config: dict,
                 noise_schedule,
                 noise_stride: int):
        super().__init__()

        self.embedding_config = embedding_config
        self.encoder_config   = encoder_config
        self.decoder_config   = decoder_config
        self.ssd_config       = ssd_config
        self.noise_schedule   = noise_schedule
        self.noise_stride     = noise_stride
        
        # Hard-coded for now
        self.input_embed  = ModelEmbedding(self.embedding_config)
        # print('a')
        self.input_encode = get_encoder(self.encoder_config)
        # print('b')
        self.input_decode = get_decoder(self.decoder_config)
        # print('c')
        self.ssd          = SSD(ssd_config, noise_schedule, noise_stride)
        # print('d')
        
        
    def forward(self, x, x_=None):
        x = self.input_embed(x)
        x = self.input_encode(x)
        # First is closed-loop, second is open-loop 
        x, x_ = self.ssd(x)
        x = self.input_decode(x)
        if not self.ssd.inference_only:
            x_ = self.input_decode(x_)
        return x, x_

        
class SSDModel(StateSpaceDiffusionModel):
    # Alternative reference
    def __init__(self, 
                 embedding_config: dict,
                 encoder_config: dict,
                 decoder_config: dict,
                 ssd_config: dict,
                 noise_schedule,
                 noise_stride: int):
        super().__init__(embedding_config,
                         encoder_config,
                         decoder_config,
                         ssd_config,
                         noise_schedule,
                         noise_stride)
        
        
# TRAINING
def train(model, autoencoder, epoch, **kwargs):
    model.train()
    return run_epoch(model, autoencoder, True, epoch, **kwargs)
    
    
def evaluate(model, autoencoder, epoch, **kwargs):
    model.eval()
    model.set_inference_only(mode=True)
    
    with torch.no_grad():
        return run_epoch(model, autoencoder, False, epoch, **kwargs)
    
    
def run_epoch(model, autoencoder, train, epoch, dataloader, scheduler, 
              optimizer, criterion, beta_weight_loss, device):
    
    model.zero_grad()
    pbar = tqdm(dataloader, leave=False)
    
    T = scheduler.T
    
    for ix, data in enumerate(pbar):
        images = data['image']
        texts = data['text']
        text_embeddings = data['text_embedding']

        # Encode inputs
        encoded_inputs = image_encode(data['image'], autoencoder, return_stat='mode', device=device)
        encoded_inputs = encoded_inputs.cpu()

        # Run forward diffusion on latent inputs
        start = time.time()
        noisy, means, noise, noise_var = scheduler.get_forward_samples(encoded_inputs, 
                                                                       np.arange(T))  # hard code   
        end = time.time()
        print(f"-- Generating {T * len(noisy)} samples took {end - start:3f} seconds --")

        # Reverse diffusion ordering
        u_input = torch.flip(noisy, [-1]) 
        u_noise = torch.flip(noise, [-1])
        u_beta  = torch.flip(noise_var, [-1])
        v_beta  = 1. / u_beta
        y_means = torch.flip(means, [-1])
        
        model.to(device)
        u_input = u_input.to(device)
        v_beta = v_beta.to(device)
        
        # c is closed loop, o is open loop
        start = time.time()
        y_c, y_o = model(u_input)
        end = time.time()
        
        # y_c = model.input_embedding.unpatch(y_c)
        y_c = unpatch(y_c, patch_embedding) # fix this
        loss_c = criterion(y_c[..., 16:], y_means[..., 16:])  # fix hardcoded
        
        if train and not model.ssd.inference_only:
            y_o = unpatch(y_o, patch_embedding) # fix this
            loss_o = criterion(y_o[..., 16:], y_means[..., 16:])
        else:
            loss_o = 0
            
        all_losses = [loss_c, loss_o]
        loss_names = ['closed-loop', 'open-loop']
        loss = loss_c + loss_o
        if beta_weight_loss:
            loss = oe.contract('b c h w t, t -> b c h w t',
                               loss, v_beta[..., 16:])
        else:
            loss = loss.mean()
            
        if train:
            loss.backward()
            optimizer.step()
            model.zero_grad()
            
        loss_desc = ' | '.join([f'{loss_names[ix]}: {all_losses[ix]:.3f}'
                                for ix in range(len(all_losses))])
        
        pbar_desc = f'Batch: {batch_ix}/{len(dataloader)} | {loss_desc}'
        pbar.set_description(pbar_desc)
        
        if not train and ix == 0:
            all_decoded_images = []
            steps = [99, 199, 299, 399, 499, 599, 699, 799, 899, 999]
            n_samples = y_c.shape[0]
            n_steps = len(steps)
            for step in steps:
                decoded_inputs = image_decode(y_c[..., step], autoencoder,
                                              config.autoencoder.mean,
                                              config.autoencoder.std, 
                                              device)
                decoded_inputs = decoded_inputs.cpu()
                decoded_images = image_to_pil(decoded_inputs)
                all_decoded_images.append(decoded_images)
               
            
            # print(f'Sample at step {step}, beta_{step}: {u_beta[step]}')
            # fig, axis = plt.subplots(10, 4, figsize=(16, 40))
            fig, axis = plt.subplots(n_steps, n_samples, 
                                     figsize=(n_samples * 4, n_steps * 4))
            
            for i in range(n_samples):
                for step_ix, step in enumerate(steps):
                    axis[plot_ix, sample_ix].imshow(
                        all_decoded_images[step_ix, i]
                    )
                    axis[plot_ix, sample_ix].set_axis_off()
                    
            plt.show()
            fig.savefig(f'azuki_generations-epoch={epoch:03d}.png')
        return model, loss
    

def main():
    # LOAD PRETRAINED MODELS
    
    # Stable Diffusion
    checkpoint_path_sd = '/dfs/scratch1/common/pretrained_models/stable-diffusion-v1-4'
    model_id_diffusion_autoencoder = 'CompVis/stable-diffusion-v1-4'

    # Load the autoencoder model which will be used to decode the latents into image space. 
    vae = AutoencoderKL.from_pretrained(model_id_diffusion_autoencoder, subfolder="vae", use_auth_token=auth_token,
                                        cache_dir=join(checkpoint_path_sd, 'vae'))
    vae.save_pretrained(join(checkpoint_path_sd, 'vae'))
    
    # CLIP
    checkpoint_path_clip = '/dfs/scratch1/common/pretrained_models/clip'
    model_id_clip = 'openai/clip-vit-large-patch14'

    # Load the tokenizer and text encoder to tokenize and encode the text. 
    tokenizer = CLIPTokenizer.from_pretrained(model_id_clip, use_auth_token=auth_token,
                                              cache_dir=checkpoint_path_clip)
    text_encoder = CLIPTextModel.from_pretrained(model_id_clip, use_auth_token=auth_token,
                                                 cache_dir=checkpoint_path_clip)
    
    # LOAD DATA
    config.dataset_name = '/dfs/scratch1/mzhang/projects/solid-diffusion/data/azuki/image'
    config.dataset_dir = '/dfs/scratch1/common/public-datasets/azuki'
    dataset = load_dataset(config.dataset_name, split="train", 
                           cache_dir=config.dataset_dir)
    
    default_transform = lambda x: get_transform(x, image_transform, 
                                                tokenizer, text_encoder, 
                                                guided=True) 
    dataset.set_transform(default_transform)
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=config.data.train_batch_size, 
                                               shuffle=True)
    eval_loader = torch.utils.data.DataLoader(dataset, batch_size=config.data.eval_batch_size, 
                                              shuffle=False)
    
    # DIFFUSION
    scheduler = DDPMScheduler(**config_diffusion.diffusion_scheduler.kwargs)
    T = config_diffusion.diffusion_scheduler.kwargs.T
    
    
    # MODEL
    embedding_config = """
    patch:
      kwargs:
        patch_size: 4
        dilation: 1
        padding: 0
        stride: 4
    position:
      kwargs:
        n_positions: 256
        output_dim: 1
    bidirectional: true
    """
    embedding_config = OmegaConf.create(embedding_config)

    input_encoder_config = f"""
    type: repeat
    kwargs:
      input_dim: 1024
      output_dim: {1024 * 2}
      input_shape: bld
    """
    input_encoder_config = OmegaConf.create(input_encoder_config)


    input_decoder_config = f"""
    type: dense
    kwargs:
      input_dim: {1024} 
      output_dim: 1024
      activation: gelu
      n_layers: 2
      n_activations: 1
      input_shape: bld
    """
    input_decoder_config = OmegaConf.create(input_decoder_config)
    
    ssd_encoder_config = f"""
    encoder:
    - layer:
        kernel:
          type: companion
          kwargs:
            d_kernel: 16
            n_heads: {1024 * 2}
            n_channels: 1
            skip_connection: true
            closed_loop: false
            train: true
        decoder:
          type: dense
          kwargs:
            input_dim: {1024 * 2}
            output_dim: {1024 * 1}
            activation: gelu
            n_layers: 2
            n_activations: 1
        skip_connection: false
        closed_loop: false
    - layer:
        kernel:
          type: companion
          kwargs:
            d_kernel: 16
            n_heads: {1024 * 1}
            n_channels: 1
            skip_connection: true
            closed_loop: false
            train: true
        decoder:
          type: dense
          kwargs:
            input_dim: {1024 * 1}
            output_dim: {512}
            activation: gelu
            n_layers: 2
            n_activations: 1
        skip_connection: false
        closed_loop: false
    """
    ssd_encoder_config = OmegaConf.create(ssd_encoder_config)
    
    ssd_decoder_config = f"""
    decoder:
    - layer:
        kernel:
          type: companion
          kwargs:
            d_kernel: 16
            n_heads: {512}
            n_channels: 1
            skip_connection: true
            closed_loop: false
            train: true
        decoder:
          type: dense
          kwargs:
            input_dim: {512}
            output_dim: {1024}
            activation: gelu
            n_layers: 2
            n_activations: 1
        skip_connection: false
        closed_loop: false
    - layer:
        kernel:
          type: companion
          kwargs:
            d_kernel: 16
            n_heads: {1024}
            n_channels: 1
            skip_connection: true
            closed_loop: false
            train: true
        decoder:
          type: identity
          kwargs:
            input_dim: {1024}
            output_dim: {1024 * 2}
        skip_connection: false
        closed_loop: false
    """
    ssd_decoder_config = OmegaConf.create(ssd_decoder_config)
    
    ssd_rollout_config = f"""
    rollout:
    - layer:
        kernel:
          type: shift
          kwargs:
            d_kernel: 16
            n_heads: {512}
            n_channels: 1
            n_hidden_state: 1
            skip_connection: false
            closed_loop: true
            train: true
        decoder:
          type: identity
          kwargs:
            input_dim: {512}
            output_dim: {512}
        skip_connection: false
        closed_loop: True
    """
    ssd_rollout_config = OmegaConf.create(ssd_rollout_config)
    
    ssd_config = OmegaConf.merge(ssd_encoder_config, ssd_decoder_config, ssd_rollout_config)
    
    seed_everything(config.seed)
    criterion = nn.MSELoss(reduction='none')

    u_beta = torch.flip(scheduler.beta_t, [0])

    model = SSDModel(embedding_config,
                     input_encoder_config,
                     input_decoder_config,
                     ssd_config,
                     u_beta,  
                     noise_stride=16)

    optim_config = {
        'lr': 1e-4,   
        'weight_decay': 5e-5
    }
    optimizer = Adam(model.parameters(), **optim_config)


    T = config_diffusion.diffusion_scheduler.kwargs.T

    train_config = {
        'dataloader': train_loader,  # Fill-in / update below 
        'scheduler': scheduler,
        'optimizer': optimizer, 
        'criterion': torch.nn.MSELoss(reduction='none'), 
        # 'criterion_weights': args.criterion_weights,  # [1., 1., 10., 10.], 
        'beta_weight_loss': True,
        'device': torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    }

    # Training
    epoch_pbar = tqdm(range(100))
    for epoch in epoch_pbar:

        if (epoch + 1) % 10 == 0:
            train(model, vae, epoch, **train_config)
            evaluate(model, vae, epoch, **train_config)
        else:
            train(model, vae, epoch, **train_config)
            
            
if __name__ == '__main__':
    main()