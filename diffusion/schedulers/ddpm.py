"""
Basic linear noise DDPM scheduler

beta_start = 1e-5
beta_end = 1e-2
T = 1e3
scheduler = DDPMScheduler(beta_start, beta_end, T)
"""
import torch
from einops import repeat


class DDPMScheduler():
    def __init__(self, beta_start: float, beta_end: float, T: int):
        assert beta_start < beta_end < 1.0, 'betas must be between 0 and 1, beta_start < beta_end'
        self.beta_start = beta_start
        self.beta_end   = beta_end
        self.T = T
        self.timesteps  = torch.arange(0, self.T, dtype=torch.int)
        
        # Sequence of betas
        self.beta_t = (self.beta_start  # self.T + 1
                       + torch.arange(0, self.T, dtype=torch.float32)
                       * (self.beta_end - self.beta_start) / (self.T - 1))
        self.sqrt_beta_t = torch.sqrt(self.beta_t)
        
        # Sequence of alphas
        self.alpha_t = 1 - self.beta_t
        self.alpha_cumprod_t = torch.cumsum(torch.log(self.alpha_t), dim=0).exp()
        self.one_minus_alpha_cumprod_t = 1 - self.alpha_cumprod_t
        self.alpha_cumprod_tm1 = torch.cat((torch.ones(1), self.alpha_cumprod_t[:-1]))
        
        self.sqrt_alpha_cumprod_t = torch.cumsum(torch.log(self.alpha_t) * 0.5, dim=0).exp()
        self.sqrt_one_minus_alpha_cumprod_t = torch.sqrt(self.one_minus_alpha_cumprod_t)
        
    def step_alpha_cumprod(self, t: int):
        return self.alpha_cumprod_t[t]
    
    def step_alpha(self, t: int):
        return self.alpha_t[t]
    
    def step_beta(self, t: int):
        return self.beta_t[t]
    
    def add_noise(self,
                  samples: torch.FloatTensor,
                  noise: torch.FloatTensor,
                  sample_t: torch.IntTensor
                 ) -> torch.FloatTensor:
        self.sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod_t.to(
            device=samples.device,  dtype=samples.dtype)
        self.sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod_t.to(
            device=samples.device, dtype=samples.dtype)
        
        sample_t = self.timesteps if sample_t is None else sample_t
        # Expand dims for multiplication
        if len(samples.shape) != 5:
            # repeat samples B x C x H x W x T
            samples = repeat(samples, 'b c h w -> b c h w t', t=len(sample_t))
            
        # Noise function should return shape B x C x H x W x T 
        noise = torch.randn_like(samples)
        
        # Get sqrt(a_bar) * sample + sqrt(1 - a_bar) * sample
        weight_input = self.sqrt_alpha_cumprod_t[sample_t]
        weight_noise = self.sqrt_one_minus_alpha_cumprod_t[sample_t]

        # Expand dims for multiplication
        weight_dims = [1] * len(samples.shape)
        weight_dims[-1] = -1

        noisy_samples = (weight_input.view(*weight_dims) * samples + 
                         weight_noise.view(*weight_dims) * noise)
        
        self.sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod_t.cpu()
        self.sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod_t.cpu()

        return noisy_samples, noise, self.beta_t # B x C x H x W x T
    
    def get_forward_samples(self, 
                            samples: torch.FloatTensor,
                            sample_t: torch.IntTensor
                           ) -> (torch.FloatTensor, torch.FloatTensor):
        """
        Returns inputs (noisy_samples) and targets (mean samples) of the forward diffusion process
        """
        
        # Forward samples should be shape B x (sample dims) x T
        sample_t = self.timesteps if sample_t is None else sample_t
        if len(samples.shape) != 5:
            # Hardcode for images, shape: B x C x H x W x T
            samples = repeat(samples, 'b c h w -> b c h w t', t=len(sample_t))
            
        noisy_samples, noise_samples, noise_var = self.add_noise(samples, None, sample_t)
        
        # Compute means -> 1st mean will be 0 matrix
        # (1 / (1 - abar_t)) * sqrt(a_t) (1 - abar_{t-1}) x_t + \sqrt{abar_{t - 1}} * (1 - a_t) x_0
        weight_t = torch.sqrt(self.alpha_t)[sample_t] * (1 - self.alpha_cumprod_tm1)[sample_t]
        weight_0 = torch.sqrt(self.alpha_cumprod_tm1)[sample_t] * self.beta_t[sample_t]
        
        mean_samples = ((1 / self.one_minus_alpha_cumprod_t[sample_t]) * 
                        (weight_t * noisy_samples + weight_0 * samples))
        
        return noisy_samples, mean_samples, noise_samples, noise_var
