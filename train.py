import torch
import time
from tqdm.auto import tqdm

import opt_einsum as oe


def train(model, **kwargs):
    model.train()
    return run_epoch(model, True, **kwargs)
    
    
    
def evaluate(model, **kwargs):
    model.eval()
    model.set_inference_only(mode=True)
    
    with torch.no_grad():
        return run_epoch(model, False, **kwargs)


def run_epoch(model, train, dataloader, optimizer, criterion, criterion_weights, beta_weight_loss, device):
        
    total_loss = {'y_co': 0, 'z_co': 0, 'y_ct': 0, 'y_ot': 0}
    pbar = tqdm(dataloader, leave=False)
    
    # data is zip(inputs, means)
    for batch_ix, data in enumerate(pbar):
        u, y, v = data
        
        # Get correct reverse process ordering
        u = torch.flip(u, [-1])  # B x C x H x W x T
        y = torch.flip(y, [-1])  # B x C x H x W x T
        # v = 2 * torch.abs(torch.flip(v, [-1]))  # hack
        v = 1. / torch.flip(v, [-1])  # T

        model.to(device)
        u = u.to(device)
        y = y.to(device)
        v = v.to(device)

        # c is closed loop, o is open loop
        start = time.time()
        y_c, y_o, z_c, z_o = model(u)
        end = time.time()
        
        if train and not model.inference_only:  # Compute 4 MSE Losses
            # Alignment between closed-loop and open-loop computations
            loss_y_co = criterion(y_c[:, model.d_kernel_decoder-1:, :], y_o[:, model.d_kernel_decoder-1:, :])
            loss_z_co = criterion(z_c[:, model.d_kernel_decoder-1:, :], z_o[:, model.d_kernel_decoder-1:, :])
            
            if beta_weight_loss:
                loss_y_co = oe.contract('b l d, l -> b l d', 
                                        loss_y_co, v[model.d_kernel_decoder-1:]).mean()
                loss_z_co = oe.contract('b l d, l -> b l d', 
                                        loss_z_co, v[model.d_kernel_decoder-1:]).mean()
            else:
                loss_y_co = loss_y_co.mean()
                loss_z_co = loss_z_co.mean()
            
            # loss_z_co = criterion(z_c[:, 1:, :], z_o[:, 1:, :])
            
        else:
            loss_y_co = torch.zeros(1).to(u.device)
            loss_z_co = torch.zeros(1).to(u.device)
            
        # Alignment between predicted and ground-truth outputs
        start_unpatch = time.time()
        y_c = model.input_embedding.unpatch(y_c)
        end_unpatch = time.time()
        
        loss_y_ct = criterion(y_c[..., model.d_kernel_decoder-1:], y[..., model.d_kernel_decoder-1:])  # d_state - 1
        if beta_weight_loss:
            loss_y_ct = oe.contract('b c h w t, t -> b c h w t', 
                                    loss_y_ct, v[model.d_kernel_decoder-1:]).mean()
        else:
            loss_y_ct = loss_y_ct.mean()
        
        if not model.inference_only:
            y_o = model.input_embedding.unpatch(y_o)      
            loss_y_ot = criterion(y_o[..., model.d_kernel_decoder-1:], y[..., model.d_kernel_decoder-1:])
            
            if beta_weight_loss:
                loss_y_ot = oe.contract('b c h w t, t -> b c h w t', 
                                        loss_y_ot, v[model.d_kernel_decoder-1:]).mean()
            else:
                loss_y_ot = loss_y_ot.mean()
        else:
            loss_y_ot = torch.zeros(1).to(u.device)
        
        # if loss_y_ct.item() < 1e-1:
        #     print(y_c[0, 0, 0, 0, -4:])
            # print(y[0, 0, 0, 0, -4:])

        all_losses = [loss_y_co, loss_z_co, loss_y_ct, loss_y_ot]
        loss = sum([criterion_weights[ix] * all_losses[ix] 
                    for ix in range(len(all_losses))])
        
        start_backprop = time.time()
        if train:
            loss.backward()
            optimizer.step()
            model.zero_grad()
        end_backprop = time.time()
        
        for ix, k in enumerate(total_loss.keys()):
            total_loss[k] += all_losses[ix].item()
            
        loss_desc = ' | '.join([f'{k}: {v / (batch_ix + 1):.2f}' 
                                for k, v in total_loss.items()])    
        time_desc = f'fwd: {end - start:.2f}s| unpatch: {end_unpatch - start_unpatch:.2f}s | back: {end_backprop - start_backprop:.2f}s'
        pbar_desc = f'Batch: {batch_ix}/{len(dataloader)} | {time_desc} | {loss_desc}'
        pbar.set_description(pbar_desc)
        
        # model.cpu()
        # loss = loss.cpu()
        # # for ix in range(len(all_losses)):
        # #     all_losses[ix] = all_losses[ix].cpu()
        # u = u.cpu()
        # y = y.cpu()
        # y_c = y_c.cpu()
        # y_o = y_o.cpu()
        # z_c = z_c.cpu()
        # z_o = z_o.cpu()
        
    return model, total_loss, len(dataloader)     