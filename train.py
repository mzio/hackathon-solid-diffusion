import torch
from tqdm.auto import tqdm


def train(model, **kwargs):
    model.train()
    model.set_inference_only(mode=False)
    return run_epoch(model, True, **kwargs)
    
    
    
def evaluate(model, **kwargs):
    model.eval()
    model.set_inference_only(mode=True)
    
    with torch.no_grad():
        return run_epoch(model, False, **kwargs)


def run_epoch(model, train, dataloader, optimizer, criterion, criterion_weights, device):
        
    total_loss = {'y_co': 0, 'z_co': 0, 'y_ct': 0, 'y_ot': 0}
    pbar = tqdm(enumerate(dataloader), leave=False)
    
    # data is zip(inputs, means)
    for batch_ix, data in pbar:
        u, y = data
        
        # Get correct reverse process ordering
        u = torch.flip(u, [-1])  # B x C x H x W x T
        y = torch.flip(y, [-1])  # B x C x H x W x T

        model.to(device)
        u = u.to(device)
        y = y.to(device)

        # c is closed loop, o is open loop
        y_c, y_o, z_c, z_o = model(u)
        
        if train:  # Compute 4 MSE Losses
            # Alignment between closed-loop and open-loop computations
            loss_y_co = criterion(y_c[:, 1:, :], y_o[:, 1:, :])
            loss_z_co = criterion(z_c[:, 1:, :], z_o[:, 1:, :])
            
            # loss_z_co = criterion(z_c[:, 1:, :], z_o[:, 1:, :])
            
        else:
            loss_y_co = 0; loss_z_co = 0
            
        # Alignment between predicted and ground-truth outputs
        y_c = model.input_embedding.unpatch(y_c)
        y_o = model.input_embedding.unpatch(y_o)            
        loss_y_ct = criterion(y_c[..., -10:], y[..., -10:])  # d_state - 1
        loss_y_ot = criterion(y_o[..., 1:], y[..., 1:])
        

        all_losses = [loss_y_co, loss_z_co, loss_y_ct, loss_y_ot]
            
        loss = sum([criterion_weights[ix] * all_losses[ix] 
                    for ix in range(len(all_losses))])
        
        if train:
            loss.backward()
            optimizer.step()
            model.zero_grad()
        
        for ix, k in enumerate(total_loss.keys()):
            total_loss[k] += all_losses[ix].item()
            
        loss_desc = ' | '.join([f'{k}: {v / (batch_ix + 1):.3f}' 
                                for k, v in total_loss.items()])    
        pbar_desc = f'Batch: ({batch_ix}/{len(dataloader)}) | {loss_desc}'
        pbar.set_description(pbar_desc)
        
        model.cpu()
        loss = loss.cpu()
        # for ix in range(len(all_losses)):
        #     all_losses[ix] = all_losses[ix].cpu()
        u = u.cpu()
        y = y.cpu()
        y_c = y_c.cpu()
        y_o = y_o.cpu()
        z_c = z_c.cpu()
        z_o = z_o.cpu()
        
    return model, total_loss, len(dataloader)     