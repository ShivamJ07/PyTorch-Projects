import torch

def get_mean_std(loader):
    channels_sum, channels_squares_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        # NCHW - Find mean and std across all channels C
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squares_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum/num_batches
    # VAR[x] = E[X**2] - E[X]**2
    # E[X] = (sum of X / n) = X bar 
    # std[x] = sqrt(VAR[X])
    std = (channels_squares_sum/num_batches - mean**2)**0.5

    return mean, std
