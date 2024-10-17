import torch

# Mate function implemented with PyTorch
def Mate(a, b):
    # Create a random mask with 0.5 probability for each element
    raded = torch.rand_like(a) > 0.5
    # Complementary mask
    raded2 = ~raded
    # Create the result array by selecting elements from a or b based on the mask
    res = torch.empty_like(a)
    res[raded] = a[raded]
    res[raded2] = b[raded2]
    return res

# Mute function implemented with PyTorch
def Mute(a, e=0.1):
    # Create a random mask with probability `e` for mutation
    raded = torch.rand_like(a) < e
    # Generate small random noise between -0.005 and 0.005
    r = torch.rand_like(a) * 0.01 - 0.005
    # Apply mutation where the mask is True
    a[raded] += r[raded]
    return a
