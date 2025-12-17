import math
import torch

@torch.no_grad()
def batched_dot(a, b):
    return (a * b).sum(dim = -1, keepdim = True)

'''Rescales and shifts a batch of sketches to be inscribed within the unit square whose corners are (-1,1) and (1,-1)'''
@torch.no_grad()
def BoundingBoxShiftScale(nodes):
    nodes = ToNaive(nodes) # Change arc parameterization to be center, radius, and terminal angles
    # Indices of Parameters to be shifted and scaled
    val_indices = [7,8,9,10,11,12,13,14,15,16,19,20]
    x_indices = [7,9,11,14,19]
    y_indices = [8,10,12,15,20]

    # The additions and subtractions in the concatenation are circle/arc center +/- radius
    max_x = torch.amax(torch.cat([nodes[...,x_indices], nodes[...,[11]] + nodes[...,[13]], nodes[...,[14]] + nodes[...,[16]]], dim = -1), dim = [1, 2], keepdim = True)
    min_x = torch.amin(torch.cat([nodes[...,x_indices], nodes[...,[11]] - nodes[...,[13]], nodes[...,[14]] - nodes[...,[16]]], dim = -1), dim = [1, 2], keepdim = True)
    max_y = torch.amax(torch.cat([nodes[...,y_indices], nodes[...,[12]] + nodes[...,[13]], nodes[...,[15]] + nodes[...,[16]]], dim = -1), dim = [1, 2], keepdim = True)
    min_y = torch.amin(torch.cat([nodes[...,y_indices], nodes[...,[12]] - nodes[...,[13]], nodes[...,[15]] - nodes[...,[16]]], dim = -1), dim = [1, 2], keepdim = True)

    width = max_x - min_x
    height = max_y - min_y

    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2

    # Shift so center of bounding box is the origin
    mask = torch.cat([nodes[...,[2]], nodes[...,2:6]], dim = -1)
    nodes[...,x_indices] = (nodes[...,x_indices] - center_x) * mask
    nodes[...,y_indices] = (nodes[...,y_indices] - center_y) * mask

    # Scale so axis aligned bounding box is inscribed in the square spanning -1 and 1
    nodes[...,val_indices] = 2 * nodes[...,val_indices] / torch.where(width > height, width, height)

    return ToIscosceles(nodes) # Convert back to original arc parameterizations

def GetUniqueIndices(tensor, num_levels):
    # Get min and max values for each feature across batch and nodes
    min_vals, _ = tensor.min(dim=-1, keepdim=True)  # Shape: (batch_size, num_nodes, 1)
    max_vals, _ = tensor.max(dim=-1, keepdim=True)  # Shape: (batch_size, num_nodes, 1)
    
    # Normalize each feature to the range [0, 1]
    tensor_normalized = (tensor - min_vals) / (max_vals - min_vals + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Scale to the quantization levels
    tensor_scaled = tensor_normalized * (num_levels - 1)
    
    # Round to the nearest quantization level
    tensor_quantized = torch.round(tensor_scaled).to(dtype=torch.int32)

    _, inverse_indices, counts = torch.unique(input = tensor_quantized, return_inverse = True, return_counts = True, sorted = False, dim = 0) # Gives which index each tensor maps to, duplicate tensors map to the same index in unsorted

    _, permutation_sort = torch.sort(inverse_indices, stable=True) # permutation_sort is the permutation of indices to make the input sorted
    cum_sum = counts.cumsum(0) # End position of each group of duplicates
    cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1])) # Start position of each group of duplicates
    first_indicies = permutation_sort[cum_sum] # First occurence of tensors

    return first_indicies

'''Changes arc parameters from onshape's representation to instead the endpoints and signed radius'''
@torch.no_grad()
def ToIscosceles(nodes):
    m = torch.tensor([[0.0, -1.0], [1.0, 0.0]], device = nodes.device).to(torch.double)

    # To Iscosceles representation
    nodes = nodes.to(torch.double)
    c = nodes[...,14:16]
    r = nodes[...,16,None]
    alpha = nodes[...,17,None] * 2 * math.pi
    beta = nodes[...,18,None] * 2 * math.pi

    a = c + r * torch.cat([alpha.cos(), alpha.sin()], dim = -1)
    b = c + r * torch.cat([beta.cos(), beta.sin()], dim = -1)
    r_s = torch.where(batched_dot((a - b) @ m, a - c) > 0, -r, r)

    nodes[...,14:16] = a
    nodes[...,16:18] = b
    nodes[...,18] = r_s.squeeze(-1)

    return nodes.to(torch.float32)

'''Changes arc parameters from iscoseles representation to the center, radius, and terminating angles'''
@torch.no_grad()
def ToNaive(nodes):
    m = torch.tensor([[0.0, -1.0], [1.0, 0.0]], device = nodes.device).to(torch.double)

    nodes = nodes.to(torch.double)
    a = nodes[...,14:16]
    b = nodes[...,16:18]
    r_s = nodes[...,18,None]

    # To Naive representation
    d = a - b

    w = batched_dot(d, d).sqrt()
    h = torch.sqrt(r_s ** 2 - w ** 2 / 4)

    c = b + d / 2 + h / w * d @ m * r_s.sign()

    al = (a - c) / r_s.abs()
    al = (torch.atan2(al[...,1], al[...,0]).unsqueeze(-1) % (2 * math.pi)) / (2 * math.pi)
    be = (b - c) / r_s.abs()
    be = (torch.atan2(be[...,1], be[...,0]).unsqueeze(-1) % (2 * math.pi)) / (2 * math.pi)

    nodes[...,14:19] = torch.cat([c, torch.abs(r_s), al, be], dim = -1)
    nodes[nodes.isnan()] = 0
    return nodes.to(torch.float32)