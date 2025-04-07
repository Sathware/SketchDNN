import math
import torch

@torch.no_grad()
def batched_dot(a, b):
    return (a * b).sum(dim = -1, keepdim = True)

# '''Changes arc parameters from naive representation to the center, start, and end points'''
# @torch.no_grad()
# def ToIscosceles(nodes):
#     # To Iscosceles representation
#     c = nodes[...,13:15] 
#     r = nodes[...,15,None] 
#     alpha = nodes[...,16,None] * 2 * math.pi 
#     beta = nodes[...,17,None] * 2 * math.pi 

#     a = c + r * torch.cat([alpha.cos(), alpha.sin()], dim = -1)
#     b = c + r * torch.cat([beta.cos(), beta.sin()], dim = -1)

#     nodes[...,13:15] = a
#     nodes[...,15:17] = b
#     nodes[...,17] = 
#     # nodes = torch.cat([nodes[...,:17], b, nodes[...,18:]], dim = -1)
#     return nodes

# '''Changes arc parameters from iscoseles representation to the center, radius, and terminating angles'''
# @torch.no_grad()
# def ToNaive(nodes):
#     c = nodes[...,13:15]
#     a = nodes[...,15:17]
#     b = nodes[...,17:19]
#     r = torch.sum((a - c) ** 2, dim = -1, keepdim = True).sqrt()

#     al = (a - c)
#     al = (torch.atan2(al[...,1], al[...,0]).unsqueeze(-1) % (2 * math.pi)) / (2 * math.pi)
#     be = (b - c)
#     be = (torch.atan2(be[...,1], be[...,0]).unsqueeze(-1) % (2 * math.pi)) / (2 * math.pi)

#     nodes[...,13:18] = torch.cat([c, r, al, be], dim = -1)
#     nodes[nodes.isnan()] = 0

#     indices = torch.arange(nodes.size(-1))  # Generate all indices
#     excluded_indices = indices[indices != 18]  # Filter out index 18
#     nodes = nodes[..., excluded_indices]
#     return nodes

'''Moves the Center of mass to the origin expects naive representation of arcs'''
@torch.no_grad()
def ToCenter(nodes):
    # Move center of mass for each sketch to the origin
    x_indices = [6,8,10,13,18]
    y_indices = [7,9,11,14,19]

    an = nodes.sum(dim = 1) # aggregate nodes

    denom = (24 - an[:,5] + an[:,1])
    avg_x = an[:,x_indices].sum(dim = -1) / denom
    avg_y = an[:,y_indices].sum(dim = -1) / denom

    mask = torch.cat([nodes[...,[1]], nodes[...,1:5]], dim = -1)
    nodes[...,x_indices] = (nodes[...,x_indices] - avg_x[:,None,None]) * mask
    nodes[...,y_indices] = (nodes[...,y_indices] - avg_y[:,None,None]) * mask

    return nodes

'''Moves the Center of mass to the origin'''
@torch.no_grad()
def ParamScale(nodes):
    # Rescale each sketch so that the maximum or minimum parameter is 1 or -1 respectively
    val_indices = [6,7,8,9,10,11,12,13,14,15,18,19]
    nodes[...,val_indices] = nodes[...,val_indices] / torch.amax(torch.abs(nodes[...,val_indices]), dim = [1, 2], keepdim = True)

    return nodes

'''Moves the Center of mass to the origin expects naive representation of arcs'''
@torch.no_grad()
def BoundingBoxShiftScale(nodes):
    # Indices of Parameters to be shifted and scaled
    val_indices = [6,7,8,9,10,11,12,13,14,15,18,19]
    x_indices = [6,8,10,13,18]
    y_indices = [7,9,11,14,19]

    # The additions and subtractions in the concatenation are circle/arc center +/- radius
    max_x = torch.amax(torch.cat([nodes[...,x_indices], nodes[...,[10]] + nodes[...,[12]], nodes[...,[13]] + nodes[...,[15]]], dim = -1), dim = [1, 2], keepdim = True)
    min_x = torch.amin(torch.cat([nodes[...,x_indices], nodes[...,[10]] - nodes[...,[12]], nodes[...,[13]] - nodes[...,[15]]], dim = -1), dim = [1, 2], keepdim = True)
    max_y = torch.amax(torch.cat([nodes[...,y_indices], nodes[...,[11]] + nodes[...,[12]], nodes[...,[14]] + nodes[...,[15]]], dim = -1), dim = [1, 2], keepdim = True)
    min_y = torch.amin(torch.cat([nodes[...,y_indices], nodes[...,[11]] - nodes[...,[12]], nodes[...,[14]] - nodes[...,[15]]], dim = -1), dim = [1, 2], keepdim = True)

    width = max_x - min_x
    height = max_y - min_y

    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2

    # Shift so center of bounding box is the origin
    mask = torch.cat([nodes[...,[1]], nodes[...,1:5]], dim = -1)
    nodes[...,x_indices] = (nodes[...,x_indices] - center_x) * mask
    nodes[...,y_indices] = (nodes[...,y_indices] - center_y) * mask

    # Scale so axis aligned bounding box is inscribed in the square spanning -1 and 1
    nodes[...,val_indices] = 2 * nodes[...,val_indices] / torch.where(width > height, width, height)

    return nodes

def GetUniqueIndices(tensor, num_levels):
    # Step 1: Get min and max values for each feature across batch and nodes
    min_vals, _ = tensor.min(dim=-1, keepdim=True)  # Shape: (batch_size, num_nodes, 1)
    max_vals, _ = tensor.max(dim=-1, keepdim=True)  # Shape: (batch_size, num_nodes, 1)
    
    # Step 2: Normalize each feature to the range [0, 1]
    tensor_normalized = (tensor - min_vals) / (max_vals - min_vals + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Step 3: Scale to the quantization levels
    tensor_scaled = tensor_normalized * (num_levels - 1)
    
    # Step 4: Round to the nearest quantization level
    tensor_quantized = torch.round(tensor_scaled).to(dtype=torch.int32)

    _, inverse_indices, counts = torch.unique(input = tensor_quantized, return_inverse = True, return_counts = True, sorted = False, dim = 0) # Gives which index each tensor maps to, duplicate tensors map to the same index in unsorted

    _, permutation_sort = torch.sort(inverse_indices, stable=True) # permutation_sort is the permutation of indices to make the input sorted
    cum_sum = counts.cumsum(0) # End position of each group of duplicates
    cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1])) # Start position of each group of duplicates
    first_indicies = permutation_sort[cum_sum] # First occurence of tensors

    return first_indicies

# import math
# import torch

# @torch.no_grad()
# def batched_dot(a, b):
#     return (a * b).sum(dim = -1, keepdim = True)

'''Changes arc parameters from naive representation to the endpoints and signed radius'''
@torch.no_grad()
def ToIscosceles(nodes):
    m = torch.tensor([[0.0, -1.0], [1.0, 0.0]], device = nodes.device).to(torch.double)

    # To Iscosceles representation
    nodes = nodes.to(torch.double)
    c = nodes[...,13:15] # torch.tensor([1.76, 2.21])
    r = nodes[...,15,None] # torch.tensor([1.0])
    alpha = nodes[...,16,None] * 2 * math.pi # torch.tensor([0.211])
    beta = nodes[...,17,None] * 2 * math.pi # torch.tensor([0.987])

    a = c + r * torch.cat([alpha.cos(), alpha.sin()], dim = -1)
    b = c + r * torch.cat([beta.cos(), beta.sin()], dim = -1)
    r_s = torch.where(batched_dot((a - b) @ m, a - c) > 0, -r, r)

    nodes[...,13:15] = a
    nodes[...,15:17] = b
    nodes[...,17] = r_s.squeeze(-1)

    return nodes.to(torch.float32)

'''Changes arc parameters from iscoseles representation to the center, radius, and terminating angles'''
@torch.no_grad()
def ToNaive(nodes):
    m = torch.tensor([[0.0, -1.0], [1.0, 0.0]], device = nodes.device).to(torch.double)

    nodes = nodes.to(torch.double)
    a = nodes[...,13:15]
    b = nodes[...,15:17]
    r_s = nodes[...,17,None]

    # To Naive representation
    d = a - b

    w = batched_dot(d, d).sqrt()
    h = torch.sqrt(r_s ** 2 - w ** 2 / 4)

    c = b + d / 2 + h / w * d @ m * r_s.sign()

    al = (a - c) / r_s.abs()
    al = (torch.atan2(al[...,1], al[...,0]).unsqueeze(-1) % (2 * math.pi)) / (2 * math.pi)
    be = (b - c) / r_s.abs()
    be = (torch.atan2(be[...,1], be[...,0]).unsqueeze(-1) % (2 * math.pi)) / (2 * math.pi)

    nodes[...,13:18] = torch.cat([c, torch.abs(r_s), al, be], dim = -1)
    nodes[nodes.isnan()] = 0
    return nodes.to(torch.float32)

# '''Moves the Center of mass to the origin expects naive representation of arcs'''
# @torch.no_grad()
# def ToCenter(nodes):
#     # Move center of mass for each sketch to the origin
#     x_indices = [6,8,10,13,18]
#     y_indices = [7,9,11,14,19]

#     an = nodes.sum(dim = 1) # aggregate nodes

#     denom = (24 - an[:,5] + an[:,1])
#     avg_x = an[:,x_indices].sum(dim = -1) / denom
#     avg_y = an[:,y_indices].sum(dim = -1) / denom

#     mask = torch.cat([nodes[...,[1]], nodes[...,1:5]], dim = -1)
#     nodes[...,x_indices] = (nodes[...,x_indices] - avg_x[:,None,None]) * mask
#     nodes[...,y_indices] = (nodes[...,y_indices] - avg_y[:,None,None]) * mask

#     return nodes

# '''Moves the Center of mass to the origin'''
# @torch.no_grad()
# def ParamScale(nodes):
#     # Rescale each sketch so that the maximum or minimum parameter is 1 or -1 respectively
#     val_indices = [6,7,8,9,10,11,12,13,14,15,18,19]
#     nodes[...,val_indices] = nodes[...,val_indices] / torch.amax(torch.abs(nodes[...,val_indices]), dim = [1, 2], keepdim = True)

#     return nodes

# '''Moves the Center of mass to the origin expects naive representation of arcs'''
# @torch.no_grad()
# def BoundingBoxScale(nodes):
#     # Rescale each sketch so that either the width or height has 4 units of length
#     val_indices = [6,7,8,9,10,11,12,13,14,15,18,19]
#     x_indices = [6,8,10,13,18]
#     y_indices = [7,9,11,14,19]

#     # The additions and subtractions in the concatenation are circle/arc center +/- radius
#     max_x = torch.amax(torch.cat([nodes[...,x_indices], nodes[...,[10]] + nodes[...,[12]], nodes[...,[13]] + nodes[...,[15]]], dim = -1), dim = [1, 2], keepdim = True)
#     min_x = torch.amin(torch.cat([nodes[...,x_indices], nodes[...,[10]] - nodes[...,[12]], nodes[...,[13]] - nodes[...,[15]]], dim = -1), dim = [1, 2], keepdim = True)
#     max_y = torch.amax(torch.cat([nodes[...,y_indices], nodes[...,[11]] + nodes[...,[12]], nodes[...,[14]] + nodes[...,[15]]], dim = -1), dim = [1, 2], keepdim = True)
#     min_y = torch.amin(torch.cat([nodes[...,y_indices], nodes[...,[11]] - nodes[...,[12]], nodes[...,[14]] - nodes[...,[15]]], dim = -1), dim = [1, 2], keepdim = True)

#     width = max_x - min_x
#     height = max_y - min_y

#     nodes[...,val_indices] = nodes[...,val_indices] / torch.where(width > height, width, height)

#     return nodes