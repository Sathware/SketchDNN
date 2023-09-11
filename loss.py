import torch
import torch.nn.functional as F
from torch import Tensor
from config import NUM_PRIMITIVE_TYPES, NUM_CONSTRAINT_TYPES

def reconstruction_loss(pred_nodes : Tensor, pred_edges : Tensor, target_nodes : Tensor, target_edges : Tensor, node_params_mask : Tensor):
    '''Node Loss'''
    bce = F.binary_cross_entropy(input = pred_nodes[:,:,0], target = target_nodes[:,:,0], reduction = 'mean')

    primitive_type_labels = torch.argmax(target_nodes[:,:,1:6], dim = 2).view(-1) # (batch_size * num_nodes) x 1
    primitive_type_logits = pred_nodes[:,:,1:6].view(-1, NUM_PRIMITIVE_TYPES)     # (batch_size * num_nodes) x num_primitive_types
    node_cross = F.cross_entropy(input = primitive_type_logits, target = primitive_type_labels, reduction = 'mean')

    # node_params_mask ensures that only relevant primtive parameters are used for loss 
    mse = F.mse_loss(input = pred_nodes[:,:,6:] * node_params_mask, target = target_nodes[:,:,6:], reduction='mean')

    node_loss = bce + node_cross + mse

    '''Edge Loss'''
    subnode_a_labels = torch.argmax(target_edges[:,:,:,0:4], dim = 3).view(-1)
    subnode_a_logits = pred_edges[:,:,:,0:4].view(-1, 4)
    sub_a_cross_entropy = F.cross_entropy(input = subnode_a_logits, target = subnode_a_labels, reduction = 'mean')

    subnode_b_labels = torch.argmax(target_edges[:,:,:,4:8], dim = 3).view(-1)
    subnode_b_logits = pred_edges[:,:,:,4:8].view(-1, 4)
    sub_b_cross_entropy = F.cross_entropy(input = subnode_b_logits, target = subnode_b_labels, reduction = 'mean')

    constraint_type_labels = torch.argmax(target_edges[:,:,:,8:], dim = 3).view(-1)
    constraint_type_logits = pred_edges[:,:,:,8:].view(-1, NUM_CONSTRAINT_TYPES)
    constraint_cross_entropy = F.cross_entropy(input = constraint_type_logits, target = constraint_type_labels, reduction = 'mean')

    edge_loss = sub_a_cross_entropy + sub_b_cross_entropy + constraint_cross_entropy

    return node_loss + edge_loss

def kl_loss(means : Tensor, logvars : Tensor):
    MAX_LOGVAR = 20
    logvars = torch.clamp(input = logvars, max = MAX_LOGVAR)

    kld = -0.5 * torch.mean(1 + logvars - means * means - torch.exp(logvars))
    kld = torch.clamp(input = kld, max = 1000)
    return kld


