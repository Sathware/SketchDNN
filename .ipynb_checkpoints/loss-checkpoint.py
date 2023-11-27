import torch
import torch.nn.functional as F
from torch import Tensor
from config import NUM_PRIMITIVE_TYPES, MAX_NUM_PRIMITIVES, NODE_FEATURE_DIMENSION, NUM_CONSTRAINT_TYPES


def reconstruction_loss(pred_nodes : Tensor, pred_edges : Tensor, target_nodes : Tensor, target_edges : Tensor):    
    #print("sub a", sub_a_cross_entropy, "sub b", sub_b_cross_entropy, "constr", constraint_cross_entropy)
    return node_loss(pred_nodes, target_nodes) #+ 0.1 * edge_loss(pred_edges, target_edges)

def kl_loss(means : Tensor, logvars : Tensor):
    # MAX_LOGVAR = 20
    # logvars = torch.clamp(input = logvars, max = MAX_LOGVAR)

    kld = -0.5 * torch.mean(1 + logvars - means * means - torch.exp(logvars))
    # kld = torch.clamp(input = kld, max = 1000)
    return kld

def node_loss(pred_nodes : Tensor, target_nodes : Tensor) -> Tensor:
    '''Node Loss'''
    weight = torch.tensor([1.0, 4.0, 4.0, 3.0, 0.1]).to(pred_nodes.device)  # Weight circles, arcs, and points higher since they are much rarer than line and none types
    primitive_type_labels = torch.argmax(target_nodes[:,:,1:6], dim = 2)    # batch_size x num_nodes (class index for each node)
    primitive_type_logits = pred_nodes[:,:,1:6].permute(0,2,1).contiguous() # batch_size x num_primitive_types x num_nodes
    
    node_cross = F.cross_entropy(
        input = primitive_type_logits, 
        target = primitive_type_labels, 
        weight = weight, 
        reduction = 'mean')

    bce = F.binary_cross_entropy(
        input = pred_nodes[:,:,0], 
        target = target_nodes[:,:,0],
        reduction = 'mean')
    
    target_params = target_nodes[:,:,6:]
    pred_params = pred_nodes[:,:,6:]
    mse = F.mse_loss(input = pred_params[torch.abs(target_params) > 1e-6], 
                     target = target_params[torch.abs(target_params) > 1e-6], 
                     reduction='mean')

    # Total node loss
    node_loss = bce + node_cross #+ 16 * mse
    
    return node_loss

def edge_loss(pred_edges : Tensor, target_edges : Tensor) -> Tensor:
    '''Edge Loss'''
    constraint_type_labels = torch.argmax(target_edges[:,:,:,8:], dim = 3)
    constraint_type_logits = pred_edges[:,:,:,8:].permute(0, 3, 1, 2).contiguous()
    # There are far more none constraint types, so weigh them less
    constraint_cross_entropy = F.cross_entropy(
        input = constraint_type_logits, 
        target = constraint_type_labels,
        weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.05]).to(pred_edges.device),
        reduction = 'mean')
    
    # Only apply subnode loss to constraints that are not none -------
    subnode_a_labels = torch.argmax(target_edges[:,:,:,0:4], dim = 3)
    subnode_a_logits = pred_edges[:,:,:,0:4].permute(0, 3, 1, 2).contiguous()
    sub_a_cross_entropy = F.cross_entropy(
        input = subnode_a_logits, 
        target = subnode_a_labels, 
        reduction = 'mean')

    subnode_b_labels = torch.argmax(target_edges[:,:,:,4:8], dim = 3)
    subnode_b_logits = pred_edges[:,:,:,4:8].permute(0, 3, 1, 2).contiguous()
    sub_b_cross_entropy = F.cross_entropy(
        input = subnode_b_logits, 
        target = subnode_b_labels, 
        reduction = 'mean')

    edge_loss = sub_a_cross_entropy + sub_b_cross_entropy + constraint_cross_entropy
    return edge_loss

# def params_mask(primitive_type_labels : Tensor, pred_params : Tensor):
#     for i in range(primitive_type_labels.size()[0]): # primitive_type_labels = batch_size x MAX_NUM_PRIMITIVES
#         for j in range(MAX_NUM_PRIMITIVES):
#             match primitive_type_labels[i][j]:
#                 case 0:
#                     # Line
#                     pred_params[i][j][4:] = 0
#                 case 1:
#                     # Circle
#                     pred_params[i][j][0:4] = 0
#                     pred_params[i][j][7:] = 0
#                 case 2:
#                     # Arc
#                     pred_params[i][j][0:7] = 0
#                     pred_params[i][j][12:] = 0
#                 case 3:
#                     # Point
#                     pred_params[i][j][:12] = 0
    
#     return pred_params