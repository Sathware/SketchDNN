import torch
import torch.nn.functional as F
from torch import Tensor
from config import node_bce_weight, node_cross_weight, node_mse_weight, node_mae_weight, edge_suba_weight, edge_subb_weight, edge_constraint_weight, kld_weight, reg_weight
from dataset1 import SketchDataset

def reconstruction_loss(pred_nodes : Tensor, pred_edges : Tensor, target_nodes : Tensor, target_edges : Tensor, params_mask : Tensor, loss_dict : dict = None) -> Tensor:    
    #print("sub a", sub_a_cross_entropy, "sub b", sub_b_cross_entropy, "constr", constraint_cross_entropy)
    #return node_loss(pred_nodes, target_nodes) + edge_loss(pred_edges, target_edges)
    # return F.mse_loss(input = pred_nodes, target = target_nodes) + F.mse_loss(input = pred_edges, target = target_edges)
    return node_loss(pred_nodes, target_nodes, params_mask, loss_dict) + edge_loss(pred_edges, target_edges, loss_dict)

def posterior_collapse_regularization(pred_nodes : Tensor, pred_edges : Tensor, loss_dict : dict = None):
    reg = (1 / torch.var(input = pred_nodes, dim = 0)).mean() + (1 / torch.var(input = pred_edges, dim = 0)).mean()
    reg = reg * reg_weight
    reg = torch.clamp(input = reg, max = 100)
    
    if loss_dict is not None:
        loss_dict["postcollapse_reg"] = reg.item()

    return reg

def kl_loss(means : Tensor, logvars : Tensor, loss_dict : dict = None) -> Tensor:
    MAX_LOGVAR = 20
    logvars = torch.clamp(input = logvars, max = MAX_LOGVAR)

    kld = -0.5 * torch.mean(1 + logvars - means * means - torch.exp(logvars)) * kld_weight
    kld = torch.clamp(input = kld, max = 100)

    if loss_dict is not None:
        loss_dict["kld"] = kld.item()

    return kld

def node_loss(pred_nodes : Tensor, target_nodes : Tensor, params_mask : Tensor, loss_dict : dict = None) -> Tensor:
    '''Node Loss'''
    # weight = torch.tensor([1.0, 2.0, 2.0, 1.0, 0.1]).to(pred_nodes.device)  # Weight circles, arcs, and points higher since they are much rarer than line and none types
    primitive_type_labels = torch.argmax(target_nodes[:,:,1:6], dim = 2)    # batch_size x num_nodes (class index for each node)
    primitive_type_logits = pred_nodes[:,:,1:6]#.permute(0,2,1).contiguous() # batch_size x num_primitive_types x num_nodes
    
    node_cross = F.cross_entropy(
        input = primitive_type_logits.reshape(-1, 5), 
        target = primitive_type_labels.flatten(),
        # weight = weight, 
        reduction = 'mean') * node_cross_weight
    
    pred_isconstruct = pred_nodes[:,:,0].squeeze()
    target_isconstruct = target_nodes[:,:,0].squeeze()
    none_node_mask = (primitive_type_labels != 4)
    bce = F.binary_cross_entropy_with_logits(
        input = torch.masked_select(pred_isconstruct, none_node_mask), 
        target = torch.masked_select(target_isconstruct, none_node_mask),
        # pos_weight = weight,
        reduction = 'mean') * node_bce_weight

    pred_params = pred_nodes[:,:,6:]
    target_params = target_nodes[:,:,6:]
    mse = F.mse_loss(input = pred_params * params_mask,
                     target = target_params,
                     reduction = 'sum') / params_mask.sum() * node_mse_weight
    # mse = F.l1_loss(input = pred_params * params_mask,
    #                  target = target_params,
    #                  reduction = 'sum') / params_mask.sum() * node_mae_weight
    # ((pred_params - target_params) ** 2 * params_mask).sum() / params_mask.sum() * node_mse_weight
    # mse = F.mse_loss(input = pred_params * params_mask,
    #                  target = target_params, 
    #                  reduction='mean')

    # Total node loss
    node_loss = bce + node_cross + mse

    if loss_dict is not None:
        loss_dict["node loss"] = node_loss.item()
        loss_dict["node bce"] = bce.item()
        loss_dict["node cross"] = node_cross.item()
        loss_dict["node mse"] = mse.item()
        
    
    return node_loss

def edge_loss(pred_edges : Tensor, target_edges : Tensor, loss_dict : dict = None) -> Tensor:
    '''Edge Loss'''
    # Only apply subnode loss to constraints that are not none -------
    subnode_a_labels = torch.argmax(target_edges[:,:,:,0:4], dim = 3)
    subnode_a_logits = pred_edges[:,:,:,0:4]#.permute(0, 3, 1, 2).contiguous()
    sub_a_cross_entropy = F.cross_entropy(
        input = subnode_a_logits.reshape(-1, 4), 
        target = subnode_a_labels.flatten(), 
        reduction = 'mean') * edge_suba_weight

    subnode_b_labels = torch.argmax(target_edges[:,:,:,4:8], dim = 3)
    subnode_b_logits = pred_edges[:,:,:,4:8]#.permute(0, 3, 1, 2).contiguous()
    sub_b_cross_entropy = F.cross_entropy(
        input = subnode_b_logits.reshape(-1, 4), 
        target = subnode_b_labels.flatten(), 
        reduction = 'mean') * edge_subb_weight
    
    constraint_type_labels = torch.argmax(target_edges[:,:,:,8:], dim = 3)
    constraint_type_logits = pred_edges[:,:,:,8:]#.permute(0, 3, 1, 2).contiguous()
    # There are far more none constraint types, so weigh them less
    constraint_cross_entropy = F.cross_entropy(
        input = constraint_type_logits.reshape(-1, 9), 
        target = constraint_type_labels.flatten(),
        # weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.05]).to(pred_edges.device),
        reduction = 'mean') * edge_constraint_weight

    edge_loss = sub_a_cross_entropy + sub_b_cross_entropy + constraint_cross_entropy

    if loss_dict is not None:
        loss_dict["edge loss"] = edge_loss.item()
        loss_dict["edge sub_a cross"] = sub_a_cross_entropy.item()
        loss_dict["edge sub_b cross"] = sub_b_cross_entropy.item()
        loss_dict["edge cross"] = constraint_cross_entropy.item()
    
    return edge_loss


def diffusion_loss(pred_nodes, pred_edges, target_nodes, target_edges, params_mask, loss_dict = None):
    '''Edge Loss'''
    # Only apply subnode loss to constraints that are not none -------
    subnode_a_labels = torch.argmax(target_edges[:,:,:,0:4], dim = 3)
    subnode_a_logits = pred_edges[:,:,:,0:4]#.permute(0, 3, 1, 2).contiguous()
    sub_a_cross_entropy = F.cross_entropy(
        input = subnode_a_logits.reshape(-1, 4), 
        target = subnode_a_labels.flatten(), 
        reduction = 'mean') * edge_suba_weight

    subnode_b_labels = torch.argmax(target_edges[:,:,:,4:8], dim = 3)
    subnode_b_logits = pred_edges[:,:,:,4:8]#.permute(0, 3, 1, 2).contiguous()
    sub_b_cross_entropy = F.cross_entropy(
        input = subnode_b_logits.reshape(-1, 4), 
        target = subnode_b_labels.flatten(), 
        reduction = 'mean') * edge_subb_weight
    
    constraint_type_labels = torch.argmax(target_edges[:,:,:,8:], dim = 3)
    constraint_type_logits = pred_edges[:,:,:,8:]#.permute(0, 3, 1, 2).contiguous()
    # There are far more none constraint types, so weigh them less
    constraint_cross_entropy = F.cross_entropy(
        input = constraint_type_logits.reshape(-1, 9), 
        target = constraint_type_labels.flatten(),
        # weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.05]).to(pred_edges.device),
        reduction = 'mean') * edge_constraint_weight

    edge_loss = sub_a_cross_entropy + sub_b_cross_entropy + constraint_cross_entropy

    '''Node Loss'''
    # weight = torch.tensor([1.0, 2.0, 2.0, 1.0, 0.1]).to(pred_nodes.device)  # Weight circles, arcs, and points higher since they are much rarer than line and none types
    primitive_type_labels = torch.argmax(target_nodes[:,:,1:6], dim = 2)    # batch_size x num_nodes (class index for each node)
    primitive_type_logits = pred_nodes[:,:,1:6]#.permute(0,2,1).contiguous() # batch_size x num_primitive_types x num_nodes
    
    node_cross = F.cross_entropy(
        input = primitive_type_logits.reshape(-1, 5), 
        target = primitive_type_labels.flatten(),
        # weight = weight, 
        reduction = 'mean') * node_cross_weight
    
    pred_isconstruct = pred_nodes[:,:,0].squeeze()
    target_isconstruct = target_nodes[:,:,0].squeeze()
    none_node_mask = (primitive_type_labels != 4)
    bce = F.binary_cross_entropy_with_logits(
        input = torch.masked_select(pred_isconstruct, none_node_mask), 
        target = torch.masked_select(target_isconstruct, none_node_mask),
        # pos_weight = weight,
        reduction = 'mean') * node_bce_weight

    # pred_noise = pred_nodes[:,:,6:]
    # mse = ((pred_noise - true_noise) ** 2 * params_mask).sum() / params_mask.sum() * node_mse_weight 
    pred_params = pred_nodes[:,:,6:]
    target_params = target_nodes[:,:,6:]
    mse = ((pred_params - target_params) ** 2 * params_mask).sum() / params_mask.sum() * node_mse_weight

    node_loss = bce + node_cross + mse

    total_loss = node_loss + edge_loss

    if loss_dict is not None:
        loss_dict["edge loss"] = edge_loss.item()
        loss_dict["edge sub_a cross"] = sub_a_cross_entropy.item()
        loss_dict["edge sub_b cross"] = sub_b_cross_entropy.item()
        loss_dict["edge cross"] = constraint_cross_entropy.item()
        loss_dict["node loss"] = node_loss.item()
        loss_dict["node bce"] = bce.item()
        loss_dict["node cross"] = node_cross.item()
        loss_dict["node mse"] = mse.item()
        loss_dict["total loss"] = total_loss.item()
    
    return total_loss