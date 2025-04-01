import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller
import math
import numpy as np
    
class MSKD(Distiller):

    def __init__(self, student, teacher, cfg):
        super(MSKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.MSKD.LOSS.CE_WEIGHT
        self.CAT_loss_weight = cfg.MSKD.LOSS.CAT_loss_weight
        self.CAM_RESOLUTION = cfg.MSKD.LOSS.CAM_RESOLUTION
        self.IF_NORMALIZE = cfg.MSKD.IF_NORMALIZE
        self.T = cfg.MSKD.LOSS.T
        self.sim_weight = cfg.MSKD.LOSS.sim_weight
        
    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)       
        tea = feature_teacher["feats"][-1]
        stu = feature_student["feats"][-1]

        
        b = tea.shape[0]
        sim_weight = self.sim_weight
        T = self.T
        IF_NORMALIZE = self.IF_NORMALIZE
        CAT_loss_weight = self.CAT_loss_weight
        if IF_NORMALIZE:
            CAT_loss_weight = CAT_loss_weight / (self.CAM_RESOLUTION * self.CAM_RESOLUTION)
        stu = F.adaptive_avg_pool2d(stu, (8, 8))
        tea = F.adaptive_avg_pool2d(tea, (8, 8))
        if kwargs["epoch"] <= 10:
            num_levels = [2]
        elif kwargs["epoch"] <= 20:
            num_levels = [2, 4]
        else:
            num_levels = [2, 4, 8]
        weight = Get_weight(stu, tea, num_levels, IF_NORMALIZE, T)
        loss = Get_loss(stu, tea, num_levels, IF_NORMALIZE)
        sim_loss = Get_sim(stu, tea, num_levels, IF_NORMALIZE)
        loss_nor = torch.mul(weight, loss).sum() / (1.0 * b)
        loss_sim = torch.mul(weight, sim_loss).sum() / (1.0 * b)
        loss_feat = CAT_loss_weight * (loss_nor + sim_weight * loss_sim)
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        losses_dict = {
            "loss_CE": loss_ce,
            "loss_CAT": loss_feat,
        }
        

        return logits_student, losses_dict


def Get_sim(stu, tea, num_levels, IF_NORMALIZE):
    sim = []
    b, c, h, w = stu.shape
    for i in range(len(num_levels)):
        level = num_levels[i]
        tensor_stu = F.normalize(F.adaptive_avg_pool2d(stu, (level, level)).view(b, c, -1), dim=-1)
        tensor_tea = F.normalize(F.adaptive_avg_pool2d(tea, (level, level)).view(b, c, -1), dim=-1)

        sim_stu = torch.bmm(tensor_stu, tensor_stu.transpose(1,2))
        sim_stu = torch.triu(sim_stu, diagonal=1)

        sim_tea = torch.bmm(tensor_tea, tensor_tea.transpose(1,2))
        sim_tea = torch.triu(sim_tea, diagonal=1)

        sim_loss = F.mse_loss(sim_stu, sim_tea, reduction='none').sum(dim=-1).mean(dim=-1) / (c*(c-1)/2)
        sim.append(sim_loss)
    loss = torch.stack(sim, dim=0)
    return loss



def Get_weight(stu, tea, num_levels, IF_NORMALIZE, T):
    diff = []
    for i in range(len(num_levels)):
        level = num_levels[i]
        tensor_stu = F.adaptive_avg_pool2d(stu, (level, level))
        tensor_tea = F.adaptive_avg_pool2d(tea, (level, level))
        loss = F.mse_loss(_Normalize(tensor_stu, IF_NORMALIZE), _Normalize(tensor_tea, IF_NORMALIZE), reduction='none').mean(dim=(1,2,3))
        diff.append(loss)
    diff_concat = torch.stack(diff, dim=0)
    diff_concat = -diff_concat / T
    return torch.softmax(diff_concat, dim=0)


def _Normalize(feat,IF_NORMALIZE):
    b, c, h, w = feat.shape
    if IF_NORMALIZE:
        feat = F.normalize(feat, dim=(2,3)) * h
    return feat


def Get_loss(CAM_Student, CAM_Teacher, num_levels, IF_NORMALIZE):
    loss_mu = []
    for i in range(len(num_levels)):
        level = num_levels[i]
        stu = F.adaptive_avg_pool2d(CAM_Student, (level, level))
        tea = F.adaptive_avg_pool2d(CAM_Teacher, (level, level))
        loss = F.mse_loss(_Normalize(stu, IF_NORMALIZE), _Normalize(tea, IF_NORMALIZE), reduction='none').mean(dim=(1,2,3))
        loss_mu.append(loss)
    loss = torch.stack(loss_mu, dim=0)
    return loss