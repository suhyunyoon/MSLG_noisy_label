import torch
import torch.nn as nn

def symmetric_crossentropy(NUM_CLASSES=10, reduction='mean'):
    """
    2019 - ICCV - Symmetric Cross Entropy for Robust Learning with Noisy Labels" 
    github repo: https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels
    """
    def criterion(y_pred, y_true):
        alpha=0.1
        beta=1.0
        y_true_1 = nn.functional.one_hot(y_true, NUM_CLASSES)
        y_pred_1 = nn.functional.softmax(y_pred, dim=1)

        y_true_2 = nn.functional.one_hot(y_true, NUM_CLASSES)
        y_pred_2 = nn.functional.softmax(y_pred, dim=1)

        y_pred_1 = torch.clamp(y_pred_1, 1e-7, 1.0)
        y_true_2 = torch.clamp(y_true_2, 1e-4, 1.0)
        loss1 = -torch.sum(y_true_1 * nn.functional.log_softmax(y_pred_1, dim=1), axis=1)
        loss2 = -torch.sum(y_pred_2 * nn.functional.log_softmax(y_true_2.type(torch.float), dim=1), axis=1)
        per_example_loss = alpha*loss1 + beta*loss2

        if reduction == 'mean':
	        return torch.mean(per_example_loss)
        elif reduction == 'none':
            return per_example_loss

    return criterion

def generalized_crossentropy(NUM_CLASSES=10, reduction='mean'):
    """
    2018 - NIPS - Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels.
    """
    def criterion(y_pred, y_true):
        q = 0.7
        ytrue_tmp = nn.functional.one_hot(y_true, NUM_CLASSES)
        ypred_tmp = nn.functional.softmax(y_pred, dim=1)
        t_loss = (1 - torch.pow(torch.sum(ytrue_tmp*ypred_tmp, axis=1), q)) / q
        
        if reduction == 'mean':
            return torch.mean(t_loss)
        elif reduction == 'none':
            return t_loss

    return criterion

def bootstrap_soft(NUM_CLASSES=10, reduction='mean'):
    """
    2015 - ICLR - Training deep neural networks on noisy labels with bootstrapping.
    github repo: https://github.com/dwright04/Noisy-Labels-with-Bootstrapping
    """
    def criterion(y_pred,y_true):
        beta = 0.95
        y_pred_softmax = nn.functional.softmax(y_pred, dim=1)
        y_true_onehot = nn.functional.one_hot(y_true ,NUM_CLASSES)
        y_true_modified = beta * y_true_onehot + (1. - beta) * y_pred_softmax
        loss = -torch.sum(y_true_modified * nn.functional.log_softmax(y_pred, dim=1), axis=1)
        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'none':
            return loss

    return criterion

def forwardloss(P, loss_object, reduction='mean'):
    """
    2017 - CVPR - Making Deep Neural Networks Robust to Label Noise: a Loss Correction Approach
    github repo: https://github.com/giorgiop/loss-correction
    """
    def criterion(y_pred, y_true):
        y_p = nn.functional.softmax(torch.mm(y_pred, P), dim=1)
        loss = loss_object(y_p, y_true)
        
        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'none':
            return loss

    return criterion

def joint_optimization(p, reduction='mean'):
    """
    2018 - CVPR - Joint optimization framework for learning with noisy labels.
    github repo: https://github.com/DaikiTanaka-UT/JointOptimization
    """
    sparse_categorical_crossentropy = nn.CrossEntropyLoss()
    def criterion(y_pred, y_true):
        ypred_tmp = nn.functional.softmax(y_pred, dim=1)
        y_pred_avg = torch.mean(ypred_tmp, axis=0)
        l_p = -torch.sum(torch.log(y_pred_avg) * p)
        l_e = -torch.sum(ypred_tmp * nn.functional.log_softmax(ypred_tmp, dim=1), axis=1)
        per_example_loss = sparse_categorical_crossentropy(y_pred,y_true) + 1.2 * l_p + 0.8 * l_e

        if reduction == 'mean':
	        return torch.mean(per_example_loss)
        elif reduction == 'none':
            return per_example_loss

    return criterion
 
