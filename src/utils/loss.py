import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for Multi-class classification.
    Giúp model tập trung học các mẫu khó (có xác suất dự đoán đúng thấp).
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        # alpha có thể là 1 list các trọng số cho từng class
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        # inputs: (Batch, Num_classes) - logits
        # targets: (Batch) - class indices
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt là xác suất dự đoán đúng của class mục tiêu
        
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            # Lấy alpha tương ứng với từng target
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_loss_function(loss_type='ce', device='cpu'):
    """
    Trả về hàm loss dựa trên tham số truyền vào.
    - ce: CrossEntropyLoss bình thường
    - focal: FocalLoss (Mặc định penalty cao cho các mẫu khó học)
    - weighted: CrossEntropyLoss với trọng số cao hơn cho class Sad (index 2)
    """
    if loss_type == 'focal':
        # Class weights: Happy=1.0, Neutral=1.5, Sad=2.5, Angry=1.0
        # Sad được boost mạnh nhất, Neutral cũng được boost nhẹ.
        alpha_weights = [1.0, 1.5, 2.5, 1.0] 
        return FocalLoss(alpha=alpha_weights, gamma=2.0, reduction='mean')
    elif loss_type == 'weighted':
        weights = torch.tensor([1.0, 1.5, 2.5, 1.0], dtype=torch.float32).to(device)
        return nn.CrossEntropyLoss(weight=weights)
    else:
        return nn.CrossEntropyLoss()
