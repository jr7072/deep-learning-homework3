import torch

class MultiClassFocalLoss(torch.nn.Module):
    '''
        loss that will focus on hard classifications and leave
        easy ones alone. This is to solve the background problem
        during training
    '''

    def __init__(self, alpha=None, gamma=2, reduction='mean'):

        super().__init__()
    
        self.gamma = gamma
        self.reduction = reduction

        if isinstance(alpha, (int, float)):
            alpha = torch.tensor(alpha)

        # make it a vector if not already
        alpha = alpha.view(-1)
        self.register_buffer('alpha', alpha) # registers to loss device
    
    def forward(self, logits: torch.tensor, labels: torch.tensor) -> torch.tensor:

        '''
            calculates multiclass focal loss

            logits: logits from model of size (b, num_class, h, w)
            labels: ground truth of size (b, h, w)

            returns: loss of size (1)
        '''

        # use this since it is numerically stable instead of just softmax with a prob calc
        ce_loss = torch.nn.functional.cross_entropy(
            logits,
            labels,
            reduction='none'
        )

        # grab how sure the model is about classification for the pixel
        pt = torch.exp(-ce_loss)

        # get focal loss for each pixel
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha != None:
            
            # multiclass alpha
            if self.alpha.shape[0] > 1:

                # multiply alhpa here
                label_copy = labels.clone()
                at = self.alpha.gather(0, label_copy.view(-1)).view_as(label_copy)
                loss = at * focal_loss
            
            else:
                loss = self.alpha * focal_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

