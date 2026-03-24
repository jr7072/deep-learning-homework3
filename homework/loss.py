import torch

class MultiClassFocalLoss(torch.nn.Module):
    '''
        loss that will focus on hard classifications and leave
        easy ones alone. This is to solve the background problem
        during training
    '''

    def __init__(self, alpha=.25, gamma=2, reduction='mean'):

        super().__init__()
    
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    
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
        loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()

