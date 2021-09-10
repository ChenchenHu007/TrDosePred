import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, loss_configs=None):
        super().__init__()

        if loss_configs is not None:
            if loss_configs['type'] == 'MAE':
                self.loss = nn.L1Loss(reduction='mean')
            elif loss_configs['type'] == 'Huber':
                print('Loss function: Huber')
                self.loss = nn.HuberLoss(reduction='mean',
                                         delta=loss_configs['delta'])  # equal to SmoothL1 when delta=1.0
        else:
            self.loss = nn.L1Loss(reduction='mean')  # MAE

    def forward(self, pred, gt):
        pred_dose = pred
        gt_dose = gt[0]
        possible_dose_mask = gt[1]

        pred_dose = pred_dose[possible_dose_mask > 0]
        gt_dose = gt_dose[possible_dose_mask > 0]

        loss = self.loss(pred_dose, gt_dose)

        return loss
