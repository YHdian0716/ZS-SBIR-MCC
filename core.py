import torch

import torch.nn.functional as F

class MCCLoss(torch.nn.Module):
    def __init__(self, mcc_sk, mcc_ph, lambda_sk, lambda_ph, device='cuda'):
        super(MCCLoss, self).__init__()
        self.mcc_sk = torch.Tensor([mcc_sk])[0]
        self.mcc_ph = torch.Tensor([mcc_ph])[0]
        self.lambda_sk = lambda_sk
        self.lambda_ph = lambda_ph
        self.device = device
        self.l1 = torch.nn.L1Loss()

    def forward(self, sk_feat, img_feat, neg_feat):
        """Performs forward pass of the loss function.

        Args:
            sk_feat (torch.Tensor): sketch feature
            img_feat (torch.Tensor): positive image feature
            neg_feat (torch.Tensor): negative image feature

        Returns:
            torch.Tensor: loss value
        """
        sk_feat = F.normalize(sk_feat, dim=-1)
        img_feat = F.normalize(img_feat, dim=-1)
        neg_feat = F.normalize(neg_feat, dim=-1)
        sk2sk_sim = sk_feat @ sk_feat.t()
        ph2ph_sim = img_feat @ img_feat.t()
        
        loss_mcc_sk = self.l1(sk2sk_sim.mean(), self.mcc_sk.to(self.device)) * self.lambda_sk
        loss_mcc_ph = self.l1(ph2ph_sim.mean(),self.mcc_ph.to(self.device)) * self.lambda_ph
        loss_total = loss_mcc_sk * self.lambda_sk + loss_mcc_ph * self.lambda_ph
        return loss_total
    
    

