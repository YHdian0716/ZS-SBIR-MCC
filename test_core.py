import torch
import pytest
from core import MCCLoss

def test_forward():
    mcc_sk = 0.0
    mcc_ph = 0.0
    lambda_sk = 0.1
    lambda_ph = 0.2
    sk_feat = torch.randn(10, 100)
    img_feat = torch.randn(10, 100)
    neg_feat = torch.randn(10, 100)

    loss_fn = MCCLoss(mcc_sk, mcc_ph, lambda_sk, lambda_ph, device='cpu')
    loss = loss_fn.forward(sk_feat, img_feat, neg_feat)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0

if __name__ == '__main__':
    pytest.main()