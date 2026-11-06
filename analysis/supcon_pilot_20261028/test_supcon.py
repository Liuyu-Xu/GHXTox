import torch
from run import supcon

def test_pairs_mask_and_gradients():
 x=torch.tensor([[1.,0.],[.9,.1],[-1.,0.],[-.9,.1]],requires_grad=True);y=torch.tensor([1.,1.,0.,0.]);g=['a','b','c','d']
 loss,n=supcon(x,y,g,torch.ones(4,dtype=torch.bool),.1);assert n==4 and torch.isfinite(loss);loss.backward();assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0

def test_no_pairs_and_flag_exclusion():
 x=torch.randn(4,3,requires_grad=True);y=torch.tensor([1.,1.,0.,0.]);g=['a','b','c','d'];mask=torch.tensor([True,False,True,True]);loss,n=supcon(x,y,g,mask,.1);assert n==2;loss.backward();assert torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad[1])==0
 x2=torch.randn(2,3,requires_grad=True);loss,n=supcon(x2,torch.tensor([0.,1.]),['a','b'],torch.ones(2,dtype=torch.bool),.1);assert n==0 and loss.item()==0;loss.backward();assert torch.isfinite(x2.grad).all()
