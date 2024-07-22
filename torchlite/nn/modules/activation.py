from torchlite import Tensor as Tensor

a=Tensor([2])
b=Tensor([1,2])
c=a*b
print(c.backward())