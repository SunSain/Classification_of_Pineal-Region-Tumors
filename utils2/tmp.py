import torch
x=torch.tensor([[ 0.3934, -0.6554, -1.0081,    0.0964,  0.4906,  0.4924],
         [ 1.1248, -2.4764, -1.2138,    0.7377, -0.2695,  0.1876],
         [ 0.7613, -2.2103, -1.2025,    0.1480,  0.5204,  0.1685],
         [ 0.6772, -2.1346, -1.0565,    0.4778,  0.3139, -0.0196]])
y=torch.tensor([[ 0.3934, -0.6554, -1.0081,    0.0964,  0.4906,  0.4924],
         [ 1.1248, -2.4764, -1.2138,    0.7377, -0.2695,  0.1876],
         [ 0.7613, -2.2103, -1.2025,    0.1480,  0.5204,  0.1685],
         [ 0.6772, -2.1346, -1.0565,    0.4778,  0.3139, -0.0196]])
allfeats = torch.cat((x,y), 0)
print("allfeats: ",allfeats)
print("allfeats.squeeze:",allfeats.squeeze())