import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def TDNN():
    net = TDNN_net()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')
    net.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), './tdnn_model.ckpt'), map_location=device)['state_dict'])
    net.eval()
    return net.to(device)

class TDNN_net(nn.Module):
    def __init__(self, feat_dim=80, train=False):
        super(TDNN_net, self).__init__()
        self.train_outputs = train
        self.conv_1 = nn.Conv1d(feat_dim,1024, 5)
        self.bn_1 = nn.BatchNorm1d(1024, affine=False)
        self.conv_2 = nn.Conv1d(1024,1024, 1)
        self.bn_2 = nn.BatchNorm1d(1024, affine=False)
        self.conv_3 = nn.Conv1d(1024, 1024, 5, dilation=2)
        self.bn_3 = nn.BatchNorm1d(1024, affine=False)
        self.conv_4 = nn.Conv1d(1024,1024, 1)
        self.bn_4 = nn.BatchNorm1d(1024, affine=False)
        self.conv_5 = nn.Conv1d(1024, 1024, 3, dilation=3)
        self.bn_5 = nn.BatchNorm1d(1024, affine=False)
        self.conv_6 = nn.Conv1d(1024,1024, 1)
        self.bn_6 = nn.BatchNorm1d(1024, affine=False)
        self.conv_7 = nn.Conv1d(1024, 1024, 3, dilation=4)
        self.bn_7 = nn.BatchNorm1d(1024, affine=False)
        self.conv_8 = nn.Conv1d(1024,1024, 1)
        self.bn_8 = nn.BatchNorm1d(1024, affine=False)
        self.conv_9 = nn.Conv1d(1024, 2000, 1)
        self.bn_9 = nn.BatchNorm1d(2000, affine=False)

        self.dense_10 = nn.Linear(6048, 512)
        self.bn_10 = nn.BatchNorm1d(512, affine=False)
        self.dense_11 = nn.Linear(512, 512)
        self.bn_11 = nn.BatchNorm1d(512, affine=False)
        self.dense_12 = nn.Linear(512, 5994)


    def forward(self, x):
        #import pdb; pdb.set_trace()
        # Repeat first and last frames to allow an xvector to be computed even for short segments
        # 13 is obtained by summing all the time delays: (5/2)*1 + (5/2)*2 + (3/2)*3 + (3/2)*4
        #x = torch.cat([x[:,:,0].resize(1,40,1).repeat(1,1,13), x, x[:,:,-1].resize(1,40,1).repeat(1,1,13)], 2)
        x = torch.cat([x[:,:,:1].repeat(1,1,13), x, x[:,:,-1:].repeat(1,1,13)], 2)
        out = self.conv_1(x)
        out = F.relu(out)
        out = self.bn_1(out)

        out = self.conv_2(out)
        out = F.relu(out)
        out = self.bn_2(out)

        out = self.conv_3(out)
        out = F.relu(out)
        out = self.bn_3(out)

        out = self.conv_4(out)
        out = F.relu(out)
        out = self.bn_4(out)
        
        out = self.conv_5(out)
        out = F.relu(out)
        out = self.bn_5(out)

        out = self.conv_6(out)
        out = F.relu(out)
        out = self.bn_6(out)
        
        out = self.conv_7(out)
        out = F.relu(out)
        out = self.bn_7(out)
        out7 = out

        out = self.conv_8(out)
        out = F.relu(out)
        out = self.bn_8(out)
        
        out = self.conv_9(out)
        out = F.relu(out)
        out = self.bn_9(out)
        out9 = out

        pooling_mean7 = torch.mean(out7, dim=2)
        pooling_std7 = torch.std(out7, dim=2, unbiased=True)

        pooling_mean9 = torch.mean(out9, dim=2)
        pooling_std9 = torch.std(out9, dim=2, unbiased=True)

        stats = torch.cat((pooling_mean7, pooling_std7, pooling_mean9, pooling_std9), 1)

        embedding_1 = self.dense_10(stats)
        out = F.relu(embedding_1)
        out = self.bn_10(out)

        embedding_2 = self.dense_11(out)
        out = F.relu(embedding_2)
        out = self.bn_11(out)

        out = self.dense_12(out)
        if self.train_outputs:
            return out, embedding_1, embedding_2 / torch.linalg.norm(embedding_2, dim=1, keepdim=True)
        else:
            return embedding_2 / torch.linalg.norm(embedding_2, dim=1, keepdim=True)


