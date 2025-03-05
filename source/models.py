import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl

# model for computing channel means and standard deviations
class ChannelStats(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_mean = torch.mean(x, axis=1)
        batch_sd = torch.std(x, axis=1)
        return batch_mean, batch_sd


class EmbeddingNet(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone

    def forward(self, x):
        return self.backbone(x)


class TransferLearning(nn.Module):
    def __init__(self, model=None):
        super(TransferLearning, self).__init__()
        if model is None:
            model = torchvision.models.resnet50(pretrained=True)
            model = nn.Sequential(*(list(model.children())[:-1]))
        self.model = model

    def forward(self, x):
        # x is an image of B, C, W, H; where B = mini batch size, C is e.g. 5 in the case of
        # cell painting data
        # we replicate each channel 3 times so it passes throught the network, then concatenate
        # the resulting embeddings
        emb_list = []
        for c in range(x.shape[1]):
            xrep = x[:, c : c + 1, ...].repeat(1, 3, 1, 1)
            emb = self.model(xrep)
            emb = torch.reshape(emb, (emb.shape[0], emb.shape[1]))
            emb_list.append(emb)

        embs_stacked = torch.cat(emb_list, axis=1)
        return embs_stacked
