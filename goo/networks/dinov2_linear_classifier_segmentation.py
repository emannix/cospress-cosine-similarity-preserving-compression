import torch
from pdb import set_trace as pb


class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, output_dim=1, normalize=None):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = torch.nn.Conv2d(in_channels, output_dim, (1,1))
        self.normalize = normalize
        if normalize is None:
            self.norm = torch.nn.Identity()
        elif normalize == 'layernorm':
            self.norm = torch.nn.LayerNorm(in_channels, elementwise_affine=False)
        elif normalize == 'batchnorm':
            self.norm = torch.nn.BatchNorm2d(in_channels, affine=False)

    def forward(self, embeddings):
        if self.normalize == 'layernorm':
            embeddings = self.norm(embeddings)
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0,3,1,2)

        if self.normalize != 'layernorm':
            embeddings = self.norm(embeddings)
        embeddings = self.classifier(embeddings)

        return embeddings
