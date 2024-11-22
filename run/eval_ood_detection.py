
from PIL import Image
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from torchvision import transforms as T

import sys
sys.path.append('.')

from goo.methods.self_supervised.proteus_vmfsne7_reversed import Proteus as TinTeM
from goo.methods.self_supervised.proteus import Proteus

import os

from pdb import set_trace as pb
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import normalize
from sklearn import metrics
import faiss


def eval_ood_KNN(indist_labelled, indist_val, outdist, k=1):

    indist_labelled = normalize(indist_labelled, axis=1)
    indist_val = normalize(indist_val, axis=1)
    outdist = normalize(outdist, axis=1)

    index = faiss.IndexFlatIP(indist_labelled.shape[1])
    index.add(np.ascontiguousarray(indist_labelled))
    D_indist, I = index.search(indist_val, k)

    D_indist = D_indist.min(axis=1)

    index = faiss.IndexFlatIP(indist_labelled.shape[1])
    index.add(np.ascontiguousarray(indist_labelled))
    D_outdist, I = index.search(outdist, k)

    D_outdist = D_outdist.min(axis=1)

    labels = [np.zeros(D_outdist.shape[0]), np.ones(D_indist.shape[0])]
    labels = np.concatenate(labels)
    dists = np.concatenate([D_outdist, D_indist]).squeeze()

    aucroc = metrics.roc_auc_score(labels, dists)

    fp95 = np.quantile(D_indist, 0.05)
    fp95_in = np.mean(D_outdist > fp95)
    return aucroc, fp95_in


# orig_mod = Proteus.load_from_checkpoint("weights/proteus_vitt_vits.ckpt")
orig_mod = TinTeM.load_from_checkpoint("weights/tintem_vitt_vits.ckpt")
orig_mod = orig_mod.student.backbone

orig_mod.cuda()
orig_mod.eval()

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=256, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    torchvision.transforms.CenterCrop(size=224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

knn_alpha = 1.0
knn_k = 1

ID_training = torchvision.datasets.CIFAR10(download=True, train=True, root='datasets/', transform=transform)
np.random.seed(0)
ID_training.data = ID_training.data[np.random.choice(ID_training.data.shape[0], size=int(knn_alpha*ID_training.data.shape[0]), replace=False)]

ID_val = torchvision.datasets.CIFAR10(download=True, train=False, root='datasets/', transform=transform)
OOD = torchvision.datasets.CIFAR100(download=True, train=False, root='datasets/', transform=transform)

ID_training = DataLoader(ID_training, batch_size=64, shuffle=False, num_workers=4, drop_last=False)
ID_val = DataLoader(ID_val, batch_size=64, shuffle=False, num_workers=4, drop_last=False)
OOD = DataLoader(OOD, batch_size=64, shuffle=False, num_workers=4, drop_last=False)

ID_training_data = []
ID_val_data = []
OOD_data = []

for inputs, targets in tqdm(ID_training, desc="Training", leave=False):
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            inputs = inputs.cuda()
            ID_training_data.append(orig_mod(inputs).cpu())

for inputs, targets in tqdm(ID_val, desc="Training", leave=False):
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
          inputs = inputs.cuda()
          ID_val_data.append(orig_mod(inputs).cpu())

for inputs, targets in tqdm(OOD, desc="Training", leave=False):
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
          inputs = inputs.cuda()
          OOD_data.append(orig_mod(inputs).cpu())

ID_training_data = torch.cat(ID_training_data, dim=0)
ID_training_data = torch.nn.functional.normalize(ID_training_data, dim=1)

ID_val_data = torch.cat(ID_val_data, dim=0)
ID_val_data = torch.nn.functional.normalize(ID_val_data, dim=1)

OOD_data = torch.cat(OOD_data, dim=0)
OOD_data = torch.nn.functional.normalize(OOD_data, dim=1)

print(eval_ood_KNN(ID_training_data.numpy(), ID_val_data.numpy(), OOD_data.numpy(), k=knn_k))


