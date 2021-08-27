import os
import sys

import torch

if os.path.abspath('..') not in sys.path:
    sys.path.insert(0, os.path.abspath('..'))

from DoseUformer.model import SwinTU3D
from DoseUformer.loss import Loss
from DataLoader.dataloader_DoseUformer import get_loader
from torch_lr_finder import LRFinder

if __name__ == '__main__':
    m = SwinTU3D(in_chans=3, embed_dim=96, depths=(2, 2, 4, 2))

    loader = get_loader(batch_size=2, num_samples_per_epoch=2 * 500)
    criterion = Loss()
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-7, weight_decay=1e-2)
    lr_finder = LRFinder(m, optimizer, criterion)
    lr_finder.range_test(loader, end_lr=200, num_iter=200)
    lr_finder.plot()
    lr_finder.reset()
