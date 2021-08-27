# -*- encoding: utf-8 -*-
import os
import sys

import torch

# from monai.utils import set_determinism
if os.path.abspath('..') not in sys.path:
    sys.path.insert(0, os.path.abspath('..'))

import argparse

from DataLoader.dataloader_DoseUformer import get_loader
from NetworkTrainer.network_trainer import NetworkTrainer
from model import SwinTU3D
from online_evaluation import online_evaluation
from loss import Loss

if __name__ == '__main__':

    # added by ChenChen Hu
    print('This script has been modified by Chenchen Hu !')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch size for training (default: 2)')
    parser.add_argument('--list_GPU_ids', nargs='+', type=int, default=[1, 0],
                        help='list_GPU_ids for training (default: [1, 0])')
    parser.add_argument('--max_iter', type=int, default=80000,
                        help='training iterations(default: 80000)')
    # added by Chenchen Hu
    parser.add_argument('--latest', type=int, default=0,
                        help='load the latest model')
    parser.add_argument('--seed', type=int, default=0,
                        help='set a fixed seed')
    parser.add_argument('--model_path', type=str, default='../../Output/DoseUformer/latest.pkl')

    args = parser.parse_args()

    # set a fixed seed
    # set_determinism(args.seed)
    #  Start training
    trainer = NetworkTrainer()
    trainer.setting.project_name = 'DoseUformer'
    trainer.setting.output_dir = '../../Output/DoseUformer'
    list_GPU_ids = args.list_GPU_ids

    # setting.network is an object
    trainer.setting.network = SwinTU3D(patch_size=(4, 4, 4), depths=(2, 2, 6, 2), norm_layer=torch.nn.LayerNorm)

    trainer.setting.max_iter = args.max_iter  # 80000 or 100000

    trainer.setting.train_loader = get_loader(  # -> data.DataLoader
        batch_size=args.batch_size,  # 2
        num_samples_per_epoch=args.batch_size * 500,  # 500 iterations per epoch => 1000 samples per epoch
        phase='train',
        num_works=4
    )

    trainer.setting.eps_train_loss = 0.01
    trainer.setting.lr_scheduler_update_on_iter = True
    trainer.setting.loss_function = Loss()
    trainer.setting.online_evaluation_function_val = online_evaluation

    # filter the relative position bias table params
    relative_params = list(filter(
        lambda kv: 'relative_position_bias_table' in kv[0], trainer.setting.network.named_parameters()))
    base_params = list(filter(
        lambda kv: 'relative_position_bias_table' not in kv[0], trainer.setting.network.named_parameters()))

    trainer.setting.optimizer = torch.optim.AdamW([
        {'params': [param[1] for param in relative_params], 'weight_decay': 0.},
        {'params': [param[1] for param in base_params], }],
        lr=3e-4, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

    trainer.setting.lr_scheduler_type = 'cosine'
    trainer.setting.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.setting.optimizer,
                                                                              T_max=args.max_iter,
                                                                              eta_min=1e-7,
                                                                              last_epoch=-1)

    # trainer.set_optimizer(optimizer_type='Adam',
    #                       cfgs={
    #                           'lr': 3e-4,
    #                           'weight_decay': 1e-4
    #                       }
    #                       )
    #
    # trainer.set_lr_scheduler(lr_scheduler_type='cosine',
    #                          cfgs={
    #                              'T_max': args.max_iter,
    #                              'eta_min': 1e-7,
    #                              'last_epoch': -1
    #                          }
    #                          )

    if not os.path.exists(trainer.setting.output_dir):
        os.mkdir(trainer.setting.output_dir)
    trainer.set_GPU_device(list_GPU_ids)

    # added by Chenchen Hu
    # load the latest model when the recovery is True and the model exists.
    if args.latest and os.path.exists(args.model_path):
        trainer.init_trainer(ckpt_file=args.model_path,
                             list_GPU_ids=list_GPU_ids,
                             only_network=False)

    trainer.run()

    trainer.print_log_to_file('# Done !\n', 'a')
