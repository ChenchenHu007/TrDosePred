import os
import random

import SimpleITK as sitk
import cv2
import numpy as np
import torch.utils.data as data

from DataAugmentation.augmentation_DoseUformer import \
    random_flip_3d, random_rotate_around_z_axis, random_translate, to_tensor

# import monai

"""
images are always C*Z*H*W
"""


def read_data(patient_dir):
    dict_images = {}
    list_structures = ['CT',
                       'PTV70',
                       'PTV63',
                       'PTV56',
                       'possible_dose_mask',
                       'Brainstem',
                       'SpinalCord',
                       'RightParotid',
                       'LeftParotid',
                       'Esophagus',
                       'Larynx',
                       'Mandible',
                       'dose']

    for structure_name in list_structures:
        structure_file = patient_dir + '/' + structure_name + '.nii.gz'

        if structure_name == 'CT':
            dtype = sitk.sitkInt16
        elif structure_name == 'dose':
            dtype = sitk.sitkFloat32
        else:
            dtype = sitk.sitkUInt8

        if os.path.exists(structure_file):  # if the file exists
            dict_images[structure_name] = sitk.ReadImage(structure_file, dtype)
            # To numpy array (C * Z * H * W)
            dict_images[structure_name] = sitk.GetArrayFromImage(dict_images[structure_name])[np.newaxis, :, :, :]
        else:
            dict_images[structure_name] = np.zeros((1, 128, 128, 128), np.uint8)

    return dict_images


def pre_processing(dict_images):
    # PTVs
    # automatically transform, dtype: float64
    PTVs = 70.0 / 70. * dict_images['PTV70'] \
           + 63.0 / 70. * dict_images['PTV63'] \
           + 56.0 / 70. * dict_images['PTV56']  # (1, 128, 128, 128)

    # OARs
    list_OAR_names = ['Brainstem',
                      'SpinalCord',
                      'RightParotid',
                      'LeftParotid',
                      'Esophagus',
                      'Larynx',
                      'Mandible']
    # OAR_all = np.concatenate([dict_images[OAR_name] for OAR_name in list_OAR_names], axis=0)
    OAR_all = np.zeros((1, 128, 128, 128), np.uint8)
    for OAR_i in range(7):
        OAR = dict_images[list_OAR_names[OAR_i]]
        OAR_all[OAR > 0] = OAR_i + 1
    # OAR_all = OAR_all / 7.

    # CT image
    # CT = monai.transforms.ScaleIntensity(minv=0., maxv=1.)(dict_images['CT'])
    CT = dict_images['CT']
    CT = np.clip(CT, a_min=-1024, a_max=1500)  # limit the values in (-1024, 1500)
    CT = CT.astype(np.float32) / 1000.  # normalization by 1000  (1, 128, 128, 128)

    # Dose image
    dose = dict_images['dose'] / 70.  # normalization by 70

    # Possible_dose_mask, the region that can receive dose
    possible_dose_mask = dict_images['possible_dose_mask']

    list_images = [
        np.concatenate((PTVs, OAR_all, CT), axis=0),  # Input (3, 128, 128, 128)
        dose,
        possible_dose_mask
    ]
    return list_images


def train_transform(list_images):
    # Random flip
    list_images = random_flip_3d(list_images, list_axis=(0, 2), p=0.8)  # flip along Z, X axis

    # Random rotation
    list_images = random_rotate_around_z_axis(list_images,
                                              list_angles=(0, 40, 80, 120, 160, 200, 240, 280, 320),
                                              list_border_value=(0, 0, 0),
                                              list_interp=(cv2.INTER_NEAREST, cv2.INTER_NEAREST, cv2.INTER_NEAREST),
                                              p=0.3)

    # Random translation, but make use the region can receive dose is remained
    list_images = random_translate(list_images,
                                   roi_mask=list_images[2][0, :, :, :],  # the possible dose mask
                                   p=0.8,
                                   max_shift=20,
                                   list_pad_value=[0, 0, 0])

    # To torch tensor
    list_images = to_tensor(list_images)
    return list_images


def val_transform(list_images):
    list_images = to_tensor(list_images)
    return list_images


class HNC3DDataset(data.Dataset):  # a custom dataset;
    def __init__(self, num_samples_per_epoch, phase, path):
        # 'train' or 'val
        self.phase = phase
        self.num_samples_per_epoch = num_samples_per_epoch
        self.transform = {'train': train_transform, 'val': val_transform}[phase]

        self.list_case_id = {'train': [path + '/pt_' + str(i) for i in range(1, 201)],
                             'val': [path + '/pt_' + str(i) for i in range(201, 241)]}[phase]

        random.shuffle(self.list_case_id)  #
        self.sum_case = len(self.list_case_id)  # train: 200; val: 40

    def __getitem__(self, index_):
        if index_ <= self.sum_case - 1:
            case_id = self.list_case_id[index_]  # case_id: absolute path of a patient
        else:
            new_index_ = index_ - (index_ // self.sum_case) * self.sum_case
            case_id = self.list_case_id[new_index_]

        dict_images = read_data(case_id)
        list_images = pre_processing(dict_images)

        list_images = self.transform(list_images)
        return list_images

    def __len__(self):
        return self.num_samples_per_epoch  # batch_size * num_iteration -> the number of samples in our dataset


def get_loader(batch_size=1,
               num_samples_per_epoch=1,
               num_works=4,
               phase='train',
               path='../../Data/OpenKBP_C3D',
               **kwargs):
    assert os.path.exists(path), path + ' does not exist!'

    dataset = HNC3DDataset(num_samples_per_epoch=num_samples_per_epoch, phase=phase, path=path)
    loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_works,
                             pin_memory=True)

    return loader
