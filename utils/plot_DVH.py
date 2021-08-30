# -*- encoding: utf-8 -*-
import os

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_config(path, config_name):
    with open(os.path.join(path, config_name)) as file:
        config = yaml.safe_load(file)
        # cfg = AttrDict(config)
        # print(cfg.project_name)

    return config


# get DVH from 3D numpy
def get_DVH_from_numpy(_dose, _label, num_point=100):
    roi_dose = _dose[_label > 0]
    _DVH_y = np.linspace(0, 100, num_point + 1)
    _DVH_x = np.zeros_like(_DVH_y)
    for y_i in range(len(_DVH_y)):
        frac = _DVH_y[y_i]
        _DVH_x[y_i] = np.percentile(roi_dose, 100 - frac)

    return _DVH_x, _DVH_y


def main(configs):
    pred_dir = os.path.join(configs['output_dir'], 'Prediction')
    gt_dir = '../../Data/OpenKBP_C3D'
    save_dir = os.path.join(configs['output_dir'], 'DVH')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    list_color = ['#3398DB', '#7CB5EC', '#FF7F50', '#90ED7D', '#F7A35C', '#8085E9', '#F15C80',
                  '#E4D354', '#8D4653', '#FF69B4', '#BA55D3', '#6495ED']

    list_OAR_names = ['Brainstem',
                      'SpinalCord',
                      'RightParotid',
                      'LeftParotid',
                      'Esophagus',
                      'Larynx',
                      'Mandible']

    list_pred_patients = os.listdir(pred_dir)
    for patient_name in list_pred_patients:
        # Read prediction and gt
        pred_nii = sitk.ReadImage(os.path.join(pred_dir, patient_name, 'dose.nii.gz'))
        pred = sitk.GetArrayFromImage(pred_nii)  # denormalized during inference
        gt_nii = sitk.ReadImage(os.path.join(gt_dir, patient_name, 'dose.nii.gz'))
        gt = sitk.GetArrayFromImage(gt_nii)

        # Start plotting
        color_ptr = -1
        fig1 = plt.figure(figsize=(16, 8))

        # PTV70
        color_ptr += 1
        color_ = list_color[color_ptr]
        PTV70_nii = sitk.ReadImage(os.path.join(gt_dir, patient_name, 'PTV70.nii.gz'))
        PTV70 = sitk.GetArrayFromImage(PTV70_nii)

        DVH_x, DVH_y = get_DVH_from_numpy(gt, PTV70, num_point=100)
        plt.plot(DVH_x, DVH_y, color=color_, linestyle='-', label='PTV70')
        DVH_x, DVH_y = get_DVH_from_numpy(pred, PTV70, num_point=100)
        plt.plot(DVH_x, DVH_y, color=color_, linestyle='--')

        # PTV63
        color_ptr += 1
        color_ = list_color[color_ptr]
        if os.path.exists(os.path.join(gt_dir, patient_name, 'PTV63.nii.gz')):
            PTV63_nii = sitk.ReadImage(os.path.join(gt_dir, patient_name, 'PTV63.nii.gz'))
            PTV63 = sitk.GetArrayFromImage(PTV63_nii)

            DVH_x, DVH_y = get_DVH_from_numpy(gt, PTV63, num_point=100)
            plt.plot(DVH_x, DVH_y, color=color_, linestyle='-', label='PTV63')
            DVH_x, DVH_y = get_DVH_from_numpy(pred, PTV63, num_point=100)
            plt.plot(DVH_x, DVH_y, color=color_, linestyle='--')

        # PTV56
        color_ptr += 1
        color_ = list_color[color_ptr]
        if os.path.exists(os.path.join(gt_dir, patient_name, 'PTV56.nii.gz')):
            PTV56_nii = sitk.ReadImage(os.path.join(gt_dir, patient_name, 'PTV56.nii.gz'))
            PTV56 = sitk.GetArrayFromImage(PTV56_nii)
            DVH_x, DVH_y = get_DVH_from_numpy(gt, PTV56, num_point=100)
            plt.plot(DVH_x, DVH_y, color=color_, linestyle='-', label='PTV56')
            DVH_x, DVH_y = get_DVH_from_numpy(pred, PTV56, num_point=100)
            plt.plot(DVH_x, DVH_y, color=color_, linestyle='--')

        # All OARs
        for OAR_i in range(0, 7):
            color_ptr += 1
            color_ = list_color[color_ptr]
            OAR_file = gt_dir + '/' + patient_name + '/' + list_OAR_names[OAR_i] + '.nii.gz'
            if os.path.exists(OAR_file):
                OAR_nii = sitk.ReadImage(OAR_file)
                OAR_ = sitk.GetArrayFromImage(OAR_nii)

                DVH_x, DVH_y = get_DVH_from_numpy(gt, OAR_, num_point=100)
                plt.plot(DVH_x, DVH_y, color=color_, linestyle='-', label=list_OAR_names[OAR_i])
                DVH_x, DVH_y = get_DVH_from_numpy(pred, OAR_, num_point=100)
                plt.plot(DVH_x, DVH_y, color=color_, linestyle='--')

        plt.ylim(0, 100)
        plt.xlim(0, 80)
        plt.ylabel('Volume (%)')
        plt.xlabel('Dose (Gy)')
        plt.legend(loc='upper right', bbox_to_anchor=(1.12, 1), borderaxespad=0.5)
        plt.savefig(os.path.join(save_dir, patient_name + '.pdf'))
        plt.close(fig1)
        print(patient_name + ' done.')


if __name__ == '__main__':
    CONFIG_PATH = '../Configs'
    cfgs = load_config(CONFIG_PATH, config_name='default_config.yaml')
    main(cfgs)

"""
cnames = {
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}
"""
