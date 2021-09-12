import numpy as np
import torch
from tqdm import tqdm

from DataLoader.dataloader_dose import val_transform, read_data, pre_processing
from Evaluate.evaluate_openKBP import get_3D_Dose_dif


def online_evaluation(trainer):
    list_patient_dirs = ['../../Data/OpenKBP_C3D/pt_' + str(i) for i in range(201, 241)]

    list_Dose_score = []

    with torch.no_grad():
        trainer.setting.network.eval()
        for patient_dir in tqdm(list_patient_dirs):
            patient_name = patient_dir.split('/')[-1]

            dict_images = read_data(patient_dir)
            list_images = pre_processing(dict_images)

            input_ = list_images[0]  # np.array
            gt_dose = list_images[1]
            possible_dose_mask = list_images[2]

            # Forward
            [input_] = val_transform([input_])
            input_ = input_.unsqueeze(0).to(trainer.setting.device)
            pred_dose = trainer.setting.network(input_)
            pred_dose = np.array(pred_dose.cpu().data[0, :, :, :, :])

            # Post processing and evaluation
            pred_dose[np.logical_or(possible_dose_mask < 1, pred_dose < 0)] = 0
            Dose_score = 70. * get_3D_Dose_dif(pred_dose.squeeze(0), gt_dose.squeeze(0),
                                               possible_dose_mask.squeeze(0))
            list_Dose_score.append(Dose_score)

            try:
                trainer.print_log_to_file('========> {}:  {}\n'.format(patient_name, Dose_score), 'a')
            except:
                pass

    try:
        trainer.print_log_to_file(
            '===============================================> mean Dose score: {}\n'.format(np.mean(list_Dose_score)),
            'a')
    except:
        pass
    # Evaluation score is the lower the better
    return np.mean(list_Dose_score)
