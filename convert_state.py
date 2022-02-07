import torch


def convert_state_dict(model_state_dict):

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:] if k[:7] == 'module.' else k  # remove 'module.' of DataParallel/DistributedDataParallel
        # Makes pre-trained weights trained before mobile model changes compatible with current model
        if name in ['rgb2lab.rgb_to_xyz', 'rgb2lab.fxfyfz_to_lab', 'lab2rgb.xyz_to_rgb', 'lab2rgb.lab_to_fxfyfz'] and \
                len(v.shape) == 2:
            v = torch.unsqueeze(torch.unsqueeze(v.transpose(1, 0), 0), 0)
        new_state_dict[name] = v

    return new_state_dict
