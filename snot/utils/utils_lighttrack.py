import torch
import yaml
import cv2
import numpy as np


# ---------------------------------
# Functions for FC tracking tools
# ---------------------------------
def load_yaml(path, subset=True):
    file = open(path, 'r')
    yaml_obj = yaml.load(file.read(), Loader=yaml.FullLoader)

    if subset:
        hp = yaml_obj['TEST']
    else:
        hp = yaml_obj

    return hp


def to_torch(ndarray):
    return torch.from_numpy(ndarray)


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    return img


def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans, out_mode='torch'):
    """
    SiamFC type cropping
    """
    crop_info = dict()

    if isinstance(pos, float):
        pos = [pos, pos]

    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    context_xmin = round(pos[0] - c)
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        # for return mask
        tete_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad))

        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        tete_im = np.zeros(im.shape[0:2])
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original

    crop_info['crop_cords'] = [context_xmin, context_xmax, context_ymin, context_ymax]
    crop_info['empty_mask'] = tete_im
    crop_info['pad_info'] = [top_pad, left_pad, r, c]

    if out_mode == "torch":
        return im_to_torch(im_patch.copy()), crop_info
    else:
        return im_patch, crop_info



# ---------------------------------
# Functions for FC tracking tools
# ---------------------------------
def python2round(f):
    """
    use python2 round function in python3
    """
    if round(f + 1) - round(f) != 1:
        return f + abs(f) / f * 0.5
    return round(f)


def name2path_backhead(path_name, sta_num=(4, 4, 4, 4, 4), head_only=False, backbone_only=False):
    backbone_name, head_name = path_name.split('+cls_')
    if not head_only:
        # process backbone
        backbone_name = backbone_name.strip('back_')[1:-1]  # length = 20 when 600M, length = 18 when 470M
        backbone_path = [[], [], [], [], []]
        for stage_idx in range(len(sta_num)):
            for block_idx in range(sta_num[stage_idx]):
                str_idx = block_idx + sum(sta_num[:stage_idx])
                backbone_path[stage_idx].append(int(backbone_name[str_idx]))
        backbone_path.insert(0, [0])
        backbone_path.append([0])
    if not backbone_only:
        # process head
        cls_name, reg_name = head_name.split('+reg_')
        head_path = {}
        cls_path = [int(cls_name[0])]
        cls_path.append([int(item) for item in cls_name[1:]])
        head_path['cls'] = cls_path
        reg_path = [int(reg_name[0])]
        reg_path.append([int(item) for item in reg_name[1:]])
        head_path['reg'] = reg_path
    # combine
    if head_only:
        backbone_path = None
    if backbone_only:
        head_path = None
    return tuple([backbone_path, head_path])


def name2path(path_name, sta_num=(4, 4, 4, 4, 4), head_only=False, backbone_only=False):
    if '_ops_' in path_name:
        first_name, ops_name = path_name.split('_ops_')
        backbone_path, head_path = name2path_backhead(first_name, sta_num=sta_num, head_only=head_only,
                                                      backbone_only=backbone_only)
        ops_path = (int(ops_name[0]), int(ops_name[1]))
        return backbone_path, head_path, ops_path
    else:
        return name2path_backhead(path_name, sta_num=sta_num, head_only=head_only, backbone_only=backbone_only)
