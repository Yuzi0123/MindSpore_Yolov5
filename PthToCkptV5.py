import argparse
import ast
import os
import torch
import mindspore as ms
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor


# keys_ms_add = ['model.105.grid_cell.0.param', 'model.105.grid_cell.1.param', 'model.105.grid_cell.2.param']
#
# key_map = {
#     'model.0.bn.running_mean': 'model.0.bn.moving_mean',
#     'model.0.bn.running_var': 'model.0.bn.moving_variance',
#     'model.0.bn.weight': 'model.0.bn.gamma',
#     'model.0.bn.bias': 'model.0.bn.beta',
#     "...": "..."
# }


def pytorch2mindspore(weight, is_model=False):
    torch_pth = weight
    new_params_list = []
    # par_dict = torch.load(torch_pth, map_location='cpu')
    par_dict = torch.load(torch_pth)
    
    if is_model:
        par_dict = par_dict["model"].state_dict()
    for k in par_dict.keys():
        new_k = ""
        if "0.conv.weight" in k:
            new_k = k
        elif "0.bn" in k:
            if "weight" in k:
                new_k = k[:-len("weight")] + "gamma"
            elif "bias" in k:
                new_k = k[:-len("bias")] + "beta"
            elif "running_mean" in k:
                new_k = k[:-len("running_mean")] + "moving_mean"
            elif "running_var" in k:
                new_k = k[:-len("running_var")] + "moving_variance"
        elif "1.conv.weight" in k:
            new_k = k
        elif "1.bn" in k:
            if "weight" in k:
                new_k = k[:-len("weight")] + "gamma"
            elif "bias" in k:
                new_k = k[:-len("bias")] + "beta"
            elif "running_mean" in k:
                new_k = k[:-len("running_mean")] + "moving_mean"
            elif "running_var" in k:
                new_k = k[:-len("running_var")] + "moving_variance"
        elif "3.conv.weight" in k:
            new_k = k
        elif "3.bn" in k:
            if "weight" in k:
                new_k = k[:-len("weight")] + "gamma"
            elif "bias" in k:
                new_k = k[:-len("bias")] + "beta"
            elif "running_mean" in k:
                new_k = k[:-len("running_mean")] + "moving_mean"
            elif "running_var" in k:
                new_k = k[:-len("running_var")] + "moving_variance"
        elif "5.conv.weight" in k:
            new_k = k
        elif "5.bn" in k:
            if "weight" in k:
                new_k = k[:-len("weight")] + "gamma"
            elif "bias" in k:
                new_k = k[:-len("bias")] + "beta"
            elif "running_mean" in k:
                new_k = k[:-len("running_mean")] + "moving_mean"
            elif "running_var" in k:
                new_k = k[:-len("running_var")] + "moving_variance"
        elif "7.conv.weight" in k:
            new_k = k
        elif "7.bn" in k:
            if "weight" in k:
                new_k = k[:-len("weight")] + "gamma"
            elif "bias" in k:
                new_k = k[:-len("bias")] + "beta"
            elif "running_mean" in k:
                new_k = k[:-len("running_mean")] + "moving_mean"
            elif "running_var" in k:
                new_k = k[:-len("running_var")] + "moving_variance"
        elif "10.conv.weight" in k:
            new_k = k
        elif "10.bn" in k:
            if "weight" in k:
                new_k = k[:-len("weight")] + "gamma"
            elif "bias" in k:
                new_k = k[:-len("bias")] + "beta"
            elif "running_mean" in k:
                new_k = k[:-len("running_mean")] + "moving_mean"
            elif "running_var" in k:
                new_k = k[:-len("running_var")] + "moving_variance"
        elif "14.conv.weight" in k:
            new_k = k
        elif "14.bn" in k:
            if "weight" in k:
                new_k = k[:-len("weight")] + "gamma"
            elif "bias" in k:
                new_k = k[:-len("bias")] + "beta"
            elif "running_mean" in k:
                new_k = k[:-len("running_mean")] + "moving_mean"
            elif "running_var" in k:
                new_k = k[:-len("running_var")] + "moving_variance"
        elif "18.conv.weight" in k:
            new_k = k
        elif "18.bn" in k:
            if "weight" in k:
                new_k = k[:-len("weight")] + "gamma"
            elif "bias" in k:
                new_k = k[:-len("bias")] + "beta"
            elif "running_mean" in k:
                new_k = k[:-len("running_mean")] + "moving_mean"
            elif "running_var" in k:
                new_k = k[:-len("running_var")] + "moving_variance"
        elif "21.conv.weight" in k:
            new_k = k
        elif "21.bn" in k:
            if "weight" in k:
                new_k = k[:-len("weight")] + "gamma"
            elif "bias" in k:
                new_k = k[:-len("bias")] + "beta"
            elif "running_mean" in k:
                new_k = k[:-len("running_mean")] + "moving_mean"
            elif "running_var" in k:
                new_k = k[:-len("running_var")] + "moving_variance"
        elif ".24." in k:
            new_k = k
        elif "cv1" in k:
            if "conv.weight" in k:
                new_k = k[:-len("cv1.conv.weight")] + "conv1.conv.weight"
            elif "bn.weight" in k:
                new_k = k[:-len("cv1.bn.weight")] + "conv1.bn.gamma"
            elif "bn.bias" in k:
                new_k = k[:-len("cv1.bn.bias")] + "conv1.bn.beta"
            elif "bn.running_mean" in k:
                new_k = k[:-len("cv1.bn.running_mean")] + "conv1.bn.moving_mean"
            elif "bn.running_var" in k:
                new_k = k[:-len("cv1.bn.running_var")] + "conv1.bn.moving_variance"
        elif "cv2" in k:
            if "conv.weight" in k:
                new_k = k[:-len("cv1.conv.weight")] + "conv1.conv.weight"
            elif "bn.weight" in k:
                new_k = k[:-len("cv1.bn.weight")] + "conv1.bn.gamma"
            elif "bn.bias" in k:
                new_k = k[:-len("cv1.bn.bias")] + "conv1.bn.beta"
            elif "bn.running_mean" in k:
                new_k = k[:-len("cv1.bn.running_mean")] + "conv1.bn.moving_mean"
            elif "bn.running_var" in k:
                new_k = k[:-len("cv1.bn.running_var")] + "conv1.bn.moving_variance"
        elif "cv3" in k:
            if "conv.weight" in k:
                new_k = k[:-len("cv1.conv.weight")] + "conv1.conv.weight"
            elif "bn.weight" in k:
                new_k = k[:-len("cv1.bn.weight")] + "conv1.bn.gamma"
            elif "bn.bias" in k:
                new_k = k[:-len("cv1.bn.bias")] + "conv1.bn.beta"
            elif "bn.running_mean" in k:
                new_k = k[:-len("cv1.bn.running_mean")] + "conv1.bn.moving_mean"
            elif "bn.running_var" in k:
                new_k = k[:-len("cv1.bn.running_var")] + "conv1.bn.moving_variance"

        if new_k == "":
            if not "num_batches_tracked" in k:
                print(f"Convert weight keys \"{k}\" not match.")
        else:
            # new_par_dict[new_k] = Tensor(par_dict[k].numpy())
            _param_dict = {'name': new_k, 'data': Tensor(par_dict[k].numpy())}
            new_params_list.append(_param_dict)

    ms_ckpt = f"torch2ms_{os.path.basename(weight)[:-len('.pt')]}.ckpt"
    save_checkpoint(new_params_list, ms_ckpt)

    print(f"Convert weight \"{torch_pth}\" to \"{ms_ckpt}\" finish.")

# def mindspore2pytorch(weight):
#     ms_ckpt = weight
#     # new_params_list = []
#     new_par_dict = {}
#     par_dict = ms.load_checkpoint(ms_ckpt)
#     for k in par_dict.keys():
#         new_k = ""
#         if ".105." in k:
#             if "grid_cell" in k:
#                 pass
#             else:
#                 new_k = k
#         elif "rbr" in k:
#             if "_conv.weight" in k:
#                 new_k = k[:-len("_conv.weight")] + ".0.weight"
#             elif "_conv.bias" in k:
#                 new_k = k[:-len("_conv.bias")] + ".0.bias"
#             elif "_norm.gamma" in k:
#                 new_k = k[:-len("_norm.gamma")] + ".1.weight"
#             elif "_norm.beta" in k:
#                 new_k = k[:-len("_norm.beta")] + ".1.bias"
#             elif "_norm.moving_mean" in k:
#                 new_k = k[:-len("_norm.moving_mean")] + ".1.running_mean"
#             elif "_norm.moving_variance" in k:
#                 new_k = k[:-len("_norm.moving_variance")] + ".1.running_var"
#         elif "bn" in k:
#             if "gamma" in k:
#                 new_k = k[:-len("gamma")] + "weight"
#             elif "beta" in k:
#                 new_k = k[:-len("beta")] + "bias"
#             elif "moving_mean" in k:
#                 new_k = k[:-len("moving_mean")] + "running_mean"
#             elif "moving_variance" in k:
#                 new_k = k[:-len("moving_variance")] + "running_var"
#         elif "conv" in k:
#             new_k = k

#         if new_k == "":
#             if not "num_batches_tracked" in k:
#                 print(f"Convert weight keys \"{k}\" not match.")
#         else:
#             new_par_dict[new_k] = torch.tensor(par_dict[k].asnumpy())
#             # _param_dict = {'name': new_k, 'data': Tensor(par_dict[k].numpy())}
#             # new_params_list.append(_param_dict)

#     torch_pt = f"ms2torch_{os.path.basename(weight)[:-len('.ckpt')]}.pt"
#     torch.save(new_par_dict, torch_pt)

#     print(f"Convert weight \"{ms_ckpt}\" to \"{torch_pt}\" finish.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--convert_mode', type=str, default='torch2ms', help='train mode, torch2ms/ms2torch')
    parser.add_argument('--weight', type=str, default='/home/data/lipw22/YoloV5s_MS/code/V5key.pt', help='source checkpoint file')
    parser.add_argument('--is_model', type=ast.literal_eval, default=False, help='Distribute train or not')
    opt = parser.parse_args()

    if opt.convert_mode == "torch2ms":
        pytorch2mindspore(weight=opt.weight, is_model=opt.is_model)
    # elif opt.convert_mode == "ms2torch":
    #     mindspore2pytorch(weight=opt.weight)
    else:
        raise NotImplementedError(f"Not support convert mode \"{opt.convert_mode}\"")