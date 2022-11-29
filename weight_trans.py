import torch
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor

# 修改src_pt_path, dst_ckpt_path
src_pt_path = 'yolov5_stat_dict.pt'
dst_ckpt_path = './ms_yolov5s.ckpt'
torch_dict = torch.load(src_pt_path)

ms_dict = {}
new_params_list = []
for k, v in torch_dict.items():
    k_backup = k
    elem_list = k.split('.')

    if elem_list[1] in ['2', '4', '6', '8', '9', '13', '17', '20', '23']:
        k = k.replace('cv1', 'conv1')
        k = k.replace('cv2', 'conv2')
        k = k.replace('cv3', 'conv3')


    if elem_list[-2] == 'bn':
        if elem_list[-1] == 'weight':
                k = k.replace('weight', 'gamma')
        elif elem_list[-1] == 'bias':
            k = k.replace('bias', 'beta')
        elif elem_list[-1] == 'running_mean':
            k = k.replace('running_mean', 'moving_mean') 
        elif elem_list[-1] == 'running_var':
            k = k.replace('running_var', 'moving_variance')            
        else:
             continue

    _param_dict = {'name': k, 'data': Tensor(torch_dict[k_backup].numpy())}
    new_params_list.append(_param_dict) 

save_checkpoint(new_params_list, dst_ckpt_path)