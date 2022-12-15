import mindspore as ms
from mindspore import nn
import math
import numpy as np

def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

def get_group_param_yolov7(model):
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.cells_and_names():
        if hasattr(v, 'beta') and isinstance(v.beta, ms.Parameter):
            pg2.append(v.beta)  # biases
        elif hasattr(v, 'bias') and isinstance(v.bias, ms.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            pg1.append(v.gamma)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, ms.Parameter):
            pg0.append(v.weight)  # apply decay
            
        # if hasattr(v, 'bias') and isinstance(v.bias, ms.Parameter):
        #     pg2.append(v.bias)  # biases
        # elif hasattr(v, 'beta') and isinstance(v.beta, ms.Parameter):
        #     pg2.append(v.beta)  # bn biases
        # if isinstance(v, nn.BatchNorm2d):
        #     pg0.append(v.gamma)  # no decay
        # elif hasattr(v, 'weight') and isinstance(v.weight, ms.Parameter):
        #     pg1.append(v.weight)  # apply decay
        
        # if hasattr(v, 'im'):
        #     if hasattr(v.im, 'implicit'):
        #         pg0.append(v.im.implicit)
        #     else:
        #         for iv in v.im:
        #             pg0.append(iv.implicit)
        # if hasattr(v, 'imc'):
        #     if hasattr(v.imc, 'implicit'):
        #         pg0.append(v.imc.implicit)
        #     else:
        #         for iv in v.imc:
        #             pg0.append(iv.implicit)
        # if hasattr(v, 'imb'):
        #     if hasattr(v.imb, 'implicit'):
        #         pg0.append(v.imb.implicit)
        #     else:
        #         for iv in v.imb:
        #             pg0.append(iv.implicit)
        # if hasattr(v, 'imo'):
        #     if hasattr(v.imo, 'implicit'):
        #         pg0.append(v.imo.implicit)
        #     else:
        #         for iv in v.imo:
        #             pg0.append(iv.implicit)
        # if hasattr(v, 'ia'):
        #     if hasattr(v.ia, 'implicit'):
        #         pg0.append(v.ia.implicit)
        #     else:
        #         for iv in v.ia:
        #             pg0.append(iv.implicit)
        # if hasattr(v, 'attn'):
        #     if hasattr(v.attn, 'logit_scale'):
        #         pg0.append(v.attn.logit_scale)
        #     if hasattr(v.attn, 'q_bias'):
        #         pg0.append(v.attn.q_bias)
        #     if hasattr(v.attn, 'v_bias'):
        #         pg0.append(v.attn.v_bias)
        #     if hasattr(v.attn, 'relative_position_bias_table'):
        #         pg0.append(v.attn.relative_position_bias_table)
        # if hasattr(v, 'rbr_dense'):
        #     if hasattr(v.rbr_dense, 'weight_rbr_origin'):
        #         pg0.append(v.rbr_dense.weight_rbr_origin)
        #     if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'):
        #         pg0.append(v.rbr_dense.weight_rbr_avg_conv)
        #     if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):
        #         pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
        #     if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'):
        #         pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
        #     if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):
        #         pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
        #     if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):
        #         pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
        #     if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):
        #         pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
        #     if hasattr(v.rbr_dense, 'vector'):
        #         pg0.append(v.rbr_dense.vector)
    return pg0, pg1, pg2

def get_lr_yolov7(opt, hyp, per_epoch_size):
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    init_lr, warmup_bias_lr, warmup_epoch, lrf = \
        hyp['lr0'], hyp['warmup_bias_lr'], hyp['warmup_epochs'], hyp['lrf']
    total_epoch, linear_lr = opt.epochs, opt.linear_lr
    if opt.optimizer == "sgd":
        with_momentum = True
    elif opt.optimizer == "momentum":
        with_momentum = True
    elif opt.optimizer == "adam":
        with_momentum = False
    elif opt.optimizer == "thor": # not use this lr
        with_momentum = False
    else:
        raise NotImplementedError

    if linear_lr:
        lf = lambda x: (1 - x / (total_epoch - 1)) * (1.0 - lrf) + lrf  # linear
    else:
        lf = one_cycle(1, lrf, total_epoch)  # cosine 1->hyp['lrf'] #1 -> 0.1

    lr_pg0, lr_pg1, lr_pg2 = [], [], []
    momentum_pg = []
    warmup_steps = max(round(warmup_epoch * per_epoch_size), 100)
    warmup_bias_steps_first = min(max(round(3 * per_epoch_size), 100), warmup_steps)
    warmup_bias_lr_first = np.interp(warmup_bias_steps_first, [0, warmup_steps], [0.0, init_lr])
    xi = [0, warmup_steps]
    for i in range(total_epoch * per_epoch_size):
        cur_epoch = i // per_epoch_size
        _lr = init_lr * lf(cur_epoch)
        if i < warmup_steps:
            lr_pg0.append(np.interp(i, xi, [0.0, _lr]))
            lr_pg1.append(np.interp(i, xi, [0.0, _lr]))
            # lr_pg2.append(np.interp(i,
            #                         [0, warmup_bias_steps_first, warmup_steps],
            #                         [warmup_bias_lr, warmup_bias_lr_first, _lr]))
            lr_pg2.append(np.interp(i, xi, [warmup_bias_lr, _lr]))
            if with_momentum:
                momentum_pg.append(np.interp(i, xi, [hyp['warmup_momentum'], hyp['momentum']]))
        else:
            lr_pg0.append(_lr)
            lr_pg1.append(_lr)
            lr_pg2.append(_lr)

    return lr_pg0, lr_pg1, lr_pg2, momentum_pg, warmup_steps


# Thor
def get_thor_lr(global_step, lr_init, decay, total_epochs, steps_per_epoch, decay_epochs=100):
    """get_model_lr"""
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    for i in range(total_steps):
        epoch = (i + 1) / steps_per_epoch
        base = (1.0 - float(epoch) / total_epochs) ** decay
        lr_local = lr_init * base
        if epoch >= decay_epochs:
            lr_local = lr_local * 0.5
        if epoch >= decay_epochs + 1:
            lr_local = lr_local * 0.5
        lr_each_step.append(lr_local)
    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]
    return learning_rate

def get_thor_damping(global_step, damping_init, decay_rate, total_epochs, steps_per_epoch):
    """get_model_damping"""
    damping_each_step = []
    total_steps = steps_per_epoch * total_epochs
    for step in range(total_steps):
        epoch = (step + 1) / steps_per_epoch
        damping_here = damping_init * (decay_rate ** (epoch / 10))
        damping_each_step.append(damping_here)
    current_step = global_step
    damping_each_step = np.array(damping_each_step).astype(np.float32)
    damping_now = damping_each_step[current_step:]
    return damping_now
