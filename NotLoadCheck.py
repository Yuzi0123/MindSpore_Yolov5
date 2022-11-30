import mindspore as ms
from mindspore import nn

class Net(nn.Cell):
    def __init__(self) -> None:
        super().__init__()

net = Net()
param_dict = ms.load_checkpoint('/data1/lurenjie-yolov5s/yolov5s_mindspore/torch2ms_V5key.ckpt')
param_not_load = ms.load_param_into_net(net, param_dict)
print(param_not_load)