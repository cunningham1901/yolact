"""

1. Run model over frame
2. Identify sandwiches
3. Make measurements and output

"""
from yolact import Yolact
from data import cfg, set_cfg

import torch
import torch.backends.cudnn as cudnn


def evalframe(net, frame):
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)

    return preds


if __name__ == "__main__":
    # Set up network with sandwich defaults
    set_cfg('yolact_plus_resnet50_config')  # Set config
    torch.set_default_tensor_type('torch.cuda.FloatTensor')  # Setup pytorch cuda
    cudann.fastest = True
    dataset = None  # Set dataset none

    net = Yolact()
    net.load_weights('weights/yolact_plus_resnet50_112_28800.pth')
    net.eval()  # set to evaluation mode
    net.cuda()  # use cuda

    path = '../images_00/0001.png'
    frame = torch.from_numpy(cv2.imread(path)).cuda().float()
    print(evalframe(net, frame))
