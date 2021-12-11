import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from ibl import models
from ibl.pca import PCA
from torch import nn
import torch.nn.functional as F
from ibl.pca import PCA
from torch.nn import Parameter
import time
from torch.autograd import Variable
import cv2
from ibl.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict, write_json


def get_model(moderlpath,modelname):
    # base_model = models.create("mbv2_ca")
    # base_model = models.create("mobilenet_v2")
    # base_model = models.create("vgg16")
    # pool_layer = models.create('netvlad', dim=base_model.feature_dim)
    # model = models.create('embednet', base_model, pool_layer)
    # model.cuda()
    # checkpoint = torch.load("/home/lhw/ros/OpenIBL/examples/model_best.pth.tar")
    # # checkpoint = torch.load("/home/lhw/ros/OpenIBL/examples/mobilenetv2.pth.tar")
    # # checkpoint = torch.load("/home/lhw/ros/OpenIBL/examples/moca.pth.tar")
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in checkpoint['state_dict'].items():
    #     name = k[7:]  # remove `module.`
    #     new_state_dict[name] = v
    # #load params
    # model.load_state_dict(new_state_dict)
    if modelname == 'vgg16':
        model = torch.load(moderlpath + 'vgg.pt')
    elif modelname == 'mobilenet_v2':
        model = torch.load(moderlpath +'mobilenet_v2.pt')
    else:
        model = torch.load(moderlpath +'mbv2_ca.pt')
    # if torch.cuda.is_available():
    model.cuda().eval()
    return model
def get_pca(moderlpath,modelname):

    if modelname == 'vgg16':
        pca = PCA(4096, True, moderlpath +"pca_params_model_best.h5")
    elif modelname == 'mobilenet_v2':
        pca = PCA(4096, True,moderlpath + "pca_mobilenetv2.h5")
    else:
        pca = PCA(4096, True, moderlpath +"pca_params_moca.h5")
    pca.load()
    return pca
transformer = transforms.Compose([ transforms.Resize((480, 640)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
                                                       std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])])
def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray
default_config = {
    'model_name': 'vgg16', #moca,mobilenet_v2
}
class FeatureNet:
    def __init__(self, moderlpath,modelname):
        self.net = get_model(moderlpath,modelname)
        self.pca = get_pca(moderlpath,modelname)

    def infer(self, image):
          # image = Image.open(imagepath).convert('RGB')
        # img = cv2.imread(imagepath)
        img = Image.fromarray(image)
        img = transformer(img)
        img = to_torch(img).cuda()
        with torch.no_grad():
            des = self.net(img.unsqueeze(0))
        des = F.normalize(des[1], p=2, dim=-1)
        if (self.pca is not None):
            des = self.pca.infer(des)
        des = des.cpu().numpy()
        # print(type(des))
        # des = np.asarray(des)
        # print(type(des))
        return des
        