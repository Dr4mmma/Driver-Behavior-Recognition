import torch
import torch.nn.functional as F
import cv2
import numpy as np

from .bbox import *

def detect(net, img, device):
    img = img - np.array([104, 117, 123])
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis]

    if 'cuda' in device:
        torch.backends.cudnn.benchmark = True

    img = torch.FloatTensor(img).to(device)
    BB, CC, HH, WW = img.size()
    with torch.no_grad():
        olist = net(img)

    bboxlist = []
    for i in range(len(olist) // 2):
        olist[i * 2] = F.softmax(olist[i * 2], dim=1)
    olist = [oelem.cpu() for oelem in olist]
    for i in range(len(olist) // 2):
        ocls, oreg = olist[i * 2], olist[i * 2 + 1]
        FB, FC, FH, FW = ocls.size()  # feature map size
        stride = 2**(i + 2)    # 4,8,16,32,64,128
        anchor = stride * 4
        poss = zip(*np.where(ocls[:, 1, :, :] > 0.8))
        for Iindex, hindex, windex in poss:
            axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
            score = ocls[0, 1, hindex, windex]
            loc = oreg[0, :, hindex, windex].contiguous().view(1, 4)
            priors = torch.Tensor([[axc, ayc, stride * 4, stride * 4]])
            variances = [0.1, 0.2]
            box = decode(loc, priors, variances)
            x1, y1, x2, y2 = box[0]
            bboxlist.append([x1, y1, x2, y2, score])
            break # take only the box with the highest score!

    bboxlist = np.array(bboxlist)
    if 0 == len(bboxlist):
        bboxlist = [None]

    return bboxlist


def flip_detect(net, img, device):
    img = cv2.flip(img, 1)
    b = detect(net, img, device)

    bboxlist = np.zeros(b.shape)
    bboxlist[:, 0] = img.shape[1] - b[:, 2]
    bboxlist[:, 1] = b[:, 1]
    bboxlist[:, 2] = img.shape[1] - b[:, 0]
    bboxlist[:, 3] = b[:, 3]
    bboxlist[:, 4] = b[:, 4]
    return bboxlist


def pts_to_bb(pts):
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)
    return np.array([min_x, min_y, max_x, max_y])
