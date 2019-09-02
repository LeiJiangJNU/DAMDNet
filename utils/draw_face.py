# author jiang
# -*- coding:utf-8-*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from scipy.spatial import Delaunay
import scipy.io as scio


import torch
import torchvision.transforms as transforms
import MobDenseNet
import numpy as np
import cv2
import dlib
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors
from utils.cv_plot import plot_pose_box
from utils.estimate_pose import parse_pose
# from utils.render import get_depths_image, cget_depths_image, cpncc
import argparse
import torch.backends.cudnn as cudnn

STD_SIZE=120
def drawMesh(img, shape, mesh, color=(0, 255, 0)):
    """
    绘制二维人脸Mesh
    :param img:
    :param shape: 2D人脸顶点
    :param mesh: 三角划分下标
    :param color:
    :return:
    """
    i=0
    for triangle in mesh:
        if i==6:
            i=0
            point1 = shape[triangle[0]].astype(np.int32)
            point2 = shape[triangle[1]].astype(np.int32)
            point3 = shape[triangle[2]].astype(np.int32)

            cv2.line(img, (point1[0], point1[1]), (point2[0], point2[1]), color, 1)
            cv2.line(img, (point2[0], point2[1]), (point3[0], point3[1]), color, 1)
            cv2.line(img, (point3[0], point3[1]), (point1[0], point1[1]), color, 1)
        i=i+1
    return img

def triDelaunay(pts68):
    tri=Delaunay(np.array([pts68[0],pts68[1]]).T)
    fig=plt.figure()
    ax=fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_trisurf(pts68[0], pts68[1], pts68[2], triangles=tri.simplices, cmap=plt.cm.Spectral)
    plt.show()
    scio.savemat("./visualize/pats68_tri.mat",{"tri":tri})

def getFaceTextureCoords(textImag):
    checkpoint_fp='models/MobDenseNet.pth.tar'
    arch='densemobilenetv4_19'
    checkpoint=torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model=getattr(MobDenseNet, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)
    model_dict=model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')]=checkpoint[k]
    model.load_state_dict(model_dict)
    cudnn.benchmark=True
    model=model.cuda()
    model.eval()
    face_detector=dlib.get_frontal_face_detector()
    # 3. forward
    transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    img_ori=textImag
    rects=face_detector(img_ori, 1)
    for rect in rects:
            # - use detected face bbox
        bbox=[rect.left(), rect.top(), rect.right(), rect.bottom()]
        roi_box=parse_roi_box_from_bbox(bbox)
        img=crop_img(img_ori, roi_box)

        # forward: one step
        img=cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
        input=transform(img).unsqueeze(0)
        with torch.no_grad():
            input=input.cuda()
            param=model(input)
            param=param.squeeze().cpu().numpy().flatten().astype(np.float32)

        # 68 pts
        pts68=predict_68pts(param, roi_box)
        return pts68[[0,1],:]