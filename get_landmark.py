# author jiang
# -*- coding:utf-8-*-
import torch
import torchvision.transforms as transforms
import numpy as np
import dlib
from dlib import rectangle
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from utils.inference import parse_roi_box_from_landmark, crop_img, predict_68pts, parse_roi_box_from_bbox
import argparse
import torch.backends.cudnn as cudnn
import MobDenseNet
import cv2
import os

STD_SIZE = 120
maxImgSizeForDetection=160
arch_denseMobileNetV4=['mobdensenet_v1']


def get_image_path_list(root):
  pass

def get_landmark_2d(root,image_path):
    # 0.read image
    img_ori=cv2.imread(os.path.join(root,image_path))
    # 1. load pre-tained model
    checkpoint_fp='models/MobDenseNet.pth.tar'
    arch='mobdensenet_v1'
    checkpoint=torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model=getattr(MobDenseNet, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)
    model_dict=model.state_dict()
    if args.mode == 'gpu':
        cudnn.benchmark=True
        model=model.cuda()
    model.eval()
    # 2. load dlib model for face detection and landmark used for face cropping
    if args.dlib_landmark:
        dlib_landmark_model='models/shape_predictor_68_face_landmarks.dat'
        face_regressor=dlib.shape_predictor(dlib_landmark_model)
    if args.dlib_bbox:
        face_detector=dlib.get_frontal_face_detector()
    # 3. forward
    tri=sio.loadmat('visualize/tri.mat')['tri'] - 1
    transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    imgScale=1
    scaledImg=img_ori
    if max(img_ori.shape) > maxImgSizeForDetection:
        imgScale=maxImgSizeForDetection / float(max(img_ori.shape))
        scaledImg=cv2.resize(img_ori, (int(img_ori.shape[1] * imgScale), int(img_ori.shape[0] * imgScale)))
    rects=face_detector(scaledImg, 1)
    for rect in rects:
        if args.dlib_landmark:
            faceRectangle=rectangle(int(rect.left() / imgScale), int(rect.top() / imgScale),
                                    int(rect.right() / imgScale), int(rect.bottom() / imgScale))
            # - use landmark for cropping
            pts=face_regressor(img_ori, faceRectangle).parts()
            pts=np.array([[pt.x, pt.y] for pt in pts]).T
            roi_box=parse_roi_box_from_landmark(pts)
        else:
            bbox=[int(rect.left() / imgScale), int(rect.top() / imgScale), int(rect.right() / imgScale),
                  int(rect.bottom() / imgScale)]
            roi_box=parse_roi_box_from_bbox(bbox)
    img=crop_img(img_ori, roi_box)
    # forward: one step
    img=cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
    input=transform(img).unsqueeze(0)
    with torch.no_grad():
        if args.mode == 'gpu':
            input=input.cuda()
        param=model(input)
        param=param.squeeze().cpu().numpy().flatten().astype(np.float32)
    # 68 pts
    pts68=predict_68pts(param, roi_box)
    return pts68
if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='R3FA inference pipeline')
    parser.add_argument('-m', '--mode', default='gpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--dlib_bbox', default='true', type=str2bool, help='whether use dlib to predict bbox')
    parser.add_argument('--dlib_landmark', default='true', type=str2bool,
                        help='whether use dlib landmark to crop image')
    parser.add_argument('--root_data', default=r'', type=str)
    args=parser.parse_args()

    files=os.listdir(args.root_data)
    for i in range(len(files)):
        pts68=get_landmark_2d(args.root_data,files[i])
        img=cv2.imread(os.path.join(args.root_data,files[i]))
        # draw landmark
        for indx in range(68):
            pos=(pts68[0, indx], pts68[1, indx])
            cv2.circle(img, pos, 3, color=(255, 255, 255), thickness=-1)
        cv2.imshow("faceDetector01", img)
    cv2.destroyAllWindows()
