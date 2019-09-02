# author jiang
# -*- coding:utf-8-*-
import torch
import torchvision.transforms as transforms
from dlib import rectangle
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors

from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import cv2
import numpy as np
from utils.cv_plot import plot_pose_box
from utils.estimate_pose import parse_pose

STD_SIZE=120

def r3fa_landmark(image,rect,model,predictor,imgScale):

    transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    Ps=[]  # Camera matrix collection
    pts_res=[]
    faceRectangle=rectangle(int(rect.left() / imgScale), int(rect.top() / imgScale), int(rect.right() / imgScale),
                            int(rect.bottom() / imgScale))

    # - use landmark for cropping
    pts=predictor(image, faceRectangle).parts()
    pts=np.array([[pt.x, pt.y] for pt in pts]).T
    roi_box=parse_roi_box_from_landmark(pts)
    #     # - use detected face bbox
    # bbox=[rects.left(), rects.top(), rects.right(), rects.bottom()]
    # roi_box=parse_roi_box_from_bbox(bbox)
    img=crop_img(image, roi_box)

    # forward: one step
    img=cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
    input=transform(img).unsqueeze(0)
    with torch.no_grad():
        input=input.cuda()
        param=model(input)
        param=param.squeeze().cpu().numpy().flatten().astype(np.float32)
    # 68 pts
    pts68=predict_68pts(param, roi_box)
    pts_res.append(pts68)
    P, pose=parse_pose(param)
    Ps.append(P)
    # # 绘制landmark
    # for indx in range(68):
    #     pos=(pts68[0, indx], pts68[1, indx])
    #     # pts.append(pos)
    #     cv2.circle(img, pos, 3, color=(255, 255, 255), thickness=-1)
    # img_ori=plot_pose_box(img, Ps, pts_res)
    return Ps,pts68[[0,1],:]