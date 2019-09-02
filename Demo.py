# author jiang
# -*- coding:utf-8-*-
import torch
import torchvision.transforms as transforms
import numpy as np
import dlib
from dlib import rectangle
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox
from utils.cv_plot import plot_pose_box
from utils.estimate_pose import parse_pose
import argparse
import torch.backends.cudnn as cudnn
import MobDenseNet
import DAMDNet
import cv2
import time


import utils.draw_face as df

STD_SIZE = 120
maxImgSizeForDetection=160
arch_denseMobileNetV4=['DAMDNet_v1']
cap = cv2.VideoCapture(0)
image_name = "./imgs/einstein.jpg"

def test_video(args):
    start_time=time.time()
    x=1  # displays the frame rate every 1 second
    counter=0
    checkpoint_fp='models/DAMDNet.pth.tar'
    arch='DAMDNet_v1'
    checkpoint=torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model=getattr(DAMDNet, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)
    model_dict=model.state_dict()
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')]=checkpoint[k]
    model.load_state_dict(model_dict)
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
    tri=sio.loadmat('visualize/tri.mat')['tri']-1
    tri_pts68=sio.loadmat('visualize/pats68_tri.mat')['tri']
    textureImg=cv2.imread(image_name)
    cameraImg=cap.read()[1]
    # textureCoords=df.getFaceTextureCoords(textureImg)
    # drawface=Drawing3DFace.Draw3DFace(cameraImg,textureImg,textureCoords,tri_pts68.T)
    transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    while True:
        # get a frame
        img_ori=cap.read()[1]
        imgScale=1
        scaledImg=img_ori
        if max(img_ori.shape) > maxImgSizeForDetection:
            imgScale=maxImgSizeForDetection / float(max(img_ori.shape))
            scaledImg=cv2.resize(img_ori, (int(img_ori.shape[1] * imgScale), int(img_ori.shape[0] * imgScale)))
        rects=face_detector(scaledImg, 1)

        Ps=[]  # Camera matrix collection
        poses=[]  # pose collection
        pts_res=[]
        # suffix=get_suffix(img_ori)
        for rect in rects:
            if args.dlib_landmark:
                faceRectangle=rectangle(int(rect.left() / imgScale), int(rect.top() / imgScale),
                                        int(rect.right() / imgScale), int(rect.bottom() / imgScale))

                # - use landmark for cropping
                pts=face_regressor(img_ori, faceRectangle).parts()
                pts=np.array([[pt.x, pt.y] for pt in pts]).T
                roi_box=parse_roi_box_from_landmark(pts)
            else:
                bbox=[int(rect.left()/imgScale), int(rect.top()/imgScale), int(rect.right()/imgScale), int(rect.bottom()/imgScale)]
                roi_box=parse_roi_box_from_bbox(bbox)
        img=crop_img(img_ori, roi_box)
        # forward: one step
        img=cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
        input=transform(img).unsqueeze(0)
        with torch.no_grad():
            if args.mode == 'gpu':
                input=input.cuda()
            s=time.time()
            param=model(input)
            print(time.time()-s)
            param=param.squeeze().cpu().numpy().flatten().astype(np.float32)
        # 68 pts
        pts68=predict_68pts(param, roi_box)
        # df.triDelaunay(pts68)
        densePts=predict_dense(param,roi_box)
        P, pose=parse_pose(param)
        Ps.append(P)
        poses.append(pose)
        # two-step for more accurate bbox to crop face
        if args.bbox_init == 'two':
            roi_box=parse_roi_box_from_landmark(pts68)
            img_step2=crop_img(img_ori, roi_box)
            img_step2=cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
            input=transform(img_step2).unsqueeze(0)
            with torch.no_grad():
                if args.mode == 'gpu':
                    input=input.cuda()
                param=model(input)
                param=param.squeeze().cpu().numpy().flatten().astype(np.float32)
            pts68=predict_68pts(param, roi_box)
        pts_res.append(pts68)
        pts=[]
        #draw landmark
        for indx in range(68):
            pos=(pts68[0, indx], pts68[1, indx])
            pts.append(pos)
            cv2.circle(img_ori, pos, 3, color=(255, 255, 255),thickness=-1)
        ##draw pose box
        if args.dump_pose:
            img_ori=plot_pose_box(img_ori, Ps, pts_res)
        #draw face mesh
        if args.dump_2D_face_mesh:
            img_ori=df.drawMesh(img_ori,densePts.T,tri.T)
        if args.dump_3D_face_mesh:
            pass
            # img=drawface.render(pts68)
        cv2.imshow("faceDetector", img_ori)
        counter+=1
        if (time.time() - start_time) > x:
            print("FPS: ", counter / (time.time() - start_time))
            counter=0
            start_time=time.time()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def video():
    while True:
        # get a frame
        ret, image=cap.read()
        cv2.imshow("faceDetector02", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='R3FA inference pipeline')
    parser.add_argument('-f', '--files', nargs='+',default='samples',
                        help='image files paths fed into network, single or multiple images')
    parser.add_argument('-m', '--mode', default='gpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--show_flg', default='True', type=str2bool, help='whether show the visualization result')
    parser.add_argument('--bbox_init', default='one', type=str,
                        help='one|two: one-step bbox initialization or two-step')
    parser.add_argument('--dump_res', default='true', type=str2bool, help='whether write out the visualization image')
    parser.add_argument('--dump_vertex', default='false', type=str2bool,
                        help='whether write out the dense face vertices to mat')
    parser.add_argument('--dump_ply', default='false', type=str2bool)
    parser.add_argument('--dump_pts', default='false', type=str2bool)
    parser.add_argument('--dump_roi_box', default='false', type=str2bool)
    parser.add_argument('--dump_pose', default='True', type=str2bool)
    parser.add_argument('--dump_2D_face_mesh', default='false', type=str2bool)
    parser.add_argument('--dump_3D_face_mesh', default='false', type=str2bool)
    parser.add_argument('--dlib_bbox', default='true', type=str2bool, help='whether use dlib to predict bbox')
    parser.add_argument('--dlib_landmark', default='true', type=str2bool,
                        help='whether use dlib landmark to crop image')

    args = parser.parse_args()
    test_video(args)
    # video()