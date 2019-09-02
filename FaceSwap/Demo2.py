import dlib
import numpy as np

import models
import FaceSwap.NonLinearLeastSquares as NonLinearLeastSquares
import FaceSwap.ImageProcessing as ImageProcessing

from FaceSwap.drawing import *
import torch.backends.cudnn as cudnn
import FaceSwap.FaceRendering as FaceRendering
import FaceSwap.utils_ as utils
import time
import torch
import MobDenseNet
import cv2
print ("Press T to draw the keypoints and the 3D model")
print ("Press R to start recording to a video file")

#you need to download shape_predictor_68_face_landmarks.dat from the link below and unpack it where the solution file is
#http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2

#loading the keypoint detection model, the image and the 3D model
predictor_path = "../models/shape_predictor_68_face_landmarks.dat"
image_name = "../imgs/jolie.jpg"
#the smaller this value gets the faster the detection will work
#if it is too small, the user's face might not be detected
maxImageSizeForDetection = 160

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
# 加载R3FA人脸检测模型
checkpoint_fp='../models/MobDenseNet.pth.tar'
arch='mobdensenet_v1'
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


mean3DShape, blendshapes, mesh, idxs3D, idxs2D = utils.load3DFaceModel("candide.npz")

projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])

modelParams = None
lockedTranslation = False
drawOverlay = True
cap = cv2.VideoCapture(0)
writer = None
cameraImg = cap.read()[1]
# Img=cap.read()[1]
textureImg = cv2.imread(image_name)
textureCoords = utils.getFaceTextureCoords(textureImg, mean3DShape, blendshapes, idxs2D, idxs3D, detector, model,predictor)
renderer = FaceRendering.FaceRenderer(cameraImg, textureImg, textureCoords, mesh)
start_time=time.time()
x=1  # displays the frame rate every 1 second
counter=0



while True:
    cameraImg = cap.read()[1]
    Img=cap.read()[1]
    shapes2D = utils.getFaceKeypoints(cameraImg, detector, model, predictor,maxImageSizeForDetection)

    if shapes2D is not None:
        for shape2D in shapes2D:
            #3D model parameter initialization
            modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], shape2D[:, idxs2D])

            #3D model parameter optimization
            modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian, ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], shape2D[:, idxs2D]), verbose=0)

            #rendering the model to an image
            shape3D = utils.getShape3D(mean3DShape, blendshapes, modelParams)
            renderedImg = renderer.render(shape3D)

            #blending of the rendered face with the image
            mask = np.copy(renderedImg[:, :, 0])
            renderedImg = ImageProcessing.colorTransfer(cameraImg, renderedImg, mask)
            # cameraImg = ImageProcessing.blendImages(renderedImg, cameraImg, mask)
       

            #drawing of the mesh and keypoints
            if drawOverlay:
                drawPoints(cameraImg, shape2D.T)
                # drawPose(cameraImg,shape2D.T,Ps)
                # drawProjectedShape(cameraImg, [mean3DShape, blendshapes], projectionModel, mesh, modelParams, lockedTranslation)

    if writer is not None:
        writer.write(cameraImg)

    counter+=1
    if (time.time() - start_time) > x:
        print("FPS: ", counter / (time.time() - start_time))
        counter=0
        start_time=time.time()
    # cv2.imshow('image', Img)
    cv2.imshow('Warpimage', cameraImg)
    key = cv2.waitKey(1)

    if key == 27:
        break
    if key == ord('t'):
        drawOverlay = not drawOverlay
    if key == ord('r'):
        if writer is None:
            print ("Starting video writer")
            writer = cv2.VideoWriter("../out.avi", cv2.cv.CV_FOURCC('X', 'V', 'I', 'D'), 25, (cameraImg.shape[1], cameraImg.shape[0]))

            if writer.isOpened():
                print ("Writer succesfully opened")
            else:
                writer = None
                print ("Writer opening failed")
        else:
            print ("Stopping video writer")
            writer.release()
            writer = None
