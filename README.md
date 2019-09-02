# Dual Attention MobDenseNet(DAMDNet) for Robust 3D Face Alignment(ICCV2019 workshop) - Pytorch

## Note
In 'Demo.py' file, you will find how to run these codes.
In 'FaceSwap/Demo2.py' file, you will find how to run face swap code.

## Abstract
3D face alignment of monocular images is a crucial process in the recognition of faces with disguise.3D face reconstruction facilitated by alignment can restore the face structure which is helpful in detcting disguise interference.This
paper proposes a dual attention mechanism and an efficient end-to-end 3D face alignment framework.We build a stable network model through Depthwise Separable Convolution, Densely Connected Convolutional and Lightweight
Channel Attention Mechanism. In order to enhance the ability of the network model to extract the spatial features of the face region, we adopt Spatial Group-wise Feature enhancement module to improve the representation ability of the network. Different loss functions are applied
jointly to constrain the 3D parameters of a 3D Morphable Model (3DMM) and its 3D vertices. We use a variety of data enhancement methods and generate large virtual pose face data sets to solve the data imbalance problem.
The experiments on the challenging AFLW,AFLW2000-3D datasets show that our algorithm significantly improves the accuracy of 3D face alignment. Our experiments using the field DFW dataset show that DAMDNet exhibits excellent performance in the 3D alignment and reconstruction
of challenging disguised faces.The model parameters and the complexity of the proposed method are also reduced significantly.
### The framework of ours proposes method
![](https://github.com/LeiJiangJNU/DAMDNet/blob/master/figures/figure1.png)

### Details of DAMDNet
![](https://github.com/LeiJiangJNU/DAMDNet/blob/master/figures/figure2.png)

### Dual Attention mechanisms
![](https://github.com/LeiJiangJNU/DAMDNet/blob/master/figures/figure3.png)

### Data Augmentation
![](https://github.com/LeiJiangJNU/DAMDNet/blob/master/figures/figure4.png)

## Experimental results

### Comparison on other method
![](https://github.com/LeiJiangJNU/DAMDNet/blob/master/figures/nme01.png)

### Comparison on different network structures
![](https://github.com/LeiJiangJNU/DAMDNet/blob/master/figures/nme02.png)

### 3D Face Alignment Results
![](https://github.com/LeiJiangJNU/DAMDNet/blob/master/figures/results.png)

If you have any question about this code, feel free to reach me(ljiang_jnu@outlook.com)
## Citation