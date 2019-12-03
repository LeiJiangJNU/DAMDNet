# Dual Attention MobDenseNet(DAMDNet) for Robust 3D Face Alignment(ICCV2019 workshop) - Pytorch

## Note
In 'Demo.py' file, you will find how to run these codes.
In 'FaceSwap/Demo2.py' file, you will find how to run face swap code.

## Abstract
This paper proposes a dual attention mechanism and an efficient end-to-end 3D face alignment framework.We build a stable network model through Depthwise Separable Convolution, Densely Connected Convolutional and Lightweight Channel Attention Mechanism. In order to enhance the ability of the network model to extract the spatial features of the face region, we adopt Spatial Group-wise Feature enhancement module to improve the representation ability of the network. Different loss functions are applied
jointly to constrain the 3D parameters of a 3D Morphable Model (3DMM) and its 3D vertices. We use a variety of data enhancement methods and generate large virtual pose face data sets to solve the data imbalance problem. The experiments on the challenging AFLW,AFLW2000-3D datasets show that our algorithm significantly improves the accuracy of 3D face alignment. Our experiments using the field DFW dataset show that DAMDNet exhibits excellent performance in the 3D alignment and reconstruction
of challenging disguised faces.The model parameters and the complexity of the proposed method are also reduced significantly.
## The framework of ours proposes method
![](https://github.com/LeiJiangJNU/DAMDNet/blob/master/figures/figure1.png)

## Details of DAMDNet
![](https://github.com/LeiJiangJNU/DAMDNet/blob/master/figures/figure2.png)

## Dual Attention mechanisms
![](https://github.com/LeiJiangJNU/DAMDNet/blob/master/figures/figure3.png)

## Data Augmentation
![](https://github.com/LeiJiangJNU/DAMDNet/blob/master/figures/figure4.png)

## Experimental results

## Comparison on other method
![](https://github.com/LeiJiangJNU/DAMDNet/blob/master/figures/nme01.png)

## Comparison on different network structures
![](https://github.com/LeiJiangJNU/DAMDNet/blob/master/figures/nme02.png)

## 3D Face Alignment Results
![](https://github.com/LeiJiangJNU/DAMDNet/blob/master/figures/results.png)

If you have any question about this code, feel free to reach me(ljiang_jnu@outlook.com)
## Citation
If you find this repository useful for your research, please cite the following paper:
```
@inproceedings{jiang2019dual,
  title={Dual Attention MobDenseNet (DAMDNet) for Robust 3D Face Alignment},
  author={Jiang, Lei and Wu, Xiao-Jun and Kittler, Josef},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision Workshops},
  pages={0--0},
  year={2019}
}
```
```
@inproceedings{jiang2019robust,
  title={Robust 3D Face Alignment with Efficient Fully Convolutional Neural Networks},
  author={Jiang, Lei and Wu, Xiao-Jun and Kittler, Josef},
  booktitle={International Conference on Image and Graphics},
  pages={266--277},
  year={2019},
  organization={Springer}
}
```
