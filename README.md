# LDFNet
This repository contains the implementation details of our paper:
"Incorporating Luminance, Depth and Color Information by Fusion-based Networks for Semantic Segmentation"
by Shang-Wei Hung, Shao-Yuan Lo


# Dependencies
Python
Pytorch 


# Network Structure
LDFNet adopts two distinctive sub-networks in a parallel manner to process multiple information. Also, LDFNet employs the luminance information to assist the processing of the depth information in the D&Y encoder.


# Evaluation
LDFNet achieves a mIoU scores of 71.3 on the cityscapes dataset without any pretrained model.
For the resolution 512x1024 input, LDFNet can run at the speed of 20.6 and 27.7 FPS on a singel Titan X and GTX 1080 Ti, respectively.


# Citing LDFNet
If you feel our LDFNet is useful for your research, please consider citying our paper:
