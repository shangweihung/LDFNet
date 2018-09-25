# LDFNet
This repository contains the implementation details of our paper: [[arXiv:1809.09077]](https://arxiv.org/abs/1809.09077)  
"**Incorporating Luminance, Depth and Color Information by Fusion-based Networks for Semantic Segmentation**"  
by Shang-Wei Hung, Shao-Yuan Lo    

![image](https://github.com/shangweihung/LDFNet/blob/master/Model_Photos/LDFNet_Overview.PNG)

# Dependencies
* Python 3  
* Pytorch 0.4.1   

# Network Structure
LDFNet adopts two distinctive sub-networks in a parallel manner to process multiple information.  
Also, LDFNet employs the luminance information to assist the processing of the depth information in the D&Y encoder.  
  
![image](https://github.com/shangweihung/LDFNet/blob/master/Model_Photos/LDFNet_Structure.PNG)

# Evaluation
LDFNet achieves a mIoU scores of **71.3 %** on the cityscapes dataset without any pretrained model.  
For the resolution 512x1024 input, LDFNet can run at the speed of **20.6** and **27.7**FPS on a singel Titan X and GTX 1080 Ti, respectively.  


# Citing LDFNet
If you feel our LDFNet is useful for your research, please consider citying our paper:  
  
* S.-W. Hung and S.-Y. Lo, “Incorporating Luminance, Depth and Color Information by Fusion-based Networks for Semantic Segmentation,” arXiv preprint arXiv: 1809.09077, 2018.  
