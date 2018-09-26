
# MyModel full model definition for Pytorch
# Sept 2017
# Eduardo Romera

# Shang-Wei Hung modified in Aug. 2018
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3,3), stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
	
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
		
        return F.relu(output)
    
class Bottleneck(nn.Module):
    def __init__(self, inCh, outCh, dilated, drop):
        super(Bottleneck, self).__init__()			

        self.conv_1x1 = nn.Conv2d(inCh, outCh, 1, bias=False)
        self.bn_1x1 = nn.BatchNorm2d(outCh, eps=1e-3)
		
        self.conv_b = nn.Conv2d(outCh, outCh, 3, padding=dilated, dilation=dilated, bias=False)
        self.bn_b = nn.BatchNorm2d(outCh, eps=1e-3)	
		
        self.dropout = nn.Dropout2d(drop, inplace=True)
		
    def forward(self, input):

        output = self.conv_1x1(input)
        output = self.bn_1x1(output)
        output = F.relu_(output)

        output = self.conv_b(output)
        output = self.bn_b(output)
        output = F.relu_(output)  		
		
        if (self.dropout.p != 0):
            output = self.dropout(output)		

        return torch.cat([input, output], 1)
		
class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3,1), padding=(1,0))

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), padding=(0,1))

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3,1), padding=(dilated,0), dilation=(dilated,1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), padding=(0,dilated), dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)
		
		
class TransitionLayer(nn.Module):
    def __init__(self, inCh, outCh):
        super(TransitionLayer, self).__init__()
		
        self.conv = nn.Conv2d(inCh, outCh, 1 ,stride=1, bias=False)
        self.bn = nn.BatchNorm2d(outCh, eps=1e-3)
		
    def forward(self, input):
	
        output = self.conv(input)
        output = self.bn(output)
        output = F.relu(output)	
        return F.avg_pool2d(output,2,stride=2)

class DenseBlock(nn.Module):
    def __init__(self, nLayer, inCh, growthRate, dilated, drop):
        super(DenseBlock, self).__init__()
		
        self.denseblock = self._make_layer(nLayer, inCh, growthRate, dilated, drop)
		
    def _make_layer(self, nLayer, inCh, growthRate, dilated, drop):
	
        layers = []
        if dilated == 0:
            dilated = [1,1,2,2]
            for layer in range(nLayer):
                layers.append(Bottleneck(inCh+layer*growthRate, growthRate, dilated[layer], drop))
        elif dilated == 1:
            dilated = [1,1,1]
            for layer in range(nLayer):
                layers.append(Bottleneck(inCh+layer*growthRate, growthRate, dilated[layer], drop))
        else:  # with dilated convolutiom
            dilated = [2,4,8,16]
            for layer in range(nLayer):
                layers.append(Bottleneck(inCh+layer*growthRate, growthRate, dilated[layer], drop))				
			
        return nn.Sequential(*layers)
		
    def forward(self, input):	
        return self.denseblock(input)
		
		
class Encoder(nn.Module):
    def __init__(self, growthRate, drop):        #growthRate, drop for DenseNet
        super().__init__()
		#---------------------ERFNET---RGB--------------------------#
        self.initial_block_rgb = DownsamplerBlock(3, 16)
        self.second_downsampleblock_rgb=DownsamplerBlock(16, 64)
        	
        self.layers_1_rgb = nn.ModuleList()        
        for x in range(0, 5):    #5 times #5 modules
			      self.layers_1_rgb.append(non_bottleneck_1d(64, 0.03, 1)) 	   		
		     
        self.third_downsampleblock_rgb=DownsamplerBlock(64, 128) 
        self.layers_2_rgb = nn.ModuleList()
        
        for x in range(0, 2):    #2 times #8 modules
            self.layers_2_rgb.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers_2_rgb.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers_2_rgb.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers_2_rgb.append(non_bottleneck_1d(128, 0.3, 16))  

        #---------------------DenseNet---Depth----------------------#
		
        nBlock = [4,3,4]  # num of modules in D&Y branch, respectively
        dilated = [0,1,2]

        # initial block
        inCh = 16
        self.downsampling_1 = DownsamplerBlock(2, 16)
		
		# -1st block
        self.block_minus1 = DenseBlock(nBlock[0], inCh, growthRate, dilated[0], drop)
        inCh = int(inCh+nBlock[0]*growthRate)  # 16 + 4 * 42 = 184
        self.trans_minus1 = TransitionLayer(inCh, int(inCh/2))
        inCh = int(inCh/2)  # 0.5 * 184 = 92
		
        self.downchannel_1=nn.Conv2d(184, 16, 1 ,stride=1, bias=False)
		
        # 1st block
        
        self.block_1 = DenseBlock(nBlock[1], inCh, growthRate, dilated[1], drop)
        inCh = int(inCh+nBlock[1]*growthRate)  # 92 + 3 * 42 = 218
        self.trans = TransitionLayer(inCh, int(inCh/2))
        inCh = int(inCh/2)  # 0.5 * 218 = 109
		
        self.upchannel_2=nn.Conv2d(92, 64, 1 ,stride=1, bias=False)
        self.downchannel_2=nn.Conv2d(218, 64, 1 ,stride=1, bias=False)
		
        # 2nd block
        self.block_2 = DenseBlock(nBlock[2], inCh, growthRate, dilated[2], drop)  # 109 + 4 * 42 =277 	
				 
        self.upchannel_3=nn.Conv2d(109, 128, 1 ,stride=1, bias=False)
        self.downchannel_3=nn.Conv2d(277, 128, 1 ,stride=1, bias=False)
  
    def forward(self, rgb_input,d_input):
        output_rgb = self.initial_block_rgb(rgb_input)
        output_d = self.downsampling_1(d_input)
		
        output_d = self.block_minus1(output_d)                 
        output_d_down1 = self.downchannel_1(output_d)          
        #-------fuse_1------
        output_rgb=output_rgb+output_d_down1                                    
        #-------------------
        
        output_rgb = self.second_downsampleblock_rgb(output_rgb)
        output_d = self.trans_minus1(output_d)
        output_d_up2=self.upchannel_2(output_d)                 
        #-------fuse_2------
        output_rgb=output_rgb+output_d_up2                                     
        #-------------------
		
        for layer in self.layers_1_rgb:
            output_rgb = layer(output_rgb)
			
        output_d = self.block_1(output_d)
        output_d_down2=self.downchannel_2(output_d)                            
        #-------fuse_3----
        output_rgb=output_rgb+output_d_down2
        #------------------	
        
        output_rgb = self.third_downsampleblock_rgb(output_rgb)
        output_d = self.trans(output_d)
        output_d_up3=self.upchannel_3(output_d)
		#-------fuse_4-------
        output_rgb=output_rgb+output_d_up3                                   
		#------------------	
        
        for layer in self.layers_2_rgb:
            output_rgb = layer(output_rgb)
        
        output_d = self.block_2(output_d)
        output_d = self.downchannel_3(output_d)          
        #------fuse_5--------
        output=output_rgb+output_d
        #-------------------	

        return output


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
		
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
	
        output = self.conv(input)
        output = self.bn(output)
		
        return F.relu(output)

class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.layers_1 = nn.ModuleList()
        self.upsample_1=UpsamplerBlock(128, 64)
        self.layers_1.append(non_bottleneck_1d(64, 0, 1))
        self.layers_1.append(non_bottleneck_1d(64, 0, 1))
		#-----------------------------------------------------------------
        self.layers_2 = nn.ModuleList()
        self.upsample_2=UpsamplerBlock(64, 16)
        self.layers_2.append(non_bottleneck_1d(16, 0, 1))
        self.layers_2.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2)

    def forward(self, input):
	
        output = input                  
        output = self.upsample_1(output)
		                                    
        for layer in self.layers_1:
            output = layer(output)
        output = self.upsample_2(output)
				                    						
        for layer in self.layers_2:
            output = layer(output)
        output = self.output_conv(output)
						                    				

        return output


			
# LDFNet

class Net(nn.Module):
    def __init__(self, num_classes, giveEncoder=None):
	    # use encoder to pass pretrained encoder
		
        super().__init__()

        if (giveEncoder == None):
            self.encoder = Encoder(growthRate=42, drop=0.05)
        else:
            self.encoder = giveEncoder
			
        self.decoder = Decoder(num_classes)
        # Only in encoder mode:
        self.encoder_output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

		
    def forward(self, rgb_input,d_input, only_encoder=False):
	
        if only_encoder:
            #print('ERROR!')
            output = self.encoder(rgb_input,d_input)
			
            return self.encoder_output_conv(output)
			
        else:
            output = self.encoder(rgb_input,d_input)
			
            return self.decoder.forward(output)
