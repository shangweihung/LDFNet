# Main code for training ERFNet model in Cityscapes dataset
# Sept 2017
# Eduardo Romera
#
# Shang Wei Hung modified in June. 2018 for LDFNet
#######################

import os
import sys
import time
import torch
import importlib

from argparse import ArgumentParser
from shutil import copyfile

import torch.cuda as cuda
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader
from datasetTrain_grayscale import cityscapes, CamVid, ITRI
from iouEval import iouEval, getColorEntry
from myTool_gray import *

import lovasz_losses as L

		
def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):

    torch.save(state, filenameCheckpoint)
	
    if is_best:
        print ("Saving model as best")
        torch.save(state, filenameBest)
		
		
def train(args, model, classNum, epochNum, encoderOnly=False):

    start_epoch = 1
    best_acc = 0
	
	# === Dataset Processing === #	
    if args.dataset == 'cityscapes':
        co_transform = MyCoTransform(encoderOnly, dataAugment=True, height=args.height)
        co_transform_val = MyCoTransform(encoderOnly, dataAugment=False, height=args.height)
        dataDir = '/media/commlab/TenTB/swhung/SegNet/Cityscapes/'
        dataset_train = cityscapes(dataDir, co_transform, 'train')
        dataset_val = cityscapes(dataDir, co_transform_val, 'val')
        saveDir = f'../save/{args.saveDir}' # #

    loader_train = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batchSize, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batchSize, shuffle=False)
    
    # === Optimization Setting === #
	
    # ** optimizer
    if args.optimizer == 'adam':	
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
	
	# ** learing rate scheduler
    my_lambda = lambda epoch: pow((1-((epoch-1)/epochNum)),0.9)  # poly
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=my_lambda)

	# ** apply loss function
    classWeight = getClassWeight(args.dataset, classNum)	
    if args.cuda:
        classWeight = classWeight.cuda()

    criterion = CrossEntropyLoss2d(weight=classWeight, ignore_index=19)
	
	# === save information in .txt files === #		
    if (encoderOnly):
        automated_log_path = saveDir + "/automated_log_encoder.txt"
        modeltxtpath = saveDir + "/model_txt_encoder.txt"
    else:
        automated_log_path = saveDir + "/automated_log.txt"
        modeltxtpath = saveDir + "/model_txt.txt"    

    if (not os.path.exists(automated_log_path)):  # do not add first line if it exists 
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))

	# === Training === #		
    for epoch in range(start_epoch, epochNum+1):
	
        print("----- TRAINING - EPOCH", epoch, "-----")
		
        model.train()

        scheduler.step(epoch-1)

        epoch_loss = []
        time_train = []

        if (args.doEvalTrain):
            iouEvalTrain = iouEval(classNum)

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("learning rate: ", param_group['lr'])
            usedLr = float(param_group['lr'])

		# ** training iteration
        for iter, (images, labels) in enumerate(loader_train):
            start_time = time.time()

            slice=torch.split(images,1,1)
            rgb=torch.cat((slice[0],slice[1],slice[2]),1)
            d=torch.cat((slice[3],slice[4]),1)                 #depth and luminance
			

            if args.cuda:
                rgb_inputs = rgb.cuda()
                d_input = d.cuda()
                targets = labels.cuda()
            
            img_size=list(targets.size())[2:4]

            
			# run the model
            if args.onlyWholeNet:
                outputs = model(inputs)
            else:
                outputs = model(rgb_inputs,d_input,only_encoder=encoderOnly)			
			
			# run the back-propagation
            loss = criterion(outputs, targets[:, 0])
            optimizer.zero_grad()			
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)

            if (args.doEvalTrain):
                iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)     
			
			# print the training loss information
            if args.iter_loss > 0 and iter % args.iter_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average:0.4} (epoch: {epoch}, iter: {iter})', 
                        "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batchSize))

            
        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        
        iouTrain = 0		
        if (args.doEvalTrain):
            iouTrain, iou_classes = iouEvalTrain.getIoU()
            iouStr = getColorEntry(iouTrain)+'{:0.2f}'.format(iouTrain*100) + '\033[0m'
            print ("EPOCH IoU on TRAIN set: ", iouStr, "%")  

        if epoch<=10 or epoch>=70:
            with torch.no_grad():
		    
		        # Validate on 500 val images after each epoch of training
                print("----- VALIDATING - EPOCH", epoch, "-----")
                
                model.eval()
		        
                epoch_loss_val = []
                time_val = []
                
                if (args.doEvalVal):
                    iouEvalVal = iouEval(classNum)
                
		        # ** valadation iteration
                for iter, (images, labels) in enumerate(loader_val):
		        
                    start_time = time.time()
		        	
                    slice=torch.split(images,1,1)
                    rgb=torch.cat((slice[0],slice[1],slice[2]),1)
                    d=torch.cat((slice[3],slice[4]),1)                 #depth and luminance
            
                    if args.cuda:
                        rgb_inputs = rgb.cuda()
                        d_input = d.cuda()
                        targets = labels.cuda()
                
                    img_size=list(targets.size())[2:4]
            
                
		    	    # run the model
                    if args.onlyWholeNet:
                        outputs = model(inputs)
                    else:
                        outputs = model(rgb_inputs,d_input,only_encoder=encoderOnly) 
                
                    loss = criterion(outputs, targets[:, 0])
		        	
                    epoch_loss_val.append(loss.item())
                    time_val.append(time.time() - start_time)
                
                    # Add batch to calculate TP, FP and FN for iou estimation
                    if (args.doEvalVal):
                        iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
		        	
		        	# print the valadation loss information
                    if args.iter_loss > 0 and iter % args.iter_loss == 0:
                        average = sum(epoch_loss_val) / len(epoch_loss_val)
                        print(f'VAL loss: {average:0.4} (epoch: {epoch}, iter: {iter})', 
                                "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batchSize))
                               
            average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
		    
		    # print epoch val IoU accuracy
            iouVal = 0
            if (args.doEvalVal):
                iouVal, iou_classes = iouEvalVal.getIoU()
                iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
                print ("EPOCH IoU on VAL set: ", iouStr, "%") 
               
            # remember best valIoU and save checkpoint
            if iouVal == 0:
                current_acc = average_epoch_loss_val
            else:
                current_acc = iouVal 
		    	
            is_best = current_acc > best_acc
            best_acc = max(current_acc, best_acc)
		    
            if encoderOnly:
                filenameCheckpoint = saveDir + '/checkpoint_enc.pth.tar'
                filenameBest = saveDir + '/model_best_encoder.pth.tar'    
            else:
                filenameCheckpoint = saveDir + '/checkpoint.pth.tar'
                filenameBest = saveDir + '/model_best.pth.tar'
		    	
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': str(model),
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filenameCheckpoint, filenameBest)
            
            if (encoderOnly):
                filename = f'{saveDir}/model_encoder-{epoch:03}.pth'
                filenamebest = f'{saveDir}/model_best_encoder.pth'
            else:
                filename = f'{saveDir}/model-{epoch:03}.pth'
                filenamebest = f'{saveDir}/model_best.pth'
		    
		    # save model after some epochs
            if args.epochs_save > 0 and iter > 0 and iter % args.epochs_save == 0:
                torch.save(model.state_dict(), filename)
                print(f'save: {filename} (epoch: {epoch})')
		    
		    # save the best model
            if (is_best):
                torch.save(model.state_dict(), filenamebest)
                print(f'save: {filenamebest} (epoch: {epoch})')
		    	
                if (not encoderOnly):
                    with open(saveDir + "/best_IoU.txt", "w") as myfile:
                        myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))   
                else:
                    with open(saveDir + "/best_IoU_encoder.txt", "w") as myfile:
                        myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))           
		    
		    # save information in .txt files
            #SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
            #Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
            with open(automated_log_path, "a") as myfile:
                myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain, iouVal, usedLr ))
    
    return model   # return model (convenience for encoder-decoder training)


def main(args):

    #== select GPU ==#
    cuda.set_device(args.gpu)
    print('CUDA? ', cuda.is_available(), ',  Running on GPU: ', cuda.current_device())

    sys.path.append(args.mainRoute + args.modelRoute)
    model_file = importlib.import_module(args.model)

	# detetmine the classNum and saving directory
    if args.changeClassNum == 0:
        if args.dataset == 'cityscapes':
            classNum = 20
            saveDir = f'../save/{args.saveDir}'
    else:
        classNum = args.changeClassNum
        saveDir = f'../save/{args.saveDir}'

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
        print('Create "', saveDir, '" directory')		

    with open(saveDir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))		
	
	# Load Model
    model = model_file.Net(classNum)
	
    copyfile(args.mainRoute + args.modelRoute + args.model + '.py', saveDir + '/network.py')

    
    if args.cuda:
        model = model.cuda()

	# ** training the encoder	
    if (not args.oneStageTrain):
        
        print("========== ENCODER TRAINING ===========")
		
        if args.imagenetEnc:
            print("Loading encoder pretrained in imagenet")
			
            from erfnet_enc_imagenet import ERFNet as ERFNet_enc_imagenet
			
            trainedEnc = torch.nn.DataParallel(ERFNet_enc_imagenet(1000))
            trainedEnc.load_state_dict(torch.load(args.imagenetEncFile)['state_dict'])
            
			# imagenet pre-trained encoder
            trainedEnc = next(trainedEnc.children()).features.encoder
            
            if (not args.cuda):
                trainedEnc = trainedEnc.cpu()  # because loaded encoder is probably saved in cuda
			
			# Load Model // if (not args.imagenetEnc), use the model loaded by previous line (line 437)		
            model = model_file.Net(classNum, giveEncoder=trainedEnc)
			
			
            if args.cuda:
                model = model.cuda()

				
        model = train(args, model, classNum, epochNum=args.epochEncoder, encoderOnly=True)
		
    #CAREFUL: for some reason, after training encoder alone, the decoder gets weights=0. 
    #We must reinit decoder weights or reload network passing only encoder in order to train decoder
    
    print("========== DECODER TRAINING ===========")
	
    if (not args.state):
	
        if (not args.oneStageTrain):
			# self-trained encoder
            trainedEnc = next(model.children())#.encoder
			
			# Load Model
            model = model_file.Net(classNum, giveEncoder=trainedEnc)  # Add decoder to encoder
            print('self-trained encoder')
        
        elif args.imagenetEnc:
            print("Loading encoder pretrained in imagenet")
			
            from erfnet_enc_imagenet import ERFNet as ERFNet_enc_imagenet
			
            trainedEnc = torch.nn.DataParallel(ERFNet_enc_imagenet(1000))
            trainedEnc.load_state_dict(torch.load(args.imagenetEncFile)['state_dict'])
            
			# imagenet pre-trained encoder
            trainedEnc = next(trainedEnc.children()).features.encoder
            
            if (not args.cuda):
                trainedEnc = trainedEnc.cpu()  # because loaded encoder is probably saved in cuda
			
			# Load Model		
            model = model_file.Net(classNum, giveEncoder=trainedEnc)  # Add decoder to encode
            print('only imagenet pre-trained encoder')
			
        else:
			# Load Model	
            model = model_file.Net(classNum)    
            print('non-trained encoder')
			
        if args.cuda:
            model = model.cuda()
    
	# ** training the whole network with pre-trained encoder
    model = train(args, model, classNum, epochNum=args.epochDecoder, encoderOnly=False)   #Train decoder
    
    print("========== TRAINING FINISHED ===========")


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')  # default=None

    parser.add_argument('--mainRoute', default="************************")  # change the path to your main file
    parser.add_argument('--model', default="DFGNet.py")
    parser.add_argument('--modelRoute', default="models/")
    parser.add_argument('--saveDir', default="test")  # equivalent to --loadDir of eval scripts
    parser.add_argument('--port', type=int, default=8097)	
    parser.add_argument('--dataset', default="cityscapes")  # cityscapes
    parser.add_argument('--changeClassNum', type=int, default=0)  # 0 means do not change classNum
    parser.add_argument('--height', type=int, default=512)  # if do not resize, set height=0
    parser.add_argument('--epochEncoder', type=int, default=150)
    parser.add_argument('--epochDecoder', type=int, default=150)
    parser.add_argument('--batchSize', type=int, default=5)  
    parser.add_argument('--optimizer', default="adam")  # adam or sgd
    parser.add_argument('--lr', type=float, default=5e-4)	
    parser.add_argument('--iter-loss', type=int, default=500)
    parser.add_argument('--epochs-save', type=int, default=0)  # 0 means OFF, you can use this value to save model every X epochs
    parser.add_argument('--onlyWholeNet', action='store_true') # e.g., SegNet, OurSegNet. it have to be turned on together with --oneStageTrain
    parser.add_argument('--oneStageTrain', action='store_true')
    parser.add_argument('--imagenetEnc', action='store_true')
    parser.add_argument('--imagenetEncFile', default="erfnet_enc_imagenet.pth.tar")
    parser.add_argument('--doEvalTrain', action='store_true')  # default=False, recommended: False (takes more time to train otherwise)
    parser.add_argument('--doEvalVal', action='store_true', default=True)  
    parser.add_argument('--resume', action='store_true')  # Use this flag to load last checkpoint for training
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)  # specify CPU ID you would use :0 or 1 or 2 or 3	
    parser.add_argument('--cuda', action='store_true', default=True)
	# NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
    
    main(parser.parse_args())

	