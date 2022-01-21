# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of CURL: Neural Curve Layers for Global Image Enhancement
https://arxiv.org/pdf/1911.13175.pdf

Please cite paper if you use this code.

Tested with Pytorch 1.7.1, Python 3.7.9

Authors: Sean Moran (sean.j.moran@gmail.com), 2020

Instructions:

To get this code working on your system / problem you will need to edit the
data loading functions, as follows:

1. main.py, change the paths for the data directories to point to your data
directory 

2. main.py, requires images_train.txt, images_valid.txt, images_test.txt, 
that list the training, validation and test images, one per line of each
txt file

3. data.py, lines 223, 240, 423, 431 change the folder names of the data input and
output directories to point to your folder names. 

We used the Samsung S7 and the Adobe datasets in the paper. They can be 
found at the following URLs:

1. Samsung S7: https://elischwartz.github.io/DeepISP/
2. Adobe5k: https://data.csail.mit.edu/graphics/fivek/

To train the model:

python main.py --valid_every=250 --num_epoch=10000 

With the above arguments, the model will be tested on the validation dataset
every 250 epochs, and the total number of epochs for training will be 10,000.

'''

import time
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import logging
import argparse
import torch.optim as optim
import numpy as np
import datetime
import os.path
import os
import evaluate
import model
import sys
from torchsummary import summary
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

np.set_printoptions(threshold=sys.maxsize)

def main():
    import data  # FIXME - import not picked up unless in main scope. Why?

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dirpath = "./log_" + timestamp
    os.mkdir(log_dirpath)

    handlers = [logging.FileHandler(
        log_dirpath + "/curl.log"), logging.StreamHandler()]
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=handlers)

    parser = argparse.ArgumentParser(
        description="Train the CURL neural network on image pairs")

    parser.add_argument(
        "--num_epoch", type=int, required=False, help="Number of epoches (default 5000)", default=100000)
    parser.add_argument(
        "--valid_every", type=int, required=False, help="Number of epoches after which to compute validation accuracy",
        default=10)
    parser.add_argument(
        "--checkpoint_filepath", required=False, help="Location of checkpoint file", default=None)
    parser.add_argument(
        "--inference_img_dirpath", required=False,
        help="Directory containing images to run through a saved CURL model instance", default=None)
    parser.add_argument(
        "--training_img_dirpath", required=False,
        help="Directory containing images to train a DeepLPF model instance", default="/home/sjm213/adobe5k/adobe5k/")
    parser.add_argument("--batch_size", type=int, required=False,help="Batch size", default=1)
    parser.add_argument("--mixed_precision", type=bool, required=False, help="Batch size", default=False)

    args = parser.parse_args()
    num_epoch = args.num_epoch
    valid_every = args.valid_every
    checkpoint_filepath = args.checkpoint_filepath
    inference_img_dirpath = args.inference_img_dirpath
    training_img_dirpath = args.training_img_dirpath
    batch_size = args.batch_size
    mixed_precision = args.mixed_precision

    logging.info('######### Parameters #########')
    logging.info('Number of epochs: ' + str(num_epoch))
    logging.info('Logging directory: ' + str(log_dirpath))
    logging.info('Dump validation accuracy every: ' + str(valid_every))
    logging.info('Training image directory: ' + str(training_img_dirpath))
    logging.info('##############################')
    
    if (checkpoint_filepath is not None) and (inference_img_dirpath is not None):

        '''
        inference_img_dirpath: the actual filepath should have "input" in the name an in the level above where the images 
        for inference are located, there should be a file "images_inference.txt with each image filename as one line i.e."
        
        images_inference.txt    ../
                                a1000.tif
                                a1242.tif
                                etc
        '''
        data_dict = data.get_data_dict(inference_img_dirpath)
        inference_ids = data.get_data_ids(inference_img_dirpath+"/images_inference.txt")
        inference_data_dict = data.filter_data_dict(data_dict, inference_ids)
        
        inference_dataset = data.Dataset(data_dict=inference_data_dict,
                                         normaliser=1,
                                         is_train=False)

        inference_data_loader = torch.utils.data.DataLoader(inference_dataset, batch_size=batch_size, shuffle=False,
                                                            num_workers=10)

        '''
        Performs inference on all the images in inference_img_dirpath
        '''
        logging.info(
            "Performing inference with images in directory: " + inference_img_dirpath)

        net = model.TriSpaceRegNet(polynomial_order=4, spatial=True, 
                                   minibatch_size=batch_size,
                                   mixed_precision=mixed_precision)
        checkpoint = torch.load(checkpoint_filepath, map_location='cuda')
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()

        criterion = model.CURLLoss()

        inference_evaluator = evaluate.Evaluator(
            criterion, inference_data_loader, "test", log_dirpath, mixed_precision=mixed_precision)

        inference_evaluator.evaluate(net, epoch=0, save_images=True)

    else:
        
        data_dict = data.get_data_dict(training_img_dirpath)
        
        training_ids = data.get_data_ids(training_img_dirpath+"/images_train.txt")
        valid_ids = data.get_data_ids(training_img_dirpath+"/images_valid.txt")
        
        training_data_dict = data.filter_data_dict(data_dict, training_ids)
        validation_data_dict = data.filter_data_dict(data_dict, valid_ids)
        
        training_dataset = data.Dataset(data_dict=training_data_dict, normaliser=1, is_train=True)
        validation_dataset = data.Dataset(data_dict=validation_data_dict, normaliser=1, is_train=False)

        training_data_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True,
                                                           pin_memory=True, num_workers=16)
        validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,
                                                             pin_memory=True, num_workers=16)
   
        net = torch.nn.DataParallel(model.TriSpaceRegNet(polynomial_order=4, spatial=True, 
                                                         minibatch_size=batch_size, 
                                                         mixed_precision=mixed_precision))
        net.to(device)
        model_parameters = filter(lambda p: p.requires_grad, net.parameters())
        logging.info('######### Network created #########')
        criterion = model.CURLLoss(ssim_window_size=5).to(device)

        '''
        The following objects allow for evaluation of a model on the validation split of the dataset
        '''
        validation_evaluator = evaluate.Evaluator(criterion, validation_data_loader, "valid", log_dirpath,
                                                  mixed_precision=mixed_precision)
        start_epoch=0

        if (checkpoint_filepath is not None) and (inference_img_dirpath is None):
            logging.info('######### Loading Checkpoint #########')
            checkpoint = torch.load(checkpoint_filepath, map_location='cuda')
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      net.parameters()), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for g in optimizer.param_groups:
                g['lr'] = 1e-5

            start_epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            net.to(device)
        else:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      net.parameters()), lr=5e-7, betas=(0.5, 0.999))
            
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, total_steps=num_epoch, verbose=True)
        best_valid_psnr = 0.0
        optimizer.zero_grad()
        net.train()
        scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

        for epoch in range(start_epoch, num_epoch):
            
            logging.info("######### Epoch {}: Train #########".format(epoch + 1))
            logging.info('Learning rate: {}'.format(optimizer.param_groups[0]['lr']))

            # train loss
            examples = 0.0
            running_loss = 0.0
            batches = 0.0
            
            batch_pbar = tqdm(enumerate(training_data_loader, 0), total=len(training_data_loader))

            for batch_num, data in batch_pbar:
                input_img_batch, gt_img_batch, mask_batch = data['input_img'].to(device, non_blocking=True), \
                                                            data['output_img'].to(device, non_blocking=True), \
                                                            data['mask'].to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # See: https://pytorch.org/docs/stable/notes/amp_examples.html#dataparallel-in-a-single-process
                with torch.cuda.amp.autocast(enabled=mixed_precision):
                    net_img_batch = net(input_img_batch, mask_batch)
                    net_img_batch = torch.clamp(net_img_batch, 0.0, 1.0)
                    # Calculate loss, leaving out masked pixels
                    loss = criterion(net_img_batch, gt_img_batch, mask_batch)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loss_scalar = loss.data.item()
                running_loss += loss_scalar
                examples += batch_size
                batches += 1
                
                writer.add_scalar('Loss/train', loss_scalar, examples)
                batch_pbar.set_description('Epoch {}. Train Loss: {}'.format(epoch, loss_scalar))

            logging.info('[%d] train loss: %.15f' %
                         (epoch + 1, running_loss / batches))
            writer.add_scalar('Loss/train_smooth', running_loss / batches, epoch + 1)
            scheduler.step()
            
            # Valid loss
            if (epoch + 1) % valid_every == 0:
                net.eval()
                logging.info("######### Epoch {}: Validation #########".format(epoch + 1))

                valid_loss, valid_psnr, valid_ssim = validation_evaluator.evaluate(net, epoch)
                
                logging.info(
                    "Saving checkpoint to file: " + 
                    'curl_validpsnr_{}_validloss_{}_epoch_{}_model.pt'.format(valid_psnr, valid_loss, epoch))

                best_valid_psnr = valid_psnr
                snapshot_prefix = os.path.join(
                    log_dirpath, 'curl')
                snapshot_path = snapshot_prefix + '_validpsnr_{}_validloss_{}_epoch_{}_model.pt'.format(valid_psnr,
                                                                                                        valid_loss,
                                                                                                        epoch + 1)
                '''
                torch.save(net, snapshot_path)
                '''

                torch.save({
                    'epoch': epoch+1,
                     'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                     'loss': valid_loss,
                     }, snapshot_path)

                net.train()


if __name__ == "__main__":
    main()
