## train.py
# conda activate torch222
# python train.py -N 32 -e 200 --lr 0.0005 --rnn_model ConvGRUCell
# tensorboard --logdir=runs

#encoding: utf-8
import time, os, argparse, sys, random, datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from pytorch_msssim import ssim, ms_ssim
from util.metric import MultiScaleSSIM, psnr

import torch
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import transforms

parser = argparse.ArgumentParser()

parser.add_argument('--test', '-vf', type=str, default='data/tiny-imagenet-200/test_set', help='folder of test images')
parser.add_argument('--iter', '-i', type=int, default=16, help='unroll iterations')
parser.add_argument('--model_path', type=str, default='checkpoint/tiny-imagenet-200-ConvGRUCell/batch32-lr0.0005-l1-06_17_18_02/_best_model_epoch_0188.pth', help='path to model')
parser.add_argument('--metric_method', type=str, default='MS_SSIM', choices=['MS_SSIM'], help='loss method')
parser.add_argument('--reconstruction_metohod', type=str, default='oneshot', choices=['one_shot', 'additive_reconstruction'],help='reconstruction method')
parser.add_argument('--rnn_model', type=str, default='ConvGRUCell', choices=['ConvGRUCell', 'ConvLSTMCell'], help='RNN model')

if __name__ == '__main__':
    args = parser.parse_args()

    last_epoch = 0
    ## load 32x32 patches from images
    import dataset

    # 32x32 random crop
    test_transform = transforms.Compose([transforms.CenterCrop((32, 32)), transforms.ToTensor()])

    # load training set
    test_set = dataset.ImageFolder(root=args.test, transform=test_transform)
    test_loader = data.DataLoader(dataset=test_set, batch_size=16, shuffle=False, num_workers=1)
    print("", "="*120, '\n ||\t[val loader] total images: {}; total batches: {}\n'.format(len(test_set), len(test_loader)), "="*120)
    ## load networks on GPU
    if args.rnn_model == 'ConvGRUCell':
        from modules import GRU_network as network
    else:
        from modules import LSTM_network as network

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(' ||\tdevice: {}\n'.format(device), "="*120, "\n")
    encoder = network.EncoderCell().to(device)
    binarizer = network.Binarizer().to(device)
    decoder = network.DecoderCell().to(device)

    # if checkpoint is provided, resume from the checkpoint
    checkpoint = torch.load(args.model_path)
    encoder.load_state_dict(checkpoint['encoder'])
    binarizer.load_state_dict(checkpoint['binarizer'])
    decoder.load_state_dict(checkpoint['decoder'])

    with torch.no_grad():
        encoder.eval(), binarizer.eval(), decoder.eval()
        for pred in test_loader:
            # print('pred:', pred.shape) # >> torch.Size([1, 3, 32, 32])
            if args.rnn_model == 'ConvGRUCell':
                # init gru state
                encoder_h_1 = Variable(torch.zeros(pred.size(0), 256, 8, 8).to(device))
                encoder_h_2 = Variable(torch.zeros(pred.size(0), 512, 4, 4).to(device))
                encoder_h_3 = Variable(torch.zeros(pred.size(0), 512, 2, 2).to(device))

                decoder_h_1 = Variable(torch.zeros(pred.size(0), 512, 2, 2).to(device))
                decoder_h_2 = Variable(torch.zeros(pred.size(0), 512, 4, 4).to(device))
                decoder_h_3 = Variable(torch.zeros(pred.size(0), 256, 8, 8).to(device))
                decoder_h_4 = Variable(torch.zeros(pred.size(0), 128, 16, 16).to(device))
            else:
                ## init lstm state
                encoder_h_1 = (Variable(torch.zeros(pred.size(0), 256, 8, 8).to(device)),
                            Variable(torch.zeros(pred.size(0), 256, 8, 8).to(device)))
                encoder_h_2 = (Variable(torch.zeros(pred.size(0), 512, 4, 4).to(device)),
                            Variable(torch.zeros(pred.size(0), 512, 4, 4).to(device)))
                encoder_h_3 = (Variable(torch.zeros(pred.size(0), 512, 2, 2).to(device)),
                            Variable(torch.zeros(pred.size(0), 512, 2, 2).to(device)))

                decoder_h_1 = (Variable(torch.zeros(pred.size(0), 512, 2, 2).to(device)),
                            Variable(torch.zeros(pred.size(0), 512, 2, 2).to(device)))
                decoder_h_2 = (Variable(torch.zeros(pred.size(0), 512, 4, 4).to(device)),
                            Variable(torch.zeros(pred.size(0), 512, 4, 4).to(device)))
                decoder_h_3 = (Variable(torch.zeros(pred.size(0), 256, 8, 8).to(device)),
                            Variable(torch.zeros(pred.size(0), 256, 8, 8).to(device)))
                decoder_h_4 = (Variable(torch.zeros(pred.size(0), 128, 16, 16).to(device)),
                            Variable(torch.zeros(pred.size(0), 128, 16, 16).to(device)))
            patches = Variable(pred.to(device))

            decoded_image = torch.zeros(1, 3, 32, 32) + 0.5
            res = patches - 0.5

            for iter_n in range(args.iter): # args.iterations = 16
                encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                    res, encoder_h_1, encoder_h_2, encoder_h_3)

                codes = binarizer(encoded)

                if args.reconstruction_metohod == 'additive_reconstruction':
                    output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4) + output
                else:
                    output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
                ## 문제 3: res = res - output || res = x_org - output
                res = res - output # output = ^x_{t-1}

                decoded_image = decoded_image + output.data.cpu()
            
            # image = image + output.data.cpu()
            batch_size = patches.size(0)
            print('batch_size:', batch_size)

            l1_loss = res.abs().mean()

            img1_for_ssim = patches.cpu()
            img2_for_ssim = decoded_image.cpu()

            X = patches.cpu().numpy()
            Y = decoded_image.numpy()
            X = (X.clip(0, 1)*255.0).astype(np.uint8)
            Y = (Y.clip(0, 1)*255.0).astype(np.uint8)
            ms_ssim_loss = MultiScaleSSIM(X, Y, max_val=255)
            ms_ssim_loss2 = ssim(torch.tensor(X), torch.tensor(Y), data_range=255, size_average=True)
            psnr_loss = psnr(X, Y)

            print('[Metric]: batch size ', batch_size, ' \n\t1. l1_loss:', l1_loss.item(), '\n\t2. MS_SSIM:', ms_ssim_loss, '\n\t3. SSIM:', ms_ssim_loss2.item(), '\n\t4. PSNR:', psnr_loss)

            image_array1 = patches[0].squeeze(0).cpu().numpy()
            image_array1 = np.squeeze(image_array1.clip(0, 1) * 255.0).astype(np.uint8).transpose(1, 2, 0)
            image_array2 = np.squeeze(decoded_image[0].numpy().clip(0, 1) * 255.0).astype(np.uint8).transpose(1, 2, 0)
            
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(image_array1)
            plt.title('Org Image')
            plt.subplot(1, 2, 2)
            plt.imshow(image_array2)
            plt.title('Decoded Image')
            plt.show()