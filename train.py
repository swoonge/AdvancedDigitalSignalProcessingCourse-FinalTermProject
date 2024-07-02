## train.py
# conda activate torch222
# python train.py -N 32 -e 200 --lr 0.0005 --rnn_model ConvGRUCell
# tensorboard --logdir=runs

#encoding: utf-8
import time, os, argparse, sys, random, datetime
from pathlib import Path
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import transforms
from pytorch_msssim import ssim
from modules.loss import gaussian_kernel, apply_gaussian_weights

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', '-N', type=int, default=32, help='batch size')
parser.add_argument('--train', '-f', type=str, default='data/tiny-imagenet-200/train_set', help='folder of training images')
parser.add_argument('--val', '-vf', type=str, default='data/tiny-imagenet-200/val_set', help='folder of validation images')
parser.add_argument('--dataset', type=str, default='tiny-imagenet-200', help='dataset')
parser.add_argument('--max-epochs', '-e', type=int, default=200, help='max epochs')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.84, help='weight of l1 loss')
parser.add_argument('--l1_gaussian',type=bool, default=True, help='use gaussian kernel for l1 loss')
parser.add_argument('--random_seed', type=int, default=0, help='random seed')
parser.add_argument('--iterations', type=int, default=16, help='unroll iterations')
parser.add_argument('--model_path', '-m', type=str, default='', help='path to model)')
parser.add_argument('--loss_method', type=str, default='mix_iter', help='loss method')
# parser.add_argument('--reconstruction_metohod', type=str, default='oneshot', choices=['one_shot', 'additive_reconstruction'],help='reconstruction method')
parser.add_argument('--rnn_model', type=str, default='ConvGRUCell', choices=['ConvGRUCell', 'ConvLSTMCell'], help='RNN model')

if __name__ == '__main__':
    args = parser.parse_args()

    # set random seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed) # if use multi-GPU
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # set up logger
    today = datetime.datetime.now().strftime("%m_%d_%H_%M")
    model_name = 'batch{}-lr{}-{}-{}' .format(args.batch_size, args.lr, args.loss_method, today)
    log_path = Path('./logs/{}-{}/{}'.format(args.dataset, args.rnn_model, model_name))
    model_out_path = Path('./checkpoint/{}-{}/{}'.format(args.dataset, args.rnn_model, model_name))
    
    print("\n", "="*120, "\n ||\tTrain network with | bath size: ", args.batch_size, " | lr: ", args.lr, " | loss method: ", args.loss_method, " | random seed: ", args.random_seed, " | iterations: ", args.iterations, "\n ||\tmodel_out_path: ", model_out_path, "\n ||\tlog_path: ",log_path, "\n", "="*120, )

    def resume():
        checkpoint = torch.load(args.model_path)
        encoder.load_state_dict(checkpoint['encoder'])
        binarizer.load_state_dict(checkpoint['binarizer'])
        decoder.load_state_dict(checkpoint['decoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        best_loss = checkpoint['loss']
        return
    
    def save(epoch, best=True):
        model_out_path.mkdir(exist_ok=True, parents=True)
        s = 'best_' if best else ''
        checkpoint = {
            'encoder': encoder.state_dict(),
            'binarizer': binarizer.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'loss': best_loss
        }
        torch.save(checkpoint, '{}/{}model_epoch_{:04d}.pth'.format(model_out_path, s, epoch))
        return

    ## load 32x32 patches from images
    import dataset

    # 32x32 random crop
    train_transform = transforms.Compose([transforms.RandomCrop((32, 32)), transforms.ToTensor()])
    val_transform = transforms.Compose([transforms.CenterCrop((32, 32)), transforms.ToTensor()])

    # load training set
    train_set = dataset.ImageFolder(root=args.train, transform=train_transform)
    train_loader = data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    print(' ||\t[train loader] total images: {}; total batches: {}\n'.format(len(train_set), len(train_loader)), "="*120)
    val_set = dataset.ImageFolder(root=args.val, transform=val_transform)
    val_loader = data.DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False, num_workers=1)
    print(' ||\t[val loader] total images: {}; total batches: {}\n'.format(len(val_set), len(val_loader)), "="*120)

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

    # set up optimizer and scheduler
    optimizer = optim.Adam([ {'params': encoder.parameters()}, {'params': binarizer.parameters()}, {'params': decoder.parameters()} ], lr=args.lr)
    lr_scheduler = LS.MultiStepLR(optimizer, milestones=[5, 10, 30, 60, 100, 200, 300, 500, 700], gamma=0.5)

    # if checkpoint is provided, resume from the checkpoint
    last_epoch = 0
    if args.model_path:
        resume()
        last_epoch = lr_scheduler.last_epoch

    ## training
    best_loss = 1e9
    best_eval_loss = 1e9
    ssim_epoch_loss = 0
    l1_epoch_loss = 0
    mix_epoch_loss = 0

    for epoch in range(last_epoch + 1, args.max_epochs + 1):
        encoder.train(), binarizer.train(), decoder.train()

        train_loader = tqdm(train_loader)
        for batch, data in enumerate(train_loader):
            if args.rnn_model == 'ConvGRUCell':
                encoder_h_1 = torch.zeros(data.size(0), 256, 8, 8).to(device)
                encoder_h_2 = torch.zeros(data.size(0), 512, 4, 4).to(device)
                encoder_h_3 = torch.zeros(data.size(0), 512, 2, 2).to(device)

                decoder_h_1 = torch.zeros(data.size(0), 512, 2, 2).to(device)
                decoder_h_2 = torch.zeros(data.size(0), 512, 4, 4).to(device)
                decoder_h_3 = torch.zeros(data.size(0), 256, 8, 8).to(device)
                decoder_h_4 = torch.zeros(data.size(0), 128, 16, 16).to(device)
            else:
                encoder_h_1 = (torch.zeros(data.size(0), 256, 8, 8).to(device),
                            torch.zeros(data.size(0), 256, 8, 8).to(device))
                encoder_h_2 = (torch.zeros(data.size(0), 512, 4, 4).to(device),
                            torch.zeros(data.size(0), 512, 4, 4).to(device))
                encoder_h_3 = (torch.zeros(data.size(0), 512, 2, 2).to(device),
                            torch.zeros(data.size(0), 512, 2, 2).to(device))

                decoder_h_1 = (torch.zeros(data.size(0), 512, 2, 2).to(device),
                            torch.zeros(data.size(0), 512, 2, 2).to(device))
                decoder_h_2 = (torch.zeros(data.size(0), 512, 4, 4).to(device),
                            torch.zeros(data.size(0), 512, 4, 4).to(device))
                decoder_h_3 = (torch.zeros(data.size(0), 256, 8, 8).to(device),
                            torch.zeros(data.size(0), 256, 8, 8).to(device))
                decoder_h_4 = (torch.zeros(data.size(0), 128, 16, 16).to(device),
                            torch.zeros(data.size(0), 128, 16, 16).to(device))
            optimizer.zero_grad()

            ssim_losses = []
            l1_losses = []
            mix_losses = []
            
            patches = data.to(device)
            res = patches - 0.5
            x_org = patches

            output = torch.zeros_like(patches) # ^x_{t-1} = 0
            decoded_images = torch.zeros_like(patches) + 0.5

            for iter_n in range(args.iterations): # args.iterations = 16
                encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(res, encoder_h_1, encoder_h_2, encoder_h_3)
                codes = binarizer(encoded)

                output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
                
                if args.l1_gaussian:
                    kernel = gaussian_kernel(5, 0, 1).to(device)
                    weighted_true = apply_gaussian_weights(res, kernel)
                    weighted_pred = apply_gaussian_weights(output, kernel)
                    loss = weighted_true - weighted_pred
                    l1_losses.append(loss.abs().mean())
                    res = res - output
                else:
                    res = res - output
                    l1_losses.append(res.abs().mean())
                
                decoded_images = decoded_images + output
                ssim_losses.append(1 - ssim(decoded_images, x_org, data_range=1.0, size_average=True))
                
            l1_loss = sum(l1_losses) / args.iterations
            ssim_loss = sum(ssim_losses) / args.iterations
            l1_epoch_loss += l1_loss.item()
            ssim_epoch_loss += ssim_loss.item()
            mix_loss = (1 - args.gamma) * l1_loss + args.gamma * ssim_loss
            mix_epoch_loss += mix_loss.item()

            if args.loss_method == 'ssim':
                ssim_loss.backward()
            elif args.loss_method == 'l1':
                l1_loss.backward()
            elif args.loss_method == 'mix_iter':
                mix_loss.backward()

            optimizer.step()
            del patches, res, l1_losses, ssim_loss, output

        lr_scheduler.step()
        ssim_epoch_loss /= len(train_loader)
        l1_epoch_loss /= len(train_loader)
        mix_epoch_loss /= len(train_loader)
        print('[TRAIN] Epoch[{}] lr: {:6f} | l1 Loss: {:.4f} | ssim Loss {:4f} | mix Loss: {:4f}'.format(epoch, lr_scheduler.get_last_lr()[0], l1_epoch_loss, ssim_epoch_loss, mix_epoch_loss))
        
        # validation
        val_epoch_loss_l1 = 0
        val_epoch_loss_ssim = 0
        val_epoch_loss_mix = 0
        val_total_t0 = time.time()
        with torch.no_grad():
            if epoch >= 0 and epoch % 1 == 0:
                for i, pred in enumerate(val_loader):
                    ### eval ###
                    encoder.eval(), binarizer.eval(), decoder.eval()
                    if args.rnn_model == 'ConvGRUCell':
                        encoder_h_1 = torch.zeros(data.size(0), 256, 8, 8).to(device)
                        encoder_h_2 = torch.zeros(data.size(0), 512, 4, 4).to(device)
                        encoder_h_3 = torch.zeros(data.size(0), 512, 2, 2).to(device)

                        decoder_h_1 = torch.zeros(data.size(0), 512, 2, 2).to(device)
                        decoder_h_2 = torch.zeros(data.size(0), 512, 4, 4).to(device)
                        decoder_h_3 = torch.zeros(data.size(0), 256, 8, 8).to(device)
                        decoder_h_4 = torch.zeros(data.size(0), 128, 16, 16).to(device)
                    else:
                        encoder_h_1 = (torch.zeros(data.size(0), 256, 8, 8).to(device),
                                    torch.zeros(data.size(0), 256, 8, 8).to(device))
                        encoder_h_2 = (torch.zeros(data.size(0), 512, 4, 4).to(device),
                                    torch.zeros(data.size(0), 512, 4, 4).to(device))
                        encoder_h_3 = (torch.zeros(data.size(0), 512, 2, 2).to(device),
                                    torch.zeros(data.size(0), 512, 2, 2).to(device))

                        decoder_h_1 = (torch.zeros(data.size(0), 512, 2, 2).to(device),
                                    torch.zeros(data.size(0), 512, 2, 2).to(device))
                        decoder_h_2 = (torch.zeros(data.size(0), 512, 4, 4).to(device),
                                    torch.zeros(data.size(0), 512, 4, 4).to(device))
                        decoder_h_3 = (torch.zeros(data.size(0), 256, 8, 8).to(device),
                                    torch.zeros(data.size(0), 256, 8, 8).to(device))
                        decoder_h_4 = (torch.zeros(data.size(0), 128, 16, 16).to(device),
                                    torch.zeros(data.size(0), 128, 16, 16).to(device))
                        
                    ssim_losses = []
                    l1_losses = []
                    mix_losses = []
                    
                    patches = data.to(device)
                    res = patches - 0.5  # r_0 = x
                    x_org = patches

                    output = torch.zeros_like(patches) # ^x_{t-1} = 0
                    decoded_images = torch.zeros_like(patches) + 0.5
                    for iter_n in range(args.iterations): # args.iterations = 16
                        encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(res, encoder_h_1, encoder_h_2, encoder_h_3)
                        codes = binarizer(encoded)

                        output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
                        if args.l1_gaussian:
                            kernel = gaussian_kernel(5, 0, 1).to(device)
                            weighted_true = apply_gaussian_weights(res, kernel)
                            weighted_pred = apply_gaussian_weights(output, kernel)
                            loss = weighted_true - weighted_pred
                            l1_losses.append(loss.abs().mean())
                            res = res - output
                        else:
                            res = res - output
                            l1_losses.append(res.abs().mean())
                        decoded_images = decoded_images + output
                        ssim_losses.append(1 - ssim(decoded_images, x_org, data_range=1.0, size_average=True))
                        
                    l1_loss = sum(l1_losses) / args.iterations
                    ssim_loss = sum(ssim_losses) / args.iterations
                    mix_loss = (1 - args.gamma) * l1_loss + args.gamma * ssim_loss

                    val_epoch_loss_l1 += l1_loss.item()
                    val_epoch_loss_ssim += ssim_loss.item()
                    val_epoch_loss_mix += mix_loss.item()

                val_epoch_loss_l1 /= len(val_loader)
                val_epoch_loss_ssim /= len(val_loader)
                val_epoch_loss_mix /= len(val_loader)
                val_total_t1 = time.time()

                print('[Val] Epoch[{}] l1 Loss: {:.4f} | ssim Loss: {:.4f} | mix Loss: {:4f} | Time: {:.5f} sec'.format(epoch, val_epoch_loss_l1, val_epoch_loss_ssim, val_epoch_loss_mix, val_total_t1 - val_total_t0))

            if (val_epoch_loss_mix <= best_loss + 1e-6):
                best_loss = val_epoch_loss_mix
                save(epoch, True)
                print('[Save] Best model saved at {} epoch'.format(epoch))
            elif epoch % 10 == 0:
                save(epoch, False)
                print('[Save] model saved at {} epoch'.format(epoch))

        # ================================================================== #
        #                        Tensorboard Logging                         #
        # ================================================================== #
        # logger.add_scalar('Train/val_loss',mean_val_loss,epoch)
        if not log_path.exists():
            log_path.mkdir(exist_ok=True, parents=True)
            logger = SummaryWriter(log_path)
        logger.add_scalar('Train/epoch_loss_l1', l1_epoch_loss, epoch)
        logger.add_scalar('Train/epoch_loss_ssim', ssim_epoch_loss, epoch)
        logger.add_scalar('Train/epoch_loss_mix', mix_epoch_loss, epoch)
        logger.add_scalar('Train/rl', lr_scheduler.get_last_lr()[0], epoch)
        logger.add_scalar('Val/val_loss_l1', val_epoch_loss_l1, epoch)
        logger.add_scalar('Val/val_loss_ssim', val_epoch_loss_ssim, epoch)
        logger.add_scalar('Val/val_loss_mix', val_epoch_loss_mix, epoch)
        print("log file saved to {}\n".format(log_path))
        
         