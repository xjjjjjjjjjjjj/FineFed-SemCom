import torch.optim as optim
from net.network import WITT
from data.datasets import get_loader
from utils import *
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from datetime import datetime
import torch.nn as nn
import argparse
from loss.distortion import *
import time
import math

import pandas as pd
import numpy as np
 
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
import matplotlib.ticker as mtick

from proplot import rc
# 统一设置字体
rc["font.family"] = "TeX Gyre Schola"
rc['tick.labelsize'] = 10
rc["axes.labelsize"] = 20
rc["axes.labelweight"] = "light"
rc["tick.labelweight"] = "bold"


parser = argparse.ArgumentParser(description='WITT')
parser.add_argument('--training', action='store_true',
                    help='training or testing')
parser.add_argument('--trainset', type=str, default='CIFAR10',
                    choices=['CIFAR10', 'DIV2K'],
                    help='train dataset name')
parser.add_argument('--testset', type=str, default='kodak',
                    choices=['CIFAR10','kodak', 'CLIC21'],
                    help='specify the testset for HR models')
parser.add_argument('--distortion-metric', type=str, default='MSE',
                    choices=['MSE', 'MS-SSIM'],
                    help='evaluation metrics')
parser.add_argument('--model', type=str, default='WITT',
                    choices=['WITT', 'WITT_W/O'],
                    help='WITT model or WITT without channel ModNet')
parser.add_argument('--channel-type', type=str, default='awgn',
                    choices=['awgn', 'rayleigh'],
                    help='wireless channel model, awgn or rayleigh')
parser.add_argument('--C', type=int, default=96,
                    help='bottleneck dimension')
parser.add_argument('--multiple-snr', type=str, default='1,4,7,10,13',
                    help='random or fixed snr')
args = parser.parse_args()

class config():
    seed = 1024
    pass_channel = True
    CUDA = True
    device = torch.device("cpu",0)
    norm = False
    # logger
    print_step = 1 #改了
    plot_step = 10000
    filename = datetime.now().__str__()[:-7]
    filename = filename.replace(':','_')
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    # training details
    normalize = False
    learning_rate = 0.0001
    tot_epoch = 10000000

    if args.trainset == 'CIFAR10':
        save_model_freq = 1 #原本是5
        image_dims = (3, 32, 32)
        train_data_dir = ["./media/Dataset/airplane/"]
        test_data_dir = ["./media/Dataset/airplane/"]
        batch_size = 128
        downsample = 2
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
    elif args.trainset == 'DIV2K':
        save_model_freq = 100
        image_dims = (3, 256, 256)
        train_data_dir = ["./media/Dataset/HR_Image_dataset/"]
        if args.testset == 'kodak':
            test_data_dir = ["./media/Dataset/kodak_test/"]
        elif args.testset == 'CLIC21':
            test_data_dir = ["./media/Dataset/CLIC21/"]
        batch_size = 16
        downsample = 4
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
            C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4],
            C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )


if args.trainset == 'CIFAR10':
    CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3)
else:
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3)


def weight_init(m):
    classname = m.__class__.__name__ 
    if classname.find('Conv') != -1:  
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def load_weights(model_path, i):
    pretrained = torch.load(model_path, map_location=torch.device('cpu'))
    net[i].load_state_dict(pretrained, strict=True)
    del pretrained



def train_one_epoch(args, a, b):
    net.train()
    elapsed, losses, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(6)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs]
    global global_step
    global loss_x
    global loss_y
    if args.trainset == 'CIFAR10': 
        for batch_idx, input in enumerate(train_loader):
            start_time = time.time()
            global_step += 1
            input = input
            recon_image, CBR, SNR, mse, loss_G = net(input)
            loss = loss_G

            #绘制loss曲线
            loss_x.append(global_step)
            loss_y.append(loss.item())
            if global_step % 100 == 0:
                plt.figure(figsize=(7,7),dpi=100)
                plt.plot(loss_x,loss_y,color='red',marker='s',markeredgecolor='r',markersize='8',linewidth='3')
                plt.legend(['WITT'],loc='lower right',fontsize='16')
                plt.xlabel("iter",fontsize='14')
                plt.ylabel("loss",fontsize='14')
                plt.tick_params(labelsize=14)
                plt.title("CIFAR3 dataset," + str(a) + " send to " + str(b),fontsize='20') #这个要改！！！！！！！！！！！！
                #ax = plt.gca()
                #ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f')) #保留两位小数
                #plt.show()
                plt.savefig('./2_19/loss/'+ str(a) + "_" + str(b) + "/" + str(global_step) + '.png', dpi=100) #这个要改！！！！！！！！！！！！

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            snrs.update(SNR)
            if mse.item() > 0:
                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                psnrs.update(psnr.item())
                msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                msssims.update(msssim)
            else:
                psnrs.update(100)
                msssims.update(100)

            if (global_step % config.print_step) == 0:
                process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
                log = (' | '.join([
                    f'encoder {a}',
                    f'decoder {b}',
                    f'Epoch {epoch}',
                    f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                    f'Time {elapsed.val:.3f}',
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                logger.info(log)
                for i in metrics:
                    i.clear()
    else:
        for batch_idx, input in enumerate(train_loader):
            start_time = time.time()
            global_step += 1
            input = input
            recon_image, CBR, SNR, mse, loss_G = net(input)
            loss = loss_G
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            cbrs.update(CBR)
            snrs.update(SNR)
            if mse.item() > 0:
                psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                psnrs.update(psnr.item())
                msssim = 1 - loss_G
                msssims.update(msssim)

            else:
                psnrs.update(100)
                msssims.update(100)

            if (global_step % config.print_step) == 0:
                process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
                log = (' | '.join([
                    f'Epoch {epoch}',
                    f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                    f'Time {elapsed.val:.3f}',
                    f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                    f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                    f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                    f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                    f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                    f'Lr {cur_lr}',
                ]))
                logger.info(log)
                for i in metrics:
                    i.clear()
    for i in metrics:
        i.clear()

def test(i_model, a, b):
    y = []#画图用的

    global psnr_x
    global psnr_y

    config.isTrain = False
    net.eval()
    elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs]
    #multiple_snr = args.multiple_snr.split(",")
    #for i in range(len(multiple_snr)):
    #    multiple_snr[i] = int(multiple_snr[i])
    
    multiple_snr = [1,4,7,10,13]
    
    results_snr = np.zeros(len(multiple_snr))
    results_cbr = np.zeros(len(multiple_snr))
    results_psnr = np.zeros(len(multiple_snr))
    results_msssim = np.zeros(len(multiple_snr))
    for i, SNR in enumerate(multiple_snr):
        with torch.no_grad():
            if args.trainset == 'CIFAR10':
                for batch_idx, input in enumerate(test_loader):
                    start_time = time.time()
                    input = input

                    image1 = np.transpose(input[0], (1, 2, 0))
                    image1 = np.clip(image1, 0, 1)
                    plt.imshow(image1)
                    plt.savefig('./222.png', dpi=100) 

                    recon_image, CBR, SNR, mse, loss_G = net(input, SNR)

                    image1 = np.transpose(recon_image[0], (1, 2, 0))
                    image1 = np.clip(image1, 0, 1)
                    plt.imshow(image1)
                    plt.savefig('./111.png', dpi=100) 
                    sys.exit()

                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR)
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                        msssims.update(msssim)
                    else:
                        psnrs.update(100)
                        msssims.update(100)

                    log = (' | '.join([
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Lr {cur_lr}',
                    ]))
                    logger.info(log)
            else:
                for batch_idx, input in enumerate(test_loader):
                    start_time = time.time()
                    input = input
                    recon_image, CBR, SNR, mse, loss_G = net(input, SNR)
                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR)
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                        #msssim = -10 * math.log10(1 - msssim)
                        msssims.update(msssim)
                    else:
                        psnrs.update(100)
                        msssims.update(100)

                    log = (' | '.join([
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Lr {cur_lr}',
                    ]))
                    logger.info(log)
        results_snr[i] = snrs.avg
        results_cbr[i] = cbrs.avg
        results_psnr[i] = psnrs.avg
        results_msssim[i] = msssims.avg

        #m = -10 * math.log10(1 - results_msssim[i])
        #y.append(m)
        #y.append(results_msssim[i])
        y.append(results_psnr[i])

        for t in metrics:
            t.clear()

    psnr_x.append(i_model)
    psnr_y.append(results_psnr[3])
    
    print("SNR: {}" .format(results_snr.tolist()))
    print("CBR: {}".format(results_cbr.tolist()))
    print("PSNR: {}" .format(results_psnr.tolist()))
    print("MS-SSIM: {}".format(results_msssim.tolist()))
    print("Finish Test!")
    
    plt.figure(figsize=(7,7),dpi=100)
    x = multiple_snr
    plt.plot(x,y,color='red',marker='s',markeredgecolor='r',markersize='8',linewidth='3')
    plt.legend(['WITT'],loc='lower right',fontsize='16')
    plt.xlabel("SNR(dB)",fontsize='14')
    plt.ylabel("PSNR(dB)",fontsize='14')
    plt.tick_params(labelsize=14)
    plt.title("CIFAR3 dataset,"+ str(a) + " send to " + str(b),fontsize='20')
    #plt.show()
    plt.savefig('./2_19/test/'+ str(a) + "_" + str(b) + "/" + str(i_model) + '.png', dpi=100)

if __name__ == '__main__':
    matplotlib.use("TKAgg")#默认的是使用qt，我电脑qt有问题

    seed_torch()
    logger = logger_configuration(config, save_log=True)
    logger.info(config.__dict__)
    torch.manual_seed(seed=config.seed)
    # net = WITT(args, config)

    # model_path = "./WITT_model/CIFAR3_AWGN_snr10_psnr/cloud_EP36.model"
    # net = {}
    # for i in range(0,3):
    #     net[i] = WITT(args, config, 0, -1, -1) 
    #     load_weights(model_path, i)
    #     torch.save(net[i].encoder.state_dict(), './WITT_model/3.1/temp/net'+str(i)+'_encoder.model')
    #     torch.save(net[i].decoder.state_dict(), './WITT_model/3.1/temp/net'+str(i)+'_decoder.model')

    # load_weights(model_path)
    #net.apply(weight_init)


    train_time = [[0,20,30],[30,0,20],[20,30,0]]
    
    for i in range(0, 3):
        for j in range(0, 3):
            if j == i:
                continue
            else: #对(i,j)通信的数据进行训练    
                loss_x = []
                loss_y = []
                psnr_x = []
                psnr_y = []

                net = WITT(args, config, 1, i, j) #这里构造(a,b)组成的模型
                # net = net
                model_params = [{'params': net.parameters(), 'lr': 0.0001}]

                config.train_data_dir = ["./media/Dataset/train/" + str(i)+"_"+str(j)+"/"]
                config.test_data_dir = ["./media/Dataset/test/" + str(i)+"_"+str(j)+"/"]

                train_loader, test_loader = get_loader(args, config) 
                cur_lr = config.learning_rate
                optimizer = optim.Adam(model_params, lr=cur_lr)
                global_step = 0
                steps_epoch = global_step // train_loader.__len__()
                if args.training:
                    for epoch in range(steps_epoch, train_time[i][j]):  #每一对（a,b)的数据训练35次  总体的epoch=1  可是这样固定的话，就不一定是最好的呀
                        train_one_epoch(args, i, j)
                        if (epoch + 1) % config.save_model_freq == 0:
                            save_model(net.encoder, save_path='./WITT_model/3.1/' + str(i) +'/encoder_EP{}.model'.format(epoch + 1))
                            save_model(net.decoder, save_path='./WITT_model/3.1/' + str(j) +'/decoder_EP{}.model'.format(epoch + 1))  #每个用户模型的名字要区分 #存每一轮的模型
                            test(epoch + 1, i, j)

                    torch.save(net.encoder.state_dict(), './WITT_model/3.1/temp/net'+str(i)+'_encoder.model') #存最新的模型，用来跟别人继续训练
                    torch.save(net.decoder.state_dict(), './WITT_model/3.1/temp/net'+str(j)+'_decoder.model')   
                    #画psnr测试变化的曲线
                    plt.figure(figsize=(7,7),dpi=100)
                    plt.plot(psnr_x,psnr_y,color='red',markeredgecolor='r',markersize='8',linewidth='3')
                    plt.legend(['WITT'],loc='lower right',fontsize='16')
                    plt.xlabel("epoch",fontsize='14')
                    plt.ylabel("PSNR(dB)",fontsize='14')
                    plt.tick_params(labelsize=14)
                    plt.title("CIFAR3 dataset,"+ str(i) + " send to " + str(j),fontsize='20')
                    #plt.show()
                    plt.savefig('./2_19/psnr/'+ str(i) + "_" + str(j) + '.png', dpi=100)     
                else:
                    test(0, i, j)
        
        
        

        

