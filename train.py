import torch.optim as optim
from net.network import WITT
from net.channel import Channel
from net.decoder import *
from net.encoder import *
from data.datasets import get_loader
from utils import *
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from datetime import datetime
import torch.nn as nn
import argparse
from loss.distortion import *
import time
import math
from random import choice

import pandas as pd
import numpy as np
 
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
import matplotlib.ticker as mtick

from proplot import rc
import csv

import copy
# 统一设置字体
rc["font.family"] = "TeX Gyre Schola"
rc['tick.labelsize'] = 10
rc["axes.labelsize"] = 20
rc["axes.labelweight"] = "light"
rc["tick.labelweight"] = "bold"


from multiprocessing import Manager,Process
from threading import Thread, current_thread


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
    device = torch.device("cuda",0)
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
        save_model_freq = 1 
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
    CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
else:
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()


def weight_init(m):
    classname = m.__class__.__name__ 
    if classname.find('Conv') != -1:  
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def load_weights(model_path, i):
    pretrained = torch.load(model_path)
    pre_net[i].load_state_dict(pretrained, strict=True)
    del pretrained

def train_one_epoch_encoder(args, a):
    net.train()
    
    global global_step
    global my_step
    global epoch
    elapsed, losses, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(6)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs]

    if args.trainset == 'CIFAR10': 
        js = [(a+5)%10, (a+6)%10]
        for batch_idx, (input1, label) in enumerate(train_loader): 
            dfx = []
            psnrx = {}
            input_len = {}
            input_sort = []
            my_step += 1
            for j in js:                  
                start_time = time.time()
                global_step += 1

                input_list = []
                for idx, input_i in enumerate(input1):
                    if label[idx][2] == str(j): 
                        input_list.append(input_i) 
                        input_sort.append(input_i) 
                input_len[j] = len(input_list) 
                input = torch.stack(input_list, 0).cuda()	

                SNR = choice([1,4,7,10,13])
                
                feature = my_encoder(input, SNR, "WITT")
                fx = feature.clone().detach().requires_grad_(True)
                fx.retain_grad()
                noisy_feature = channel.forward(fx, SNR) 
                recon_image = decoders[j](noisy_feature, SNR, "WITT")
                squared_difference = torch.nn.MSELoss(reduction='mean')
                mse = squared_difference(input * 255., recon_image.clamp(0., 1.) * 255.)
                loss = mse
                CBR = feature.numel() / 2 / input.numel()
                #SNR = 10   

                optimizer.zero_grad()
                loss.backward()
                dfx.append(fx.grad.clone().detach())

                elapsed.update(time.time() - start_time)
                losses.update(loss.item())
                cbrs.update(CBR)
                snrs.update(SNR)
                if mse.item() > 0:
                    psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                    psnrs.update(psnr.item())
                    psnrx[j] = psnr.item() 
                    msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                    msssims.update(msssim)
                else:
                    psnrs.update(100)
                    msssims.update(100)

                if (global_step % config.print_step) == 0:
                    process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
                    log = (' | '.join([
                        f'encoder {a}',
                        f'decoder {j}',
                        f'Epoch {epoch}',
                        f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                        f'Time {elapsed.val:.3f}',
                        f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Lr {0.0001}',
                    ]))
                    logger.info(log)
                    for i in metrics:
                        i.clear()

            optimizer.zero_grad()
            input_sort = torch.stack(input_sort, 0).cuda()
            my_fx = my_encoder(input_sort, 10, "WITT_W/O")
            dd = torch.cat(dfx,dim=0)
            my_fx.backward(dd)
            optimizer.step()

            psnr = 0
            for j in js:
                psnr += psnrx[j] * input_len[j] 
            psnr /= 128          

            loss_x.append(my_step) 
            psnr_y.append(psnr)

            plt.figure(figsize=(14,7),dpi=100)
            plt.plot(loss_x,psnr_y,color='red',linewidth='1')
            plt.xlabel("iter",fontsize='14')
            plt.ylabel("psnr",fontsize='14')
            plt.tick_params(labelsize=14)
            plt.title(str(a) + "_encoder,train psnr",fontsize='20') 
            plt.savefig("./train_psnr/" + str(a) + "_encoder.png", dpi=100)
            plt.close()

    for i in metrics:
        i.clear()

def train_one_epoch_decoder(args, a):
    net.train()
    global global_step
    global my_step
    global epoch
    elapsed, losses, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(6)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs]

    if args.trainset == 'CIFAR10': 
        js = [(a+10-5)%10, (a+10-6)%10]
        for batch_idx, (input1, label) in enumerate(train_loader): 
            dfx = []
            psnrx = {}
            input_len = {}
            input_sort = []
            my_step += 1
            for j in js:                 
                start_time = time.time()
                global_step += 1

                input_list = []
                for idx, input_i in enumerate(input1):
                    if label[idx][0] == str(j): 
                        input_list.append(input_i) 
                        input_sort.append(input_i) 
                input_len[j] = len(input_list) 
                input = torch.stack(input_list, 0).cuda()	
                
                SNR = choice([1,4,7,10,13])
                
                feature = encoders[j](input, SNR, "WITT")
                noisy_feature = channel.forward(feature, SNR) 
                recon_image = my_decoder(noisy_feature, SNR, "WITT")
                squared_difference = torch.nn.MSELoss(reduction='mean')
                mse = squared_difference(input * 255., recon_image.clamp(0., 1.) * 255.)
                loss = mse
                CBR = feature.numel() / 2 / input.numel()
                #SNR = 10   

                loss.backward() 

                elapsed.update(time.time() - start_time)
                losses.update(loss.item())
                cbrs.update(CBR)
                snrs.update(SNR)
                if mse.item() > 0:
                    psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                    psnrs.update(psnr.item())
                    psnrx[j] = psnr.item() 
                    msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                    msssims.update(msssim)
                else:
                    psnrs.update(100)
                    msssims.update(100)

                if (global_step % config.print_step) == 0:
                    process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
                    log = (' | '.join([
                        f'encoder {j}',
                        f'decoder {a}',
                        f'Epoch {epoch}',
                        f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                        f'Time {elapsed.val:.3f}',
                        f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Lr {0.0001}',
                    ]))
                    logger.info(log)
                    for i in metrics:
                        i.clear()

            optimizer.step()
            optimizer.zero_grad()

            psnr = 0
            for j in js:
                psnr += psnrx[j] * input_len[j] 
            psnr /= 128          


            loss_x.append(my_step) 
            psnr_y.append(psnr)

            plt.figure(figsize=(14,7),dpi=100)
            plt.plot(loss_x,psnr_y,color='red',linewidth='1')
            plt.xlabel("iter",fontsize='14')
            plt.ylabel("psnr",fontsize='14')
            plt.tick_params(labelsize=14)
            plt.title(str(a) + "_decoder,train psnr",fontsize='20') 
            plt.savefig("./train_psnr/" + str(a) + "_decoder.png", dpi=100)
            plt.close()
    
    for i in metrics:
        i.clear()

def test(a, b, f):
    global psnr_t
    config.isTrain = False
    net.eval()
    elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs]
    multiple_snr = [10]
    
    results_snr = np.zeros(len(multiple_snr))
    results_cbr = np.zeros(len(multiple_snr))
    results_psnr = np.zeros(len(multiple_snr))
    results_msssim = np.zeros(len(multiple_snr))
    for i, SNR in enumerate(multiple_snr):
        with torch.no_grad():
            if args.trainset == 'CIFAR10':
                for batch_idx, input in enumerate(test_loader[j]):
                    start_time = time.time()
                    input = input.cuda()
                    recon_image, CBR, SNR, mse, loss_G = net(input, SNR)
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
                        f'encoder {a}',
                        f'decoder {b}',
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        #f'Lr {cur_lr}',
                    ]))
                    logger.info(log)
            
        results_snr[i] = snrs.avg
        results_cbr[i] = cbrs.avg
        results_psnr[i] = psnrs.avg
        results_msssim[i] = msssims.avg


        for t in metrics:
            t.clear()

    if f == -1:
        psnr_t.append(results_psnr[0])
    else:
        my_psnr[b].append(results_psnr[0])

        
    print("SNR: {}" .format(results_snr.tolist()))
    print("CBR: {}".format(results_cbr.tolist()))
    print("PSNR: {}" .format(results_psnr.tolist()))
    print("MS-SSIM: {}".format(results_msssim.tolist()))
    print("Finish Test!")


if __name__ == '__main__':       
    seed_torch()
    logger = logger_configuration(config, save_log=True)
    logger.info(config.__dict__)
    torch.manual_seed(seed=config.seed)
    


    if args.training:     
        model_path = "/root/autodl-fs/10_clients/fedavg10/model/EP90.model"
        pre_net = {}
        for i in range(0,10):
            pre_net[i] = WITT(args, config, 0, -1, -1) 
            load_weights(model_path, i)
            save_model(pre_net[i].encoder, './model/net'+str(i)+'_encoder.model')
            save_model(pre_net[i].decoder, './model/net'+str(i)+'_decoder.model')
            
        
        x_test = []
        psnr_test = []
        
        my_train_dir = "./media/train/"      
        my_test_dir = "./media/test/"
        
        
        for whole_epoch in range(0,1000):
            turn = 0
            
            #训练encoder
            for i in range(0, 10):  
                ###########################接收方############################
                js = [(i+5)%10, (i+6)%10]
                #############################################################
                my_step = 0            
                my_psnr = {} 
                loss_x = []
                psnr_y = []
                xx = []
                cur_lr = 0.0001 

                test_loader = {}
                state_dict = {}
                dir_list = []
                decoders = {}

                net =  WITT(args, config, 1, i, i).cuda()
                my_encoder = copy.deepcopy(net.encoder).cuda()
                optimizer = optim.Adam(my_encoder.parameters(), lr=0.0001)

                for j in js:
                    dir_list.append(my_train_dir + str(i)+"_"+str(j)+"/")

                for j in js:
                    config.train_data_dir = dir_list
                    config.test_data_dir = [my_test_dir + str(i)+"_"+str(j)+"/"]
                    train_loader, test_loader[j] = get_loader(args, config) 
                    my_psnr[j] = []

                    state_dict[j] = torch.load('./model/net'+str(j)+'_decoder.model')
                    decoders[j] = copy.deepcopy(net.decoder).cuda()
                    decoders[j].load_state_dict(state_dict[j]) 
                    decoders[j].train()
                

                channel = Channel(args, config)
                for epoch in range(0 ,25):                 
                    global_step = 0             
                    train_one_epoch_encoder(args, i) 
                    net.encoder.load_state_dict(my_encoder.state_dict())
#                     if (epoch+1) % 10 == 0:
#                         save_model(net.encoder, save_path='./model/net'+str(i)+'_encoder.model') 


                    if epoch % 1 == 0:
                        xx.append(epoch)
                        for j in js:         
                            net.decoder.load_state_dict(state_dict[j])
                            test(i, j, 0)
                            plt.figure(figsize=(7,7),dpi=100)

                            plt.plot(xx ,my_psnr[j],color='red',linewidth='3')
#                             for aa, bb in zip(xx, my_psnr[j]):
#                                  plt.text(aa, round(bb,2), (aa,round(bb,2)),ha='center', va='bottom', fontsize=10)
                            plt.xlabel("epoch",fontsize='14')
                            plt.ylabel("PSNR(dB)",fontsize='14')
                            plt.tick_params(labelsize=14)
                            plt.title("CIFAR3 dataset,test psnr"+ str(i) + " send to " + str(j),fontsize='20')
                            plt.savefig('./test_psnr/'+ str(i) + '_en/' + str(i) + "_" + str(j) + "_" + str(whole_epoch) +'.png', dpi=100)
                            plt.close()
                            
                save_model(net.encoder, save_path='./model/net'+str(i)+'_encoder.model') 
            turn = turn + 1


            #训练decoder
            for i in range(0, 10):  
                ###########################发送方############################
                js = [(i+10-5)%10, (i+10-6)%10]
                #############################################################
                my_step = 0            
                my_psnr = {} #记录(i,j)训练过程中的变化
                loss_x = []
                psnr_y = []
                xx = []
                cur_lr = 0.0001 

                test_loader = {}
                state_dict = {}
                dir_list = []
                encoders = {}

                net =  WITT(args, config, 1, i, i).cuda()
                my_decoder = copy.deepcopy(net.decoder).cuda()
                optimizer = optim.Adam(my_decoder.parameters(), lr=0.0001)

                for j in js:
                    dir_list.append(my_train_dir + str(j)+"_"+str(i)+"/")

                for j in js:
                    config.train_data_dir = dir_list
                    config.test_data_dir = [my_test_dir + str(j)+"_"+str(i)+"/"]
                    train_loader, test_loader[j] = get_loader(args, config) 
                    my_psnr[j] = []

                    state_dict[j] = torch.load('./model/net'+str(j)+'_encoder.model')
                    encoders[j] = copy.deepcopy(net.encoder).cuda()
                    encoders[j].load_state_dict(state_dict[j]) 
                    encoders[j].train()
                

                channel = Channel(args, config)
                for epoch in range(0 ,25):          
                    global_step = 0             
                    train_one_epoch_decoder(args, i) 
                    net.decoder.load_state_dict(my_decoder.state_dict())
                     if (epoch+1) % 10 == 0:
                         save_model(net.decoder, save_path='./model/net'+str(i)+'_decoder.model') 

                    if epoch % 1 == 0:
                        xx.append(epoch)
                        for j in js:        
                            net.encoder.load_state_dict(state_dict[j])
                            test(i, j, 0)

                            plt.figure(figsize=(7,7),dpi=100)

                            plt.plot(xx ,my_psnr[j],color='red',linewidth='3')
                             for aa, bb in zip(xx, my_psnr[j]):
                                  plt.text(aa, round(bb,2), (aa,round(bb,2)),ha='center', va='bottom', fontsize=10)
                            plt.xlabel("epoch",fontsize='14')
                            plt.ylabel("PSNR(dB)",fontsize='14')
                            plt.tick_params(labelsize=14)
                            plt.title("CIFAR3 dataset,test psnr"+ str(j) + " send to " + str(i),fontsize='20')
                            plt.savefig('./test_psnr/'+ str(i) + '_de/' + str(j) + "_" + str(i) + "_" + str(whole_epoch) +'.png', dpi=100)
                            plt.close()
                            
                save_model(net.decoder, save_path='./model/net'+str(i)+'_decoder.model') 
            turn = turn + 1
            
            #一个全局round结束，测试平均psnr
            psnr_t = []
            for i in range(0, 10):
                js = [(i+5)%10, (i+6)%10]
                for j in js:
                    net =  WITT(args, config, 1, i, j).cuda()
                    config.train_data_dir = [my_train_dir + str(i)+"_"+str(j)+"/"]
                    config.test_data_dir = [my_test_dir + str(i)+"_"+str(j)+"/"]
                    train_loader, test_loader[j] = get_loader(args, config) 
                    test(i, j, -1)
            psnr_array = np.array(psnr_t)
            x_test.append(whole_epoch)
            psnr_test.append(np.mean(psnr_array))   
            
            my_list = [[whole_epoch + 1,np.mean(psnr_array)]]
            with open("psnr.csv", "a", newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerows(my_list)
            
            plt.figure(figsize=(7,7),dpi=100)
            for aa, bb in zip(x_test, psnr_test):
                plt.text(aa, round(bb,2), (aa,round(bb,2)),ha='center', va='bottom', fontsize=10)
            plt.plot(x_test,psnr_test,color='red',linewidth='3')
            plt.xlabel("global_epoch",fontsize='14')
            plt.ylabel("PSNR_avg",fontsize='14')
            plt.tick_params(labelsize=14)
            plt.title("minibatch with fedavg",fontsize='20')
            plt.savefig('./test.png', dpi=100) 
            plt.close()
                    
        

        
        
        

        

