import utilities as ut
from pudb import set_trace
import pandas as pd
import random
import numpy as np
import torchvision
import math
import torch
from pytorch_lightning.loggers import TensorBoardLogger
import os
from net import IDENet
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from multiprocessing import Pool, cpu_count
import pysam
from itertools import repeat
from functools import partial
import time

# from cython.parallel import prange, parallel, threadid
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
torch.multiprocessing.set_sharing_strategy('file_system')

seed_everything(2022)


data_dir = "../datasets/NA12878_PacBio_MtSinai/"

bam_path = data_dir + "sorted_final_merged.bam"

vcf_filename = data_dir + "insert_result_data.csv.vcf"


def p(sum_data):
    chromosome = sum_data
    hight = 224
    # copy begin
    print("deal " + chromosome)
    p_position = torch.load(data_dir + 'position/' + chromosome + '/positive' + '.pt')

    positive_cigar_img = torch.empty(len(p_position), 4, hight, hight)
    negative_cigar_img = torch.empty(len(p_position), 4, hight, hight)


    for i in range(len(p_position)):
        # img/positive_cigar_img
        print("cigar start")
        save_path = data_dir + 'split_image/' + chromosome

        data = torch.load(save_path + '/pn_cigar_new_img' + str(i) + '.pt')

        positive_cigar_img[i] = data[0]

        negative_cigar_img[i] = data[1]

        print("cigar end")

    save_path = data_dir + 'image/' + chromosome
    torch.save(positive_cigar_img, save_path + '/positive_cigar_new_img' + '.pt')
    torch.save(negative_cigar_img, save_path + '/negative_cigar_new_img' + '.pt')



import argparse   #步骤一

def parse_args():
    """
    :return:进行参数的解析
    """
    description = "you should add those parameter"                   # 步骤二
    parser = argparse.ArgumentParser(description=description)        # 这些参数都有默认值，当调用parser.print_help()或者运行程序时由于参数不正确(此时python解释器其实也是调用了pring_help()方法)时，
                                                                     # 会打印这些描述信息，一般只需要传递description参数，如上。
    help = "The path of address"
    parser.add_argument('--chr',help = help)                   # 步骤三，后面的help是我的描述
    parser.add_argument('--len',help = help)                   # 步骤三，后面的help是我的描述
    args = parser.parse_args()                                       # 步骤四
    return args

if __name__ == '__main__':
    args = parse_args()
    # print(args.chr)            #直接这么获取即可。
    # print(type(args.chr))
    p(args.chr)