import os
from datetime import datetime
# 打印时间函数
import subprocess
import utilities as ut
from pudb import set_trace
import pandas as pd
import random
import numpy as np
import torchvision
import math
import torch
import os
from pytorch_lightning import seed_everything
from multiprocessing import Pool, cpu_count
import pysam
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
seed_everything(2022)


data_dir = "../datasets/NA12878_PacBio_MtSinai/"

bam_path = data_dir + "sorted_final_merged.bam"

vcf_filename = data_dir + "insert_result_data.csv.vcf"


def p(chr):
    p_position = torch.load(data_dir + 'position/' + chr + '/positive' + '.pt')
    for i in range(len(p_position)):
        # subprocess.call("python create_process_file.py --chr " + chr + " --len " + str(len), shell = True)
        # fd = open(chr + ".txt")
        # subprocess.Popen("python create_process_file.py --chr " + chr + " --len " + str(len), shell=True)
        # p_position = torch.load(data_dir + 'position/' + chr + '/positive' + '.pt')

        save_path = data_dir + 'split_image/' + chr

        if not os.path.exists(save_path + '/pn_cigar_new_img' + str(i) + '.pt'):
            print("python par.py --chr " + chr + " --len " + str(i))
            subprocess.Popen("python par.py --chr " + chr + " --len " + str(i), shell=True)
            d = subprocess.getoutput("ps -aux | grep xwm | grep python | grep len | awk '{print $14}'").split()
            while len(d) > 16:
                time.sleep(10)
                d = subprocess.getoutput("ps -aux | grep xwm | grep python | grep len | awk '{print $14}'").split()




import argparse   #步骤一

def parse_args():
    """
    :return:进行参数的解析
    """
    description = "you should add those parameter"                   # 步骤二
    parser = argparse.ArgumentParser(description=description)        # 这些参数都有默认值，当调用parser.print_help()或者运行程序时由于参数不正确(此时python解释器其实也是调用了pring_help()方法)时，
                                                                     # 会打印这些描述信息，一般只需要传递description参数，如上。
    help = "The path of address"
    parser.add_argument('--chr',help = help)             # 步骤三，后面的help是我的描述
    args = parser.parse_args()                                       # 步骤四
    return args

if __name__ == '__main__':
    args = parse_args()
    # print(args.chr)            #直接这么获取即可。
    # print(type(args.chr))
    p(args.chr)