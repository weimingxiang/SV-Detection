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

position_enforcement_refresh = 0
img_enforcement_refresh = 0
sign_enforcement_refresh = 0 # attention
cigar_enforcement_refresh = 0

# sam_file = pysam.AlignmentFile(bam_path, "rb")
# chr_list = sam_file.references
# chr_length = sam_file.lengths
# sam_file.close()

hight = 224

# data_list = []
# for chromosome, chr_len in zip(chr_list, chr_length):
#     if not os.path.exists(data_dir + 'flag/' + chromosome + '.txt'):
#         data_list.append((chromosome, chr_len))

def process(bam_path, chromosome, pic_length, data_dir):

    # ref_chromosome_filename = data_dir + "chr/" + chromosome + ".fa"
    # fa = pysam.FastaFile(ref_chromosome_filename)
    # chr_string = fa.fetch(chromosome)
    sam_file = pysam.AlignmentFile(bam_path, "rb")

    conjugate_m = torch.zeros(pic_length, dtype=torch.int)
    conjugate_i = torch.zeros(pic_length, dtype=torch.int)
    conjugate_d = torch.zeros(pic_length, dtype=torch.int)
    conjugate_s = torch.zeros(pic_length, dtype=torch.int)

    for read in sam_file.fetch(chromosome):
        if read.is_unmapped:
            continue
        start = read.reference_start
        if start % 5000 == 0:
            print(str(chromosome) + " " + str(start))

        # ref_read = chr_string[start:end]

        # read = read.get_forward_sequence()

        reference_index = start # % 2 == 0 :1  % 2 == 1 :0
        for operation, length in read.cigar: # (operation, length)
            if operation == 3 or operation == 7 or operation == 8:
                reference_index += length
            elif operation == 0:
                conjugate_m[reference_index:reference_index + length] += 1
                reference_index += length
            elif operation == 1:
                conjugate_i[reference_index] += length
            elif operation == 4:
                conjugate_s[reference_index - int(length / 2):reference_index + int(length / 2)] += 1
            elif operation == 2:
                conjugate_d[reference_index:reference_index + length] += 1
                reference_index += length

    sam_file.close()

    # rd_count = MaxMinNormalization(rd_count)  # The scope of rd_count value is [0, 1]

    return torch.cat([conjugate_m.unsqueeze(0), conjugate_i.unsqueeze(0), conjugate_d.unsqueeze(0), conjugate_s.unsqueeze(0)], 0)



def my(b_e, chromosome, flag):
    sam_file = pysam.AlignmentFile(bam_path, "rb")
    print("===== finish(position) " + chromosome + " " + flag)
    try:
        return ut.cigar_new_img_single_optimal(sam_file, chromosome, b_e[0], b_e[1])
    except Exception as e:
        print(e)
        print("Exception cigar_img_single_optimal")
        try:
            return ut.cigar_new_img_single_memory(sam_file, chromosome, b_e[0], b_e[1])
        except Exception as e:
            print(e)
            print("Exception cigar_new_img_single_memory")
            sam_file.close()
            # time.sleep(60)
            return my(b_e, chromosome, flag)



def p(sum_data):
    chromosome, chr_len = sum_data
    i = int(chr_len)
    # copy begin
    print("deal " + chromosome)
    p_position = torch.load(data_dir + 'position/' + chromosome + '/positive' + '.pt')
    n_position = torch.load(data_dir + 'position/' + chromosome + '/negative' + '.pt')
    # img/positive_cigar_img
    print("cigar start")
    positive_cigar_img = torch.empty(2, 4, hight, hight)

    positive_cigar_img[0] = my(p_position[i], chromosome, "p " + str(i))

    positive_cigar_img[1] = my(n_position[i], chromosome, "n " + str(i))

    save_path = data_dir + 'split_image/' + chromosome

    ut.mymkdir(save_path)

    torch.save(positive_cigar_img, save_path + '/pn_cigar_new_img' + str(i) + '.pt')

    print("cigar end")



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
    p([args.chr, int(args.len)])