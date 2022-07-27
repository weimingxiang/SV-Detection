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
import list2img

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

import sys
import os
import random
import numpy as np
from pudb import set_trace
import argparse
from glob import glob
import torch
import torchvision
from multiprocessing import Pool, cpu_count
import pysam
import time

hight = 224
resize = torchvision.transforms.Resize([hight, hight])

# # 通过pos计算值
# def pos2value(lis, pos):
#     pos_float = pos - int(pos)
#     x1 = lis[int(pos)] * (1 - pos_float)
#     x2 = lis[int(pos) + 1] * pos_float
#     return x1 + x2

# #计算上下四分位数
# #计算上下边缘
# #计算中位数
# def count_quartiles_median(lis):
#     length = float(len(lis)) - 1
#     q1 = length / 4
#     q3 = length * 3 / 4
#     q4 = q3 + 1.5 * (q3 - q1)
#     q = q1 - 1.5 * (q3 - q1)
#     q2 = length / 2  # 中位数
#     return pos2value(lis, q), pos2value(lis, q1), pos2value(lis, q2), pos2value(lis, q3), pos2value(lis, q4)

def get_rms(records):
    """
    均方根值 反映的是有效值而不是平均值
    """
    return np.sqrt(sum([x ** 2 for x in records]) / len(records))

# def get_gm(records, i):
#     """
#     几何平均
#     """
#     return (np.prod(records)) ** (1 / len(records))

def get_hm(records):
    """
    调和平均
    """
    return len(records) / sum([1 / x for x in records])

def get_cv(records): #标准分和变异系数
    mean = np.mean(records)
    std = np.std(records)
    cv = std / mean
    return mean, std, cv

def mid_list2img(mid_sign_list, chromosome):
    mid_sign_img = torch.zeros(len(mid_sign_list), 11)
    for i, mid_sign in enumerate(mid_sign_list):
        if i % 50000 == 0:
            print(str(chromosome) + "\t" + str(i))
        mid_sign_img[i] = torch.tensor(list2img.deal_list(mid_sign_list))

    return mid_sign_img

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
        if read.is_unmapped or read.is_secondary:
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

    # copy begin
    print("deal " + chromosome)


    # ut.mymkdir(data_dir + "chromosome_sign/")
    # chromosome_sign, mid_sign, mid_sign_list = ut.preprocess(bam_path, chromosome, chr_len, data_dir)
    # torch.save(chromosome_sign, data_dir + "chromosome_sign/" + chromosome + ".pt")
    # torch.save(mid_sign, data_dir + "chromosome_sign/" + chromosome + "_mids_sign.pt")
    # mid_sign_list = torch.load(data_dir + "chromosome_sign/" + chromosome + "_m(i)d_sign.pt")
    # del chromosome_sign
    # del mid_sign
    # mid_sign_img = ut.mid_list2img(mid_sign_list, chromosome)
    # del mid_sign_list
    # ut.mymkdir(data_dir + "chromosome_img/")
    # torch.save(mid_sign_img, data_dir + "chromosome_img/" + chromosome + "_m(i)d_sign.pt")


    # # 1
    # mid_sign = process(bam_path, chromosome, chr_len, data_dir)
    # torch.save(mid_sign, data_dir + "chromosome_sign/" + chromosome + "_mids_sign.pt")

    # # 2
    # p_position = torch.load(data_dir + 'position/' + chromosome + '/positive' + '.pt')
    # n_position = torch.load(data_dir + 'position/' + chromosome + '/negative' + '.pt')

    # mid_sign = torch.load(data_dir + "chromosome_sign/" + chromosome + "_mids_sign.pt")

    # positive_img_mid = torch.empty(len(p_position), 4, hight, hight)
    # negative_img_mid = torch.empty(len(n_position), 4, hight, hight)

    # for i, b_e in enumerate(p_position):
    #     positive_img_mid[i] = ut.to_img_mid_single(mid_sign[:, b_e[0]:b_e[1]]) # dim 3
    #     print("===== finish(positive_img) " + chromosome + " " + str(i))


    # for i, b_e in enumerate(n_position):
    #     negative_img_mid[i] = ut.to_img_mid_single(mid_sign[:, b_e[0]:b_e[1]]) # dim 3

    #     print("===== finish(negative_img) " + chromosome + " " + str(i))

    # save_path = data_dir + 'image/' + chromosome

    # torch.save(positive_img_mid, save_path + '/positive_img_mids' + '.pt')
    # torch.save(negative_img_mid, save_path + '/negative_img_mids' + '.pt')

    # # 3
    # mid_sign_list = torch.load(data_dir + "chromosome_sign/" + chromosome + "_m(i)d_sign.pt")

    # # # mid_sign_img = mid_list2img(mid_sign_list, chromosome)
    # mid_sign_img = torch.tensor(list2img.deal_list(mid_sign_list))

    # del mid_sign_list

    # ut.mymkdir(data_dir + "chromosome_img/")
    # torch.save(mid_sign_img, data_dir + "chromosome_img/" + chromosome + "_m(i)d_sign.pt")


    # mid_sign_list = torch.load(data_dir + "chromosome_sign/" + chromosome + "_m(i)d_sign.pt")
    # ut.data_write_csv(data_dir + "chromosome_sign/" + chromosome + "_m(i)d_sign.csv", mid_sign_list)

    # p_position = torch.load(data_dir + 'position/' + chromosome + '/positive' + '.pt')
    # n_position = torch.load(data_dir + 'position/' + chromosome + '/negative' + '.pt')
    # # img/positive_cigar_img
    # print("cigar start")
    # positive_cigar_img = torch.empty(len(p_position), 4, hight, hight)
    # negative_cigar_img = torch.empty(len(n_position), 4, hight, hight)

    # for i in range(len(p_position)):
    # # for i, b_e in enumerate(p_position):
    #     positive_cigar_img[i] = my(p_position[i], chromosome, "p " + str(i))
    # for i in range(len(n_position)):
    # # for i, b_e in enumerate(n_position):
    #     negative_cigar_img[i] = my(n_position[i], chromosome, "n " + str(i))

    # # pool = Pool(maxtasksperchild = 10)
    # # a = pool.imap(partial(my, chromosome = chromosome, flag = "p"), p_position)
    # # print("============")
    # # b = pool.imap(partial(my, chromosome = chromosome, flag = "n"), n_position)
    # # print("pool close begin")
    # # # b = pool.starmap(my, zip(n_position, repeat(chromosome), repeat("n")))
    # # pool.close()
    # # print("pool close end")
    # # pool.join()
    # # print("pool join end")

    # # for i, item in enumerate(a):
    # #     positive_cigar_img[i] = item
    # #     print("===== finish(position) " + chromosome + " p " + str(i))

    # # for i, item in enumerate(b):
    # #     negative_cigar_img[i] = item
    # #     print("===== finish(position) " + chromosome + " n " + str(i))

    # # del a
    # # del b

    # save_path = data_dir + 'image/' + chromosome

    # torch.save(positive_cigar_img, save_path + '/positive_cigar_new_img' + '.pt')
    # torch.save(negative_cigar_img, save_path + '/negative_cigar_new_img' + '.pt')

    p_position = torch.load(data_dir + 'position/' + chromosome + '/positive' + '.pt')
    n_position = torch.load(data_dir + 'position/' + chromosome + '/negative' + '.pt')

    chromosome_sign = torch.load(data_dir + "chromosome_sign/" + chromosome + ".pt")
    mid_sign = torch.load(data_dir + "chromosome_sign/" + chromosome + "_mids_sign.pt")
    # mid_sign_img = torch.load(data_dir + "chromosome_img/" + chromosome + "_m(i)d_sign.pt")


    positive_img = torch.empty(len(p_position), 3, hight, hight)
    negative_img = torch.empty(len(n_position), 3, hight, hight)

    positive_img_mid = torch.empty(len(p_position), 4, hight, hight)
    negative_img_mid = torch.empty(len(n_position), 4, hight, hight)
    # positive_img_i = torch.empty(len(p_position), 512, 11)
    # negative_img_i = torch.empty(len(n_position), 512, 11)


    # insert_chromosome = insert_result_data[insert_result_data["CHROM"] == chromosome]
    # for index, row in insert_chromosome.iterrows():
    #     gap = int((row["END"] - row["POS"]) / 4)
    #     if gap == 0:
    #         gap = 1
    #     # positive
    #     begin = row["POS"] - 1 - gap
    #     end = row["END"] - 1 + gap
    #     if begin < 0:
    #         begin = 0
    #     if end >= len(rd_depth):
    #         end = len(rd_depth) - 1
    #     positive_img.append(chromosome_sign[:, begin:end])
    #     #f positive_cigar_img = torch.cat((positive_cigar_img, ut.cigar_img(chromosome_cigar, chromosome_cigar_len, refer_q_table[begin], refer_q_table[end]).unsqueeze(0)), 0)
    #     positive_cigar_img = torch.cat((positive_cigar_img, ut.cigar_img_single(bam_path, chromosome, begin, end).unsqueeze(0)), 0)

    #     #negative
    #     random_begin = random.randint(1,len(rd_depth))
    #     while random_begin == row["POS"]:
    #         random_begin = random.randint(1,len(rd_depth))
    #     begin = random_begin - 1 - gap
    #     end = begin + row["END"] - row["POS"] + 2 * gap
    #     if begin < 0:
    #         begin = 0
    #     if end >= len(rd_depth):
    #         end = len(rd_depth) - 1
    #     negative_img.append(chromosome_sign[:, begin:end])
    #     #f negative_cigar_img = torch.cat((negative_cigar_img, ut.cigar_img(chromosome_cigar, chromosome_cigar_len, refer_q_table[begin], refer_q_table[end]).unsqueeze(0)), 0)
    #     negative_cigar_img = torch.cat((negative_cigar_img, ut.cigar_img_single(bam_path, chromosome, begin, end).unsqueeze(0)), 0)

    # resize = torchvision.transforms.Resize([512, 11])

    for i, b_e in enumerate(p_position):
        positive_img[i] = ut.to_input_image_single(chromosome_sign[:, b_e[0]:b_e[1]]) # dim 3
        positive_img_mid[i] = ut.to_input_image_single(mid_sign[:, b_e[0]:b_e[1]]) # dim 3
        # positive_img_i[i] = resize(mid_sign_img[b_e[0]:b_e[1]].unsqueeze(0))
        print("===== finish(positive_img) " + chromosome + " " + str(i))


    for i, b_e in enumerate(n_position):
        negative_img[i] = ut.to_input_image_single(chromosome_sign[:, b_e[0]:b_e[1]])
        negative_img_mid[i] = ut.to_input_image_single(mid_sign[:, b_e[0]:b_e[1]]) # dim 3
        # negative_img_i[i] = resize(mid_sign_img[b_e[0]:b_e[1]].unsqueeze(0))

        print("===== finish(negative_img) " + chromosome + " " + str(i))


    # _positive_img, _negative_img = pool.starmap(ut.to_input_image, zip([positive_img, negative_img], [rd_depth_mean] * 2))
    # t_positive_img = ut.to_input_image(positive_img, rd_depth_mean)
    # t_negative_img = ut.to_input_image(negative_img, rd_depth_mean)
    print("save image start")

    save_path = data_dir + 'image/' + chromosome

    ut.mymkdir(save_path)
    # pool.starmap(torch.save, zip([_positive_img, _negative_img, positive_cigar_img, negative_cigar_img], [save_path + '/positive_img' + '.pt', save_path + '/negative_img' + '.pt', save_path + '/positive_cigar_img' + '.pt', save_path + '/negative_cigar_img' + '.pt']))
    torch.save(positive_img, save_path + '/positive_img' + '.pt')
    torch.save(negative_img, save_path + '/negative_img' + '.pt')

    torch.save(positive_img_mid, save_path + '/positive_img_mids' + '.pt')
    torch.save(negative_img_mid, save_path + '/negative_img_mids' + '.pt')

    # torch.save(positive_img_i, save_path + '/positive_img_m(i)d' + '.pt')
    # torch.save(negative_img_i, save_path + '/negative_img_m(i)d' + '.pt')


    print("cigar end")

    # copy end
    torch.save(1, data_dir + 'flag/' + chromosome + '.txt2')





# pool = Pool()                # 创建进程池对象，进程数与multiprocessing.cpu_count()相同
# pool.imap_unordered(p, data_list)
# # pool.map(p, data_list)
# pool.close()
# pool.join()

# # p("chr1")


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