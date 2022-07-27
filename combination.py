import utilities as ut
from pudb import set_trace
import pandas as pd
import random
import numpy as np
import torch
import os
from multiprocessing import Pool, cpu_count
import pysam
import torchvision
from functools import partial




data_dir = "../datasets/NA12878_PacBio_MtSinai/"

bam_path = data_dir + "sorted_final_merged.bam"

vcf_filename = data_dir + "insert_result_data.csv.vcf"


# get chr list
sam_file = pysam.AlignmentFile(bam_path, "rb")
chr_list = sam_file.references
chr_length = sam_file.lengths
sam_file.close()

hight = 224

all_p_img = torch.empty(22199, 3+4+4, hight, hight)
all_p_list = torch.empty(22199, 512, 11)

# all_n_img = torch.empty(22199, 3+4+4, hight, hight)
# all_n_list = torch.empty(22199, 512, 11)


pool = Pool()
for index in range(22199):
    print("======= deal p " + str(index) + " =======")
    b = [data_dir + '/positive_img/' + str(index) + '.pt', data_dir + '/positive_list/' + str(index) + '.pt', data_dir + '/negative_img/' + str(index) + '.pt', data_dir + '/negative_list/' + str(index) + '.pt']

    all_p_img[index] = pool.apply_async(torch.load, (b[0], )).get()
    all_p_list[index] = pool.apply_async(torch.load, (b[1], )).get()
    # all_n_img[index] = pool.apply_async(torch.load, (b[2], )).get()
    # all_n_list[index] = pool.apply_async(torch.load, (b[3], )).get()

pool.close()
pool.join()   #调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
print("finish")

torch.save(all_p_img, data_dir + '/all_p_img' + '.pt')
torch.save(all_p_list, data_dir + '/all_p_list' + '.pt')
# torch.save(all_n_img, data_dir + '/all_n_img' + '.pt')
# torch.save(all_n_list, data_dir + '/all_n_list' + '.pt')

# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

pool = Pool()
for index in range(22199):
    print("======= deal n " + str(index) + " =======")
    b = [data_dir + '/positive_img/' + str(index) + '.pt', data_dir + '/positive_list/' + str(index) + '.pt', data_dir + '/negative_img/' + str(index) + '.pt', data_dir + '/negative_list/' + str(index) + '.pt']

    all_p_img[index] = pool.apply_async(torch.load, (b[2], )).get()
    all_p_list[index] = pool.apply_async(torch.load, (b[3], )).get()
    # all_n_img[index] = pool.apply_async(torch.load, (b[2], )).get()
    # all_n_list[index] = pool.apply_async(torch.load, (b[3], )).get()

pool.close()
pool.join()   #调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
print("finish")

torch.save(all_p_img, data_dir + '/all_n_img' + '.pt')
torch.save(all_p_list, data_dir + '/all_n_list' + '.pt')
# torch.save(all_n_img, data_dir + '/all_n_img' + '.pt')
# torch.save(all_n_list, data_dir + '/all_n_list' + '.pt')