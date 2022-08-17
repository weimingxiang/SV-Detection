import utilities as ut
from pudb import set_trace
import pandas as pd
import random
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
import os
from net import IDENet
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from multiprocessing import Pool, cpu_count
import pysam
import time
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest import Repeater
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
import list2img



os.environ["CUDA_VISIBLE_DEVICES"] = "0"


seed_everything(2022)

# data_dir = "../datasets/NA12878_PacBio_MtSinai/"
data_dir = "/home/xwm/DeepSVFilter/datasets/NA12878_PacBio_MtSinai/"


bam_path = data_dir + "sorted_final_merged.bam"

ins_vcf_filename = data_dir + "insert_result_data.csv.vcf"
del_vcf_filename = data_dir + "delete_result_data.csv.vcf"


all_enforcement_refresh = 0
position_enforcement_refresh = 0
img_enforcement_refresh = 0
sign_enforcement_refresh = 0 # attention
cigar_enforcement_refresh = 0

# get chr list
sam_file = pysam.AlignmentFile(bam_path, "rb")
chr_list = sam_file.references
chr_length = sam_file.lengths
sam_file.close()

hight = 224

all_ins_img = torch.empty(0, 3, hight, hight)
all_del_img = torch.empty(0, 3, hight, hight)
all_negative_img = torch.empty(0, 3, hight, hight)

all_ins_img_mid = torch.empty(0, 4, hight, hight)
all_del_img_mid = torch.empty(0, 4, hight, hight)
all_negative_img_mid = torch.empty(0, 4, hight, hight)

all_ins_list = torch.empty(0, 512, 11)
all_del_list = torch.empty(0, 512, 11)
all_negative_list = torch.empty(0, 512, 11)

# pool = Pool(2)
for chromosome, chr_len in zip(chr_list, chr_length):

    print("======= deal " + chromosome + " =======")

    print("img start")
    print("loading")

    save_path = data_dir + 'image/' + chromosome

    ins_img = torch.load(save_path + '/ins_img' + '.pt')
    del_img = torch.load(save_path + '/del_img' + '.pt')

    negative_img = torch.load(save_path + '/negative_img' + '.pt')


    ins_img_mid = torch.load(save_path + '/ins_img_mid' + '.pt')
    del_img_mid = torch.load(save_path + '/del_img_mid' + '.pt')

    negative_img_mid = torch.load(save_path + '/negative_img_mid' + '.pt')


    ins_img_i = torch.load(save_path + '/ins_img_i' + '.pt')
    del_img_i = torch.load(save_path + '/del_img_i' + '.pt')

    negative_img_i = torch.load(save_path + '/negative_img_i' + '.pt')

    all_ins_img = torch.cat((all_ins_img, ins_img), 0)
    all_del_img = torch.cat((all_del_img, del_img), 0)
    all_negative_img = torch.cat((all_negative_img, negative_img), 0)


    all_ins_img_mid = torch.cat((all_ins_img_mid, ins_img_mid), 0)
    all_del_img_mid = torch.cat((all_del_img_mid, del_img_mid), 0)
    all_negative_img_mid = torch.cat((all_negative_img_mid, negative_img_mid), 0)

    all_ins_list = torch.cat((all_ins_list, ins_img_i), 0)
    all_del_list = torch.cat((all_del_list, del_img_i), 0)

    all_negative_list = torch.cat((all_negative_list, negative_img_i), 0)



torch.save(all_ins_list, data_dir + '/all_ins_list' + '.pt')
torch.save(all_del_list, data_dir + '/all_del_list' + '.pt')
torch.save(all_negative_list, data_dir + '/all_negative_list' + '.pt')
del all_ins_list
del all_del_list
del all_negative_list

print("======== 1 ========")

all_ins_img = torch.cat([all_ins_img, all_ins_img_mid], 1) # 3, 3, 3
del all_ins_img_mid
torch.save(all_ins_img, data_dir + '/all_ins_img' + '.pt')
del all_ins_img
print("======== 2 ========")

all_del_img = torch.cat([all_del_img, all_del_img_mid], 1) # 3, 3, 3
del all_del_img_mid
torch.save(all_del_img, data_dir + '/all_del_img' + '.pt')
del all_del_img
print("======== 3 ========")

all_n_img = torch.cat([all_negative_img, all_negative_img_mid], 1)
del all_negative_img_mid
torch.save(all_n_img, data_dir + '/all_n_img' + '.pt')
del all_n_img

print("======== 4 ========")

