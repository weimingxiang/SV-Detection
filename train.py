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
from hyperopt import hp
num_cuda = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = num_cuda
my_label = "7+11channel_predict_all"
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

if os.path.exists(data_dir + '/all_n_img' + '.pt') and not all_enforcement_refresh:
    # pool = Pool(2)
    # print("loading")
    # all_p_img = torch.load(data_dir + '/all_p_img' + '.pt')
    # all_n_img = torch.load(data_dir + '/all_n_img' + '.pt')
    # # all_p_img, all_n_img = pool.imap(torch.load, [data_dir + '/all_p_img' + '.pt', data_dir + '/all_n_img' + '.pt'])
    # # all_positive_img_i_list = torch.load(data_dir + '/all_p_list' + '.pt')
    # # all_negative_img_i_list = torch.load(data_dir + '/all_n_list' + '.pt')
    # # all_p_list, all_n_list = pool.imap(torch.load, [data_dir + '/all_p_list' + '.pt', data_dir + '/all_n_list' + '.pt'])

    # pool.close()
    print("loaded")
else:
    all_ins_img = torch.empty(0, 3, hight, hight)
    all_del_img = torch.empty(0, 3, hight, hight)
    all_negative_img = torch.empty(0, 3, hight, hight)

    all_ins_img_mid = torch.empty(0, 4, hight, hight)
    all_del_img_mid = torch.empty(0, 4, hight, hight)
    all_negative_img_mid = torch.empty(0, 4, hight, hight)

    all_ins_cigar_img = torch.empty(0, 4, hight, hight)
    all_del_cigar_img = torch.empty(0, 4, hight, hight)
    all_negative_cigar_img = torch.empty(0, 4, hight, hight)

    all_ins_list = torch.empty(0, 512, 11)
    all_del_list = torch.empty(0, 512, 11)
    all_negative_list = torch.empty(0, 512, 11)

    # pool = Pool(2)
    for chromosome, chr_len in zip(chr_list, chr_length):
        print("======= deal " + chromosome + " =======")

        print("position start")
        if os.path.exists(data_dir + 'position/' + chromosome + '/insert' + '.pt') and not position_enforcement_refresh:
            print("loading")
            ins_position = torch.load(data_dir + 'position/' + chromosome + '/insert' + '.pt')
            del_position = torch.load(data_dir + 'position/' + chromosome + '/delete' + '.pt')
            n_position = torch.load(data_dir + 'position/' + chromosome + '/negative' + '.pt')
        else:
            ins_position = []
            del_position = []
            n_position = []
            # insert
            insert_result_data = pd.read_csv(ins_vcf_filename, sep = "\t", index_col=0)
            insert_chromosome = insert_result_data[insert_result_data["CHROM"] == chromosome]
            row_pos = []
            for index, row in insert_chromosome.iterrows():
                row_pos.append(row["POS"])

            set_pos = set()

            for pos in row_pos:
                set_pos.update(range(pos - 100, pos + 100))

            for pos in row_pos:
                gap = 112
                # positive
                begin = pos - 1 - gap
                end = pos - 1 + gap
                if begin < 0:
                    begin = 0
                if end >= chr_len:
                    end = chr_len - 1

                ins_position.append([begin, end])

            # delete
            delete_result_data = pd.read_csv(del_vcf_filename, sep = "\t", index_col=0)
            delete_chromosome = delete_result_data[delete_result_data["CHROM"] == chromosome]
            row_pos = []
            row_end = []
            for index, row in delete_chromosome.iterrows():
                row_pos.append(row["POS"])
                row_end.append(row["END"])

            for pos in row_pos:
                set_pos.update(range(pos - 100, pos + 100))

            for pos, end in zip(row_pos, row_end):
                gap = int((end - pos) / 4)
                if gap == 0:
                    gap = 1
                # positive
                begin = pos - 1 - gap
                end = end - 1 + gap
                if begin < 0:
                    begin = 0
                if end >= chr_len:
                    end = chr_len - 1

                del_position.append([begin, end])

                #negative
                del_length = end - begin

                for _ in range(2):
                    end = begin

                    while end - begin < del_length / 2 + 1:
                        random_begin = random.randint(1, chr_len)
                        while random_begin in set_pos:
                            random_begin = random.randint(1, chr_len)
                        begin = random_begin - 1 - gap
                        end = begin + del_length
                        if begin < 0:
                            begin = 0
                        if end >= chr_len:
                            end = chr_len - 1


                    n_position.append([begin, end])


            save_path = data_dir + 'position/' + chromosome
            ut.mymkdir(save_path)
            torch.save(ins_position, save_path + '/insert' + '.pt')
            torch.save(del_position, save_path + '/delete' + '.pt')
            torch.save(n_position, save_path + '/negative' + '.pt')
        print("position end")

        print("img start")
        if os.path.exists(data_dir + 'image/' + chromosome + '/positive_img' + '.pt') and not img_enforcement_refresh:
            print("loading")
            # pool = Pool()
            # t_positive_img, t_negative_img = pool.map(torch.load, [data_dir + 'image/' + chromosome + '/positive_img' + '.pt', data_dir + 'image/' + chromosome + '/negative_img' + '.pt'])
            # pool.close()

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


        # if os.path.exists(data_dir + 'image_rd/' + chromosome + '/positive_img' + '.pt') and not enforcement_refresh:
        #     print("loading")
        #     _positive_img, _negative_img = pool.map(torch.load, [data_dir + 'image_rd/' + chromosome + '/positive_img' + '.pt', data_dir + 'image_rd/' + chromosome + '/negative_img' + '.pt'])
        #     print("load end")

        else:
            # chromosome_sign
            if os.path.exists(data_dir + "chromosome_sign/" + chromosome + ".pt") and not sign_enforcement_refresh:
                chromosome_sign = torch.load(data_dir + "chromosome_sign/" + chromosome + ".pt")
                mid_sign = torch.load(data_dir + "chromosome_sign/" + chromosome + "_mids_sign.pt")
                mid_sign_img = torch.load(data_dir + "chromosome_img/" + chromosome + "_m(i)d_sign.pt")
            else:
                ut.mymkdir(data_dir + "chromosome_sign/")
                chromosome_sign, mid_sign, mid_sign_list = ut.preprocess(bam_path, chromosome, chr_len, data_dir)
                torch.save(chromosome_sign, data_dir + "chromosome_sign/" + chromosome + ".pt")
                torch.save(mid_sign, data_dir + "chromosome_sign/" + chromosome + "_mids_sign.pt")
                torch.save(mid_sign_list, data_dir + "chromosome_sign/" + chromosome + "_m(i)d_sign.pt")
                # mid_sign_img = ut.mid_list2img(mid_sign_list, chromosome)
                mid_sign_img = torch.tensor(list2img.deal_list(mid_sign_list))
                ut.mymkdir(data_dir + "chromosome_img/")
                torch.save(mid_sign_img, data_dir + "chromosome_img/" + chromosome + "_m(i)d_sign.pt")
            #f # cigar
            # if os.path.exists(data_dir + "chromosome_cigar/" + chromosome + ".pt") and not cigar_enforcement_refresh:
            #     chromosome_cigar, chromosome_cigar_len, refer_q_table = torch.load(data_dir + "chromosome_cigar/" + chromosome + ".pt")
            # else:
            #     ut.mymkdir(data_dir + "chromosome_cigar/")
            #     chromosome_cigar, chromosome_cigar_len, refer_q_table = ut.preprocess_cigar(bam_path, chromosome)
            #     torch.save([chromosome_cigar, chromosome_cigar_len, refer_q_table], data_dir + "chromosome_cigar/" + chromosome + ".pt")
            #     # torch.save(chromosome_cigar, data_dir + "chromosome_cigar/" + chromosome + ".pt")

            # rd_depth_mean = torch.mean(chromosome_sign[2].float())

            ins_img = torch.empty(len(ins_position), 3, hight, hight)
            del_img = torch.empty(len(del_position), 3, hight, hight)
            negative_img = torch.empty(len(n_position), 3, hight, hight)

            ins_img_mid = torch.empty(len(ins_position), 4, hight, hight)
            del_img_mid = torch.empty(len(del_position), 4, hight, hight)
            negative_img_mid = torch.empty(len(n_position), 4, hight, hight)

            ins_img_i = torch.empty(len(ins_position), 512, 11)
            del_img_i = torch.empty(len(del_position), 512, 11)
            negative_img_i = torch.empty(len(n_position), 512, 11)


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

            resize = torchvision.transforms.Resize([512, 11])

            pool = Pool()

            for i, b_e in enumerate(ins_position):
                # ins_img[i] = ut.to_input_image_single(chromosome_sign[:, b_e[0]:b_e[1]]) # dim 3
                ins_img[i] = pool.apply_async(ut.to_input_image_single, (chromosome_sign[:, b_e[0]:b_e[1]], )).get()
                # ins_img_mid[i] = ut.to_input_image_single(mid_sign[:, b_e[0]:b_e[1]]) # dim 4
                ins_img_mid[i] = pool.apply_async(ut.to_input_image_single, (mid_sign[:, b_e[0]:b_e[1]], )).get()
                # ins_img_i[i] = resize(mid_sign_img[b_e[0]:b_e[1]].unsqueeze(0))
                ins_img_i[i] = pool.apply_async(resize, (mid_sign_img[b_e[0]:b_e[1]].unsqueeze(0), )).get()

                print("===== finish(ins_img) " + chromosome + " " + str(i))

            for i, b_e in enumerate(del_position):
                # del_img[i] = ut.to_input_image_single(chromosome_sign[:, b_e[0]:b_e[1]]) # dim 3
                del_img[i] = pool.apply_async(ut.to_input_image_single, (chromosome_sign[:, b_e[0]:b_e[1]], )).get()
                # del_img_mid[i] = ut.to_input_image_single(mid_sign[:, b_e[0]:b_e[1]]) # dim 4
                del_img_mid[i] = pool.apply_async(ut.to_input_image_single, (mid_sign[:, b_e[0]:b_e[1]], )).get()
                # del_img_i[i] = resize(mid_sign_img[b_e[0]:b_e[1]].unsqueeze(0))
                del_img_i[i] = pool.apply_async(resize, (mid_sign_img[b_e[0]:b_e[1]].unsqueeze(0), )).get()

                print("===== finish(del_img) " + chromosome + " " + str(i))


            for i, b_e in enumerate(n_position):
                # negative_img[i] = ut.to_input_image_single(chromosome_sign[:, b_e[0]:b_e[1]])
                negative_img[i] = pool.apply_async(ut.to_input_image_single, (chromosome_sign[:, b_e[0]:b_e[1]], )).get()
                # negative_img_mid[i] = ut.to_input_image_single(mid_sign[:, b_e[0]:b_e[1]]) # dim 4
                negative_img_mid[i] = pool.apply_async(ut.to_input_image_single, (mid_sign[:, b_e[0]:b_e[1]], )).get()
                # negative_img_i[i] = resize(mid_sign_img[b_e[0]:b_e[1]].unsqueeze(0))
                negative_img_i[i] = pool.apply_async(resize, (mid_sign_img[b_e[0]:b_e[1]].unsqueeze(0), )).get()


                print("===== finish(negative_img) " + chromosome + " " + str(i))


            # _positive_img, _negative_img = pool.starmap(ut.to_input_image, zip([positive_img, negative_img], [rd_depth_mean] * 2))
            # t_positive_img = ut.to_input_image(positive_img, rd_depth_mean)
            # t_negative_img = ut.to_input_image(negative_img, rd_depth_mean)
            pool.close()
            pool.join()
            print("save image start")

            save_path = data_dir + 'image/' + chromosome

            ut.mymkdir(save_path)
            # pool.starmap(torch.save, zip([_positive_img, _negative_img, positive_cigar_img, negative_cigar_img], [save_path + '/positive_img' + '.pt', save_path + '/negative_img' + '.pt', save_path + '/positive_cigar_img' + '.pt', save_path + '/negative_cigar_img' + '.pt']))

            torch.save(ins_img, save_path + '/ins_img' + '.pt')
            torch.save(del_img, save_path + '/del_img' + '.pt')

            torch.save(negative_img, save_path + '/negative_img' + '.pt')


            torch.save(ins_img_mid, save_path + '/ins_img_mid' + '.pt')
            torch.save(del_img_mid, save_path + '/del_img_mid' + '.pt')

            torch.save(negative_img_mid, save_path + '/negative_img_mid' + '.pt')


            torch.save(ins_img_i, save_path + '/ins_img_i' + '.pt')
            torch.save(del_img_i, save_path + '/del_img_i' + '.pt')

            torch.save(negative_img_i, save_path + '/negative_img_i' + '.pt')

        print("img end")

        # img/positive_cigar_img
        print("cigar start")
        if os.path.exists(data_dir + 'image/' + chromosome + '/positive_cigar_new_img' + '.pt') and not cigar_enforcement_refresh:
            print("loading")
            ins_cigar_img = torch.load(data_dir + 'image/' + chromosome + '/ins_cigar_new_img' + '.pt')
            del_cigar_img = torch.load(data_dir + 'image/' + chromosome + '/del_cigar_new_img' + '.pt')
            negative_cigar_img = torch.load(data_dir + 'image/' + chromosome + '/negative_cigar_new_img' + '.pt')
            # 由于未刷新数据增加的代码
            # all_p_img0 = positive_cigar_img[:, 0, :, :] + positive_cigar_img[:, 5, :, :]
            # all_n_img0 = negative_cigar_img[:, 0, :, :] + negative_cigar_img[:, 5, :, :]
            # positive_cigar_img = torch.cat([all_p_img0.unsqueeze(1), positive_cigar_img[:, 1:3, :, :]], dim = 1)
            # negative_cigar_img = torch.cat([all_n_img0.unsqueeze(1), negative_cigar_img[:, 1:3, :, :]], dim = 1)
            # save_path = data_dir + 'image/' + chromosome
            # torch.save(positive_cigar_img, save_path + '/positive_cigar_img' + '.pt')
            # torch.save(negative_cigar_img, save_path + '/negative_cigar_img' + '.pt')
            # end 从头跑程序需注释
        else:
            # sam_file = pysam.AlignmentFile(bam_path, "rb")
            ins_cigar_img = torch.empty(len(ins_position), 4, hight, hight)
            del_cigar_img = torch.empty(len(del_position), 4, hight, hight)
            negative_cigar_img = torch.empty(len(n_position), 4, hight, hight)
            for i, b_e in enumerate(ins_position):
                #f positive_cigar_img = torch.cat((positive_cigar_img, ut.cigar_img(chromosome_cigar, chromosome_cigar_len, refer_q_table[begin], refer_q_table[end]).unsqueeze(0)), 0)
                zoom = 1
                fail = 1
                while fail:
                    try:
                        fail = 0
                        ins_cigar_img[i] = ut.cigar_new_img_single_optimal(bam_path, chromosome, b_e[0], b_e[1], zoom)
                    except Exception as e:
                        fail = 1
                        zoom += 1
                        print(e)
                        print("Exception cigar_img_single_optimal" + str(zoom))
                #     try:
                #         positive_cigar_img[i] = ut.cigar_img_single_optimal_time2sapce(sam_file, chromosome, b_e[0], b_e[1])
                #     except Exception as e:
                #         print(e)
                #         print("Exception cigar_img_single_optimal_time2sapce")
                #         try:
                #             positive_cigar_img[i] = ut.cigar_img_single_optimal_time3sapce(sam_file, chromosome, b_e[0], b_e[1])
                #         except Exception as e:
                #             print(e)
                #             print("Exception cigar_img_single_optimal_time3sapce")
                #             positive_cigar_img[i] = ut.cigar_img_single_optimal_time6sapce(sam_file, chromosome, b_e[0], b_e[1])



                print("===== finish(ins_cigar_img) " + chromosome + " " + str(i))

            for i, b_e in enumerate(del_position):
                #f positive_cigar_img = torch.cat((positive_cigar_img, ut.cigar_img(chromosome_cigar, chromosome_cigar_len, refer_q_table[begin], refer_q_table[end]).unsqueeze(0)), 0)
                zoom = 1
                fail = 1
                while fail:
                    try:
                        fail = 0
                        del_cigar_img[i] = ut.cigar_new_img_single_optimal(bam_path, chromosome, b_e[0], b_e[1], zoom)
                    except Exception as e:
                        fail = 1
                        zoom += 1
                        print(e)
                        print("Exception cigar_img_single_optimal" + str(zoom))
                #     try:
                #         positive_cigar_img[i] = ut.cigar_img_single_optimal_time2sapce(sam_file, chromosome, b_e[0], b_e[1])
                #     except Exception as e:
                #         print(e)
                #         print("Exception cigar_img_single_optimal_time2sapce")
                #         try:
                #             positive_cigar_img[i] = ut.cigar_img_single_optimal_time3sapce(sam_file, chromosome, b_e[0], b_e[1])
                #         except Exception as e:
                #             print(e)
                #             print("Exception cigar_img_single_optimal_time3sapce")
                #             positive_cigar_img[i] = ut.cigar_img_single_optimal_time6sapce(sam_file, chromosome, b_e[0], b_e[1])



                print("===== finish(del_position) " + chromosome + " " + str(i))


            for i, b_e in enumerate(n_position):
                #f negative_cigar_img = torch.cat((negative_cigar_img, ut.cigar_img(chromosome_cigar, chromosome_cigar_len, refer_q_table[begin], refer_q_table[end]).unsqueeze(0)), 0)
                zoom = 1
                fail = 1
                while fail:
                    try:
                        fail = 0
                        negative_cigar_img[i] = ut.cigar_new_img_single_optimal(bam_path, chromosome, b_e[0], b_e[1], zoom)
                    except Exception as e:
                        fail = 1
                        zoom += 1
                        print(e)
                        print("Exception cigar_img_single_optimal" + str(zoom))

                    # try:
                    #     negative_cigar_img[i] = ut.cigar_img_single_optimal_time2sapce(sam_file, chromosome, b_e[0], b_e[1])
                    # except Exception as e:
                    #     print(e)
                    #     print("Exception cigar_img_single_optimal_time2sapce")
                    #     try:
                    #         negative_cigar_img[i] = ut.cigar_img_single_optimal_time3sapce(sam_file, chromosome, b_e[0], b_e[1])
                    #     except Exception as e:
                    #         print(e)
                    #         print("Exception cigar_img_single_optimal_time3sapce")
                    #         negative_cigar_img[i] = ut.cigar_img_single_optimal_time6sapce(sam_file, chromosome, b_e[0], b_e[1])


                print("===== finish(n_position) " + chromosome + " " + str(i))
            # sam_file.close()

            save_path = data_dir + 'image/' + chromosome

            torch.save(ins_cigar_img, save_path + '/ins_cigar_new_img' + '.pt')
            torch.save(del_cigar_img, save_path + '/del_cigar_new_img' + '.pt')
            torch.save(negative_cigar_img, save_path + '/negative_cigar_new_img' + '.pt')
        print("cigar end")

        all_ins_img = torch.cat((all_ins_img, ins_img), 0)
        all_del_img = torch.cat((all_del_img, del_img), 0)
        all_negative_img = torch.cat((all_negative_img, negative_img), 0)

        all_ins_cigar_img = torch.cat((all_ins_cigar_img, ins_cigar_img), 0)
        all_del_cigar_img = torch.cat((all_del_cigar_img, del_cigar_img), 0)
        all_negative_cigar_img = torch.cat((all_negative_cigar_img, negative_cigar_img), 0)

        all_ins_img_mid = torch.cat((all_ins_img_mid, ins_img_mid), 0)
        all_del_img_mid = torch.cat((all_del_img_mid, del_img_mid), 0)
        all_negative_img_mid = torch.cat((all_negative_img_mid, negative_img_mid), 0)

        all_ins_list = torch.cat((all_ins_list, ins_img_i), 0)
        all_del_list = torch.cat((all_del_list, del_img_i), 0)

        all_negative_list = torch.cat((all_negative_list, negative_img_i), 0)


    all_ins_img = torch.cat([all_ins_img, all_ins_img_mid, all_ins_cigar_img], 1) # 3, 4, 3
    all_del_img = torch.cat([all_del_img, all_del_img_mid, all_del_cigar_img], 1) # 3, 4, 3
    all_n_img = torch.cat([all_negative_img, all_negative_img_mid, all_negative_cigar_img], 1)

    torch.save(all_ins_img, data_dir + '/all_ins_img' + '.pt')
    torch.save(all_del_img, data_dir + '/all_del_img' + '.pt')
    torch.save(all_n_img, data_dir + '/all_n_img' + '.pt')

    torch.save(all_ins_list, data_dir + '/all_ins_list' + '.pt')
    torch.save(all_del_list, data_dir + '/all_del_list' + '.pt')
    torch.save(all_negative_list, data_dir + '/all_negative_list' + '.pt')




logger = TensorBoardLogger(os.path.join("/home/xwm/DeepSVFilter/code", "channel_predict"), name=my_label)

checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints_predict/" + my_label,
    filename='{epoch:02d}-{validation_mean:.2f}-{train_mean:.2f}',
    monitor="validation_mean",
    verbose=False,
    save_last=None,
    save_top_k=1,
    # save_weights_only=True,
    mode="max",
    auto_insert_metric_name=True,
    every_n_train_steps=None,
    train_time_interval=None,
    every_n_epochs=None,
    save_on_train_epoch_end=None,
    every_n_val_epochs=None
)

def main_train():
    config = {
        "lr": 7.1873e-06,
        "batch_size": 14, # 14,
        "beta1": 0.9,
        "beta2": 0.999,
        'weight_decay': 0.0011615,
        # "classfication_dim_stride": 20, # no use
    }
    # config = {
    #     "lr": 1.11376e-7,
    #     "batch_size": 4, # 14,
    #     "beta1": 0.899906,
    #     "beta2": 0.998613,
    #     'weight_decay': 0.0049974,
    #     "classfication_dim_stride": 201,
    # }


    model = IDENet(data_dir, config)

    # resume = "./checkpoints_predict/" + my_label + "/epoch=33-validation_mean=0.95-train_mean=0.97.ckpt"

    trainer = pl.Trainer(
        max_epochs=30,
        gpus=1,
        check_val_every_n_epoch=1,
        # replace_sampler_ddp=False,
        logger=logger,
        # val_percent_check=0,
        callbacks=[checkpoint_callback],
        # resume_from_checkpoint=resume
        # auto_lr_find=True,
    )

    trainer.fit(model)


def train_tune(config, checkpoint_dir=None, num_epochs=200, num_gpus=1):
    # config.update(ori_config)
    model = IDENet(data_dir, config)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        check_val_every_n_epoch=1,
        logger=logger,
        # progress_bar_refresh_rate=0,
        callbacks=[checkpoint_callback],
        # callbacks = TuneReportCallback(
        # {
        #     "validation_loss": "validation_loss",
        #     "validation_0_f1": "validation_0_f1",
        #     "validation_1_f1": "validation_1_f1",
        #     "validation_2_f1": "validation_2_f1",
        #     "validation_mean": "validation_mean",
        # },
        # on="validation_end"),
        # auto_scale_batch_size="binsearch",
    )
    trainer.fit(model)

class MyStopper(tune.Stopper):
    def __init__(self, metric, value, epoch = 1):
        self._metric = metric
        self._value = value
        self._epoch = epoch

    def __call__(self, trial_id, result):
        """Return a boolean representing if the tuning has to stop."""
        # If the current iteration has to stop
        # if result[self._metric] < self._mean:
        #     # we increment the total counter of iterations
        #     self._iterations += 1
        # else:
        #     self._iterations = 0


        # and then call the method that re-executes
        # the checks, including the iterations.
        # return self._iterations >= self._patience
        return (result["training_iteration"] > self._epoch) and (result[self._metric] < self._value)


    def stop_all(self):
        """Return whether to stop and prevent trials from starting."""
        return False

# def stopper(trial_id, result):
#     return result["validation_mean"] <= 0.343

def gan_tune(num_samples=-1, num_epochs=30, gpus_per_trial=1):
    # config = {
    #     "lr": tune.loguniform(1e-7, 1e-5),
    #     "batch_size": 14,
    #     "beta1": 0.9, # tune.uniform(0.895, 0.905),
    #     "beta2": 0.999, # tune.uniform(0.9989, 0.9991),
    #     'weight_decay': tune.uniform(0, 0.01),
    #     # "conv2d_dim_stride": tune.lograndint(1, 6),
    #     # "classfication_dim_stride": tune.lograndint(20, 700),
    # }
    config = {
        "batch_size": 14,
        "beta1": 0.9,
        "beta2": 0.999,
        "lr": 7.187267009530772e-06,
        "weight_decay": 0.0011614665567890423
        # "classfication_dim_stride": 20, # no use
    }

    bayesopt = HyperOptSearch(config, metric="validation_mean", mode="max")
    re_search_alg = Repeater(bayesopt, repeat=1)

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2,
        )

    reporter = CLIReporter(
        metric_columns=['train_loss', "train_mean", 'validation_loss', "validation_mean"]
        )

    analysis = tune.run(
        tune.with_parameters(
            train_tune,
            num_epochs=num_epochs,
        ),
        local_dir="/home/xwm/DeepSVFilter/code/",
        resources_per_trial={
            "cpu": 5,
            "gpu": 1,
        },
        # stop = MyStopper("validation_mean", value = 0.343, epoch = 1),
        config=config,
        num_samples=num_samples,
        metric='validation_mean',
        mode='max',
        scheduler=scheduler,
        progress_reporter=reporter,
        resume="AUTO",
        search_alg=re_search_alg,
        max_failures = -1,
        reuse_actors = True,
        # server_port = 60060,
        name="tune" + num_cuda)



# main_train()
# # ray.init(num_cpus=12, num_gpus=3)
ray.init()
gan_tune()
