import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pytorch_lightning.core.hooks import CheckpointHooks
import pytorch_lightning as pl
from ray import tune
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch.distributions.multivariate_normal as mn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import torchvision
import utilities as ut
from pudb import set_trace
from multiprocessing import Pool, cpu_count
import random
from transformers import AlbertModel
from sklearn.metrics import classification_report

class MultiLP(nn.Module):
    def __init__(self, full_dim):
        super(MultiLP, self).__init__()
        dim1 = full_dim[:-1]
        dim2 = full_dim[1:]
        self.layers=nn.ModuleList(
            nn.Sequential(
                nn.Linear(k, m),
                nn.ReLU(),
            ) for k, m in zip(dim1, dim2)
        )
    def forward(self,x):
        out=x
        for i,layer in enumerate(self.layers):
            out=layer(out)
        return out

class classfication(nn.Module):
    def __init__(self, full_dim):
        super(classfication, self).__init__()
        dim1 = full_dim[:-1]
        dim2 = full_dim[1:]
        self.layers=nn.ModuleList(
            nn.Sequential(
                nn.Linear(k, m),
                nn.ReLU(),
            ) for k, m in zip(dim1, dim2)
        )
    def forward(self,x):
        out=x
        for i,layer in enumerate(self.layers):
            out=layer(out)
        return out

class attention(nn.Module):
    def __init__(self, dim, out_dim):
        super(attention, self).__init__()
        self.Q_K = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.Sigmoid(),
        )
        self.V = nn.Sequential(
            nn.Linear(dim, out_dim),
        )

    def forward(self,x):
        qk = self.Q_K(x)
        v = self.V(x)
        out = torch.mul(qk, v)
        return out

class attention_classfication(nn.Module):
    def __init__(self, full_dim):
        super(attention_classfication, self).__init__()
        dim1 = full_dim[:-1]
        dim2 = full_dim[1:]
        self.layers=nn.ModuleList(
            nn.Sequential(
                attention(k, m),
                nn.Linear(m, m),
                nn.ReLU(inplace=True),
            ) for k, m in zip(dim1, dim2)
        )
    def forward(self,x):
        out=x
        for i,layer in enumerate(self.layers):
            out=layer(out)
        return out

class resnet_attention_classfication(nn.Module):
    def __init__(self, full_dim):
        super(resnet_attention_classfication, self).__init__()
        dim1 = full_dim[:-1]
        dim2 = full_dim[1:]
        self.layers=nn.ModuleList(
            nn.Sequential(
                attention(k, m),
                nn.Linear(m, m),
                nn.ReLU(inplace=True),
            ) for k, m in zip(dim1, dim2)
        )

        self.res2=nn.ModuleList(
            nn.Sequential(
                nn.Linear(full_dim[2 * index], full_dim[2 * (index + 1)]),
                nn.ReLU(inplace=True),
            ) for index in range(int(len(full_dim) / 2) - 1)
        )

    def forward(self,x):
        out = x
        for i in range(len(self.layers)):
            if i % 2 == 0:
                x = out
                out = self.layers[i](out)
            else:
                out = self.layers[i](out) + self.res2[int(i / 2)](x)

        return out

class conv2ds_sequential(nn.Module):
    def __init__(self, full_dim):
        super(conv2ds_sequential, self).__init__()
        dim1 = full_dim[:-1]
        dim2 = full_dim[1:]
        self.layers=nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels=k, out_channels=m, kernel_size=3, stride=1, padding=1), # (m, 224, 224)
                nn.BatchNorm2d(m),
                nn.ReLU(inplace=True),
            ) for k, m in zip(dim1, dim2)
        )
    def forward(self,x):
        out=x
        for i,layer in enumerate(self.layers):
            out=layer(out)
        return out

class conv2ds_after_resnet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(conv2ds_after_resnet, self).__init__()
        self.layers=nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels=k, out_channels=k+1, kernel_size=3, stride=1), # (m, 224, 224)
                nn.BatchNorm2d(k+1),
                nn.Conv2d(in_channels=k+1, out_channels=k+1, kernel_size=3, stride=1), # (m, 224, 224)
                nn.ReLU(inplace=True),
            ) for k in range(in_dim, out_dim)
        )

        self.layers2=nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels=k, out_channels=k+1, kernel_size=3, stride=1), # (m, 224, 224)
                nn.BatchNorm2d(k+1),
                nn.ReLU(inplace=True),
            ) for k in range(in_dim, out_dim)
        )

    def forward(self,x):
        out=x
        for i in range(len(self.layers)):
            out=self.layers[i](out) + self.layers2[i](out)

        return out

class FocalLoss(nn.Module):
    def __init__(self,alpha=0.25, gamma=2.0,use_sigmoid=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        r"""
        Focal loss
        :param pred: shape=(B,  HW)
        :param label: shape=(B, HW)
        """
        if self.use_sigmoid:
            pred = self.sigmoid(pred)
        pred = pred.view(-1)
        label = target.view(-1)
        pos = torch.nonzero(label > 0).squeeze(1)
        pos_num = max(pos.numel(),1.0)
        mask = ~(label == -1)
        pred = pred[mask]
        label= label[mask]
        focal_weight = self.alpha *(label- pred).abs().pow(self.gamma) * (label> 0.0).float() + (1 - self.alpha) * pred.abs().pow(self.gamma) * (label<= 0.0).float()
        loss = F.binary_cross_entropy(pred, label, reduction='none') * focal_weight
        return loss.sum()/pos_num

class IDENet(pl.LightningModule):

    def __init__(self, path, config):
        super(IDENet, self).__init__()

        self.lr = config["lr"]
        self.beta1 = config['beta1']
        self.beta2 = config['beta2']

        self.weight_decay = config['weight_decay']
        self.batch_size = config["batch_size"]
        # self.conv2d_dim_stride = config["conv2d_dim_stride"]  # [1, 3]
        # self.classfication_dim_stride = config["classfication_dim_stride"] #[1, 997]

        self.path = path
        # self.positive_img = positive_img
        # self.negative_img = negative_img
        # self.p_list = p_list
        # self.n_list = n_list

        # self.conv2ds = nn.Sequential(
        #     nn.Conv2d(in_channels=9, out_channels=8, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(in_channels=8, out_channels=7, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(in_channels=7, out_channels=6, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(in_channels=6, out_channels=5, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(in_channels=5, out_channels=4, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, stride=1, padding=1),
        # )
        # conv2d_dim = list(range(11, 3, -self.conv2d_dim_stride))

        conv2d_dim = list(range(7, 3, -1))
        # conv2d_dim = list(range(1, 3, 1)) # test
        conv2d_dim.append(3) # 6 -> 3
        self.conv2ds = conv2ds_sequential(conv2d_dim)

        # conv2d_dim = [1, 2, 3]
        # self.conv2ds = conv2ds_sequential(conv2d_dim)

        self.resnet_model = torchvision.models.resnet50(pretrained=True) # [224, 224] -> 1000

        # self.attention = attention(1000, 500)
        # full_dim = [1000 * 11, 500 * 11, 250 * 11, 125 * 11, 1375]
        # self.resnet_fullconnect = MultiLP(full_dim)
        # self.resnet_conv2ds = conv2ds_after_resnet(1, 6)
        # full_dim = [990 * 6, 990 * 3, 495 * 3, 743]
        # self.resnet_fullconnect = MultiLP(full_dim)


        # full_dim = [1000, 500, 250, 125, 62, 31, 15, 7]
        # full_dim = range(1000 + 768, 3, -self.classfication_dim_stride) # 1000 + 768 -> 2

        full_dim = [1000 + 768, 768 * 2, 768, 384, 192, 96, 48, 24, 12, 6]
        # full_dim = [1000, 768 * 2, 768, 384, 192, 96, 48, 24, 12, 6] # test
        self.classfication = resnet_attention_classfication(full_dim)

        self.softmax = nn.Sequential(
            nn.Linear(full_dim[-1], 3),
            nn.Softmax(1)
        )

        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = FocalLoss() # 样本不平衡loss


        # full_dim = [9, 4, 2, 1]
        # self.fullconnect = MultiLP(full_dim)

        # self.lstm_layer=torch.nn.LSTM(input_size=9, hidden_size=25, num_layers=5, bias=True,batch_first=True,dropout=0.2,bidirectional=False)

        # self.pool = nn.MaxPool1d(2, stride=2)
        # self.conv1d = nn.Conv1d(in_channels=1, out_channels = 512, kernel_size = 2)

        # full_dim = [11, 16, 32, 64, 128]
        # full_dim = list(range(11, 128, self.albert_dim_stride)) # 1000 + 768 -> 2
        # full_dim.append(128) # 6 -> 3
        # self.albert_fullconnect = MultiLP(full_dim)
        # self.conv1d = torch.nn.Conv1d(in_channels=128, out_channels = 128, kernel_size = 2, stride  = 1)

        self.bert = AlbertModel.from_pretrained("albert-base-v2")
        self.bert.embeddings.word_embeddings = nn.modules.sparse.Embedding(30000, 11, padding_idx=0)
        self.bert.embeddings.position_embeddings = nn.modules.sparse.Embedding(512, 11)
        self.bert.embeddings.token_type_embeddings = nn.modules.sparse.Embedding(2, 11)
        self.bert.embeddings.LayerNorm = nn.modules.normalization.LayerNorm((11,), eps=1e-12, elementwise_affine=True)
        self.bert.encoder.embedding_hidden_mapping_in = nn.modules.linear.Linear(in_features=11, out_features=768, bias=True)

    def training_validation_step(self, batch, batch_idx):
        x, y = batch  # x2(length, 12)
        del batch
        x1 = x["image"]
        x2 = x["list"]

        x1 = self.conv2ds(x1)

        # x1 = self.conv2ds(x1[:, 0:1, :, :]) # test

        x1 = self.resnet_model(x1)

        # x1 = x[:, :7 * 224 * 224].reshape(-1, 7, 224, 224)

        # x2 = x[:, 11 * 224 * 224:].reshape(-1, 9)
        # del x
        # x_sm = torch.empty(len(x2), 2)
        # x_lstm = torch.empty(len(x2), 25 * 5) # hidden_size * num_layers
        # # sum and max
        # for i, xx in enumerate(x2):
        #     xx = self.fullconnect(xx)
        #     x_sm[i][0] = torch.sum(xx)
        #     x_sm[i][1] = torch.max(xx)
        # # while len(xx)  池化小于最大长度后使用albert
        # x_id = torch.zeros([len(x2), 512, 128])
        # x_mask = torch.zeros([len(x2), 512], dtype=torch.int)
        # for i, xx in enumerate(x2):
        #     xx = self.albert_fullconnect(xx).t()
        #     while xx.shape[-1] > 512:
        #         xx = self.conv1d(xx)
        #     # n * 128
        #     x_id[i, :xx.shape[-1]] = xx.t()
        #     x_mask[i, :xx.shape[-1]] = 1

        # output = self.bert(input_ids=None,
        #     attention_mask=None,
        #     token_type_ids=x_mask,
        #     position_ids=None,
        #     head_mask=None,
        #     inputs_embeds=x_id,
        #     output_attentions=None,
        #     output_hidden_states=None,
        #     return_dict=None)

        # x2 = x2.reshape(-1, 11)  # b, 256, 11
        # x2 = self.albert_fullconnect(x2).reshape(-1, 512, 128)
        #     # n * 128

        # ===================== #
        x2 = self.bert(inputs_embeds=x2)[1]

        # output = self.bert(input_ids=None,
        #     attention_mask=None,
        #     token_type_ids=None,
        #     position_ids=None,
        #     head_mask=None,
        #     inputs_embeds=x2,
        #     output_attentions=None,
        #     output_hidden_states=None,
        #     return_dict=None)

        # output[1]    # b, 768

        # # 直接使用LSTM
        # for i, xx in enumerate(x2):
        #     x_lstm[i] = self.lstm_layer(xx.unsqueeze(0)).reshape(-1)

        y_t = torch.empty(len(y), 3).cuda()
        for i, y_item in enumerate(y):
            if y_item == 0:
                y_t[i] = torch.tensor([1, 0, 0])
            elif y_item == 1:
                y_t[i] = torch.tensor([0, 1, 0])
            else:
                y_t[i] = torch.tensor([0, 0, 1])

        y_hat = self.classfication(torch.cat([x1, x2], 1))
        # y_hat = self.classfication(x1) # test

        # y_hat = torch.cat([y_hat, xx2], 0)
        y_hat = self.softmax(y_hat)
        loss = self.criterion(y_hat, y_t)
        return loss, y, y_hat

    # def training_validation_step(self, batch, batch_idx):
    #     x, y = batch  # x2(length, 12)
    #     del batch
    #     x1 = x["image"]
    #     x2 = x["list"]

    #     x1 = self.conv2ds(x1)
    #     x1 = self.resnet_model(x1)

    #     # x1 = x[:, :7 * 224 * 224].reshape(-1, 7, 224, 224)

    #     # x2 = x[:, 11 * 224 * 224:].reshape(-1, 9)
    #     # del x
    #     # x_sm = torch.empty(len(x2), 2)
    #     # x_lstm = torch.empty(len(x2), 25 * 5) # hidden_size * num_layers
    #     # # sum and max
    #     # for i, xx in enumerate(x2):
    #     #     xx = self.fullconnect(xx)
    #     #     x_sm[i][0] = torch.sum(xx)
    #     #     x_sm[i][1] = torch.max(xx)
    #     # # while len(xx)  池化小于最大长度后使用albert
    #     # x_id = torch.zeros([len(x2), 512, 128])
    #     # x_mask = torch.zeros([len(x2), 512], dtype=torch.int)
    #     # for i, xx in enumerate(x2):
    #     #     xx = self.albert_fullconnect(xx).t()
    #     #     while xx.shape[-1] > 512:
    #     #         xx = self.conv1d(xx)
    #     #     # n * 128
    #     #     x_id[i, :xx.shape[-1]] = xx.t()
    #     #     x_mask[i, :xx.shape[-1]] = 1

    #     # output = self.bert(input_ids=None,
    #     #     attention_mask=None,
    #     #     token_type_ids=x_mask,
    #     #     position_ids=None,
    #     #     head_mask=None,
    #     #     inputs_embeds=x_id,
    #     #     output_attentions=None,
    #     #     output_hidden_states=None,
    #     #     return_dict=None)

    #     # x2 = x2.reshape(-1, 11)  # b, 256, 11
    #     # x2 = self.albert_fullconnect(x2).reshape(-1, 512, 128)
    #     #     # n * 128
    #     x2 = self.bert(inputs_embeds=x2)[1]

    #     # output = self.bert(input_ids=None,
    #     #     attention_mask=None,
    #     #     token_type_ids=None,
    #     #     position_ids=None,
    #     #     head_mask=None,
    #     #     inputs_embeds=x2,
    #     #     output_attentions=None,
    #     #     output_hidden_states=None,
    #     #     return_dict=None)

    #     # output[1]    # b, 768

    #     # # 直接使用LSTM
    #     # for i, xx in enumerate(x2):
    #     #     x_lstm[i] = self.lstm_layer(xx.unsqueeze(0)).reshape(-1)

    #     y_t = torch.empty(len(y), 3).cuda()
    #     for i, y_item in enumerate(y):
    #         if y_item == 0:
    #             y_t[i] = torch.tensor([1, 0, 0])
    #         elif y_item == 1:
    #             y_t[i] = torch.tensor([0, 1, 0])
    #         else:
    #             y_t[i] = torch.tensor([0, 0, 1])

    #     y_hat = self.classfication(torch.cat([x1, x2], 1))
    #     # y_hat = self.classfication(x1)

    #     # y_hat = torch.cat([y_hat, xx2], 0)
    #     y_hat = self.softmax(y_hat)
    #     loss = self.criterion(y_hat, y_t)
    #     return loss, y, y_hat


    def training_step(self, batch, batch_idx):
        loss, y, y_hat = self.training_validation_step(batch, batch_idx)

        # opt_e = self.optimizers()
        # self.manual_backward(loss)
        # opt_e.step()

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # set_trace()
        return {'loss': loss, 'y': y, 'y_hat' : torch.argmax(y_hat, dim = 1)}

    def training_epoch_end(self, output):
        # set_trace()
        y = []
        y_hat = []

        for out in output:
            y.extend(out['y'])
            y_hat.extend(out['y_hat'])

        y = torch.tensor(y).reshape(-1)
        y_hat = torch.tensor(y_hat).reshape(-1)

        metric = classification_report(y, y_hat, output_dict = True)

        self.log('train_mean', metric['accuracy'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # self.log('train_macro_f1', metric['macro avg']['f1-score'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_macro_pre', metric['macro avg']['precision'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_macro_re', metric['macro avg']['recall'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # self.log('train_0_f1', metric['0']['f1-score'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_0_pre', metric['0']['precision'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_0_re', metric['0']['recall'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # self.log('train_1_f1', metric['1']['f1-score'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_1_pre', metric['1']['precision'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_1_re', metric['1']['recall'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # self.log('train_2_f1', metric['2']['f1-score'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_2_pre', metric['2']['precision'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_2_re', metric['2']['recall'], on_step=False, on_epoch=True, prog_bar=True, logger=True)


    def validation_step(self, batch, batch_idx):
        loss, y, y_hat = self.training_validation_step(batch, batch_idx)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # set_trace()

        return {'y': y, 'y_hat' : torch.argmax(y_hat, dim = 1)}

    def validation_epoch_end(self, output):
        y = []
        y_hat = []

        for out in output:
            y.extend(out['y'])
            y_hat.extend(out['y_hat'])

        y = torch.tensor(y).reshape(-1)
        y_hat = torch.tensor(y_hat).reshape(-1)

        metric = classification_report(y, y_hat, output_dict = True)


        self.log('validation_mean', metric['accuracy'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # self.log('validation_macro_f1', metric['macro avg']['f1-score'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('validation_macro_pre', metric['macro avg']['precision'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('validation_macro_re', metric['macro avg']['recall'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # self.log('train_0_f1', metric['0']['f1-score'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('validation_0_pre', metric['0']['precision'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('validation_0_re', metric['0']['recall'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # self.log('train_1_f1', metric['1']['f1-score'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('validation_1_pre', metric['1']['precision'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('validation_1_re', metric['1']['recall'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # self.log('train_2_f1', metric['2']['f1-score'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('validation_2_pre', metric['2']['precision'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('validation_2_re', metric['2']['recall'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # tune.report(validation_mean = torch.mean((y == y_hat).float()))


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, output):
        self.validation_epoch_end(output)

    def prepare_data(self):
        train_proportion = 0.8
        input_data = ut.IdentifyDataset(self.path)
        dataset_size = len(input_data)
        indices = list(range(dataset_size))
        split = int(np.floor(train_proportion * dataset_size))
        random.seed(10)
        random.shuffle(indices)
        train_indices, test_indices = indices[:split], indices[split:]
        self.train_dataset= Subset(input_data, train_indices)
        self.test_dataset= Subset(input_data, test_indices)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=int(cpu_count()), prefetch_factor=10, shuffle=True) # sampler=self.wsampler)

    def val_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=int(cpu_count()))

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=int(cpu_count()))

    # @property
    # def automatic_optimization(self):
    #     return False

    def configure_optimizers(self):
        opt_e = torch.optim.Adam(
            self.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)
        # opt_d = torch.optim.Adam(
        #     self.line.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        return [opt_e]
