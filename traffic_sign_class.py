# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import csv
import argparse
import matplotlib.pyplot as plt
import cv2
import math
import random
import time
import os
from visdom import Visdom


class DataSet(data.Dataset):  # 数据集
    def __init__(self, is_train=True, do_transform=None):
        self.do_transform = do_transform  # 图片数据转换
        self.data_path = ['./traffic-sign/train_label.csv', './traffic-sign/test_label.csv']  # 标签文件位置
        self.rotate_range = 15.0  # 增广时图片旋转范围
        self.img_size = 224  # 模型输入图片尺寸
        train_set, test_set = self.read_csv()  # 训练 测试集
        train_set = self.augment_train(train_set)  # 训练集增广
        test_set = self.process_test(test_set)  # 测试集增加属性
        self.data_set = train_set if is_train else test_set  # 选定数据集
        self.class_id, self.class_count, self.class_weight = self.cal_class_weight(train_set)  # 分类id、数量、权重
        self.border = math.ceil(self.img_size / 2 * (2 ** 0.5 * math.sin((45 + self.rotate_range) / 180 * math.pi) - 1))

    def read_csv(self):  # 读取.csv标签文件
        data_all = []
        for _i in self.data_path:  # 对应分别形成训练集、验证集
            sub_data = []
            with open(_i, 'r') as f:
                csv_data = csv.reader(f)
                _ = next(csv_data)  # csv file first line, the head
                for line in csv_data:
                    sub_data.append({'image_location': line[1], 'class_id': int(line[2])})
            data_all.append(sub_data)
        train_set, test_set = data_all
        return train_set, test_set

    def augment_train(self, train_set):  # 训练集增广
        class_id, class_count = self.analyze_trainset(train_set)  # 分析训练集得到各类标识牌训练图片数量
        augment_trainset = []
        max_sample_num = max(class_count)  # 最多训练图片类的图片数量
        for _i in train_set:
            augment_num = max_sample_num / class_count[class_id.index(_i['class_id'])]  # 增广倍数
            if augment_num < 2:  # 样本数量差小于两倍，不作增广
                augment_trainset.append({'image_location': _i['image_location'], 'class_id': _i['class_id'],
                                         'rotate_angle': 0})
            else:  # 样本数量差大于两倍，进行旋转增广
                rotate_interval = self.rotate_range / augment_num  # 旋转角度间隔 = 旋转角度范围 / 增广倍数
                for _j in range(int(-augment_num / 2), int(augment_num / 2) + 1):
                    augment_trainset.append({'image_location': _i['image_location'], 'class_id': _i['class_id'],
                                             'rotate_angle': _j * rotate_interval})
        return augment_trainset

    @staticmethod
    def process_test(test_set):  # 为测试集增加rotate_angle键值，便于程序运行
        process_test_set = []
        for _i in test_set:
            process_test_set.append({'image_location': _i['image_location'], 'class_id': _i['class_id'],
                                     'rotate_angle': 0})
        return process_test_set

    def cal_class_weight(self, train_set):  # 计算分类权重，供后续模型训练时使用
        class_weight = []
        class_id, class_count = self.analyze_trainset(train_set)  # 再次分析训练集得到增广后各类图片数量
        max_sample_num = max(class_count)  # 最多训练图片类的图片数量
        for _i in range(len(class_id)):
            class_weight.append(max_sample_num / class_count[_i])  # 计算权重，图片少的类将得到更大权重
        return class_id, class_count, class_weight

    def __getitem__(self, idx):
        one_data = self.data_set[idx]
        img = cv2.imread(one_data['image_location'])  # 读取图片
        img_class = one_data['class_id']  # 图片类id
        img_rotate_angle = one_data['rotate_angle']  # 图片旋转角度
        img = self.pre_treat(img, img_rotate_angle)  # 进行预处理
        if self.do_transform is not None:  # 进行图片数据转换
            img = self.do_transform(img)
        return img, img_class

    def __len__(self):
        return len(self.data_set)

    def pre_treat(self, img, img_rotate_angle):
        img_resize = cv2.resize(img, (self.img_size, self.img_size))  # 图片resize
        img_rotate = self.rotate_img(img_resize, img_rotate_angle)  # 图片旋转
        return img_rotate

    def rotate_img(self, img, img_rotate_angle):
        border = self.border
        # 边界填充，避免旋转后图片出现黑色区域
        border_img = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_REPLICATE)
        h, _, _ = border_img.shape
        rotate_m = cv2.getRotationMatrix2D((h / 2, h / 2), img_rotate_angle, 1)  # 计算旋转矩阵
        img_rotate = cv2.warpAffine(border_img, rotate_m, (h, h))  # 进行旋转，并输出h * h大小图片
        half_size = self.img_size / 2  # 图片一半边长值，供对旋转后图片以中心为起点进行剪裁
        return img_rotate[int(h / 2 - half_size): int(h / 2 + half_size),
               int(h / 2 - half_size): int(h / 2 + half_size), :]

    @staticmethod
    def analyze_trainset(train_set):  # 统计训练集各类训练图片数量
        class_id = []
        class_count = []
        for _i in train_set:
            if _i['class_id'] not in class_id:
                class_id.append(_i['class_id'])
                class_count.append(1)
            else:
                _index = class_id.index(_i['class_id'])
                class_count[_index] += 1
        return class_id, class_count

    @staticmethod
    def show_img(img, img_path, info):  # 按需可视化中间结果
        img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_cvt)
        plt.title(img_path + "_" + info)
        plt.show()


class ConvBnLeakyrelu(nn.Module):  # conv + bn + lenkyrelu 基本块
    def __init__(self, in_chnls, out_chnls, ksize=3):
        super(ConvBnLeakyrelu, self).__init__()
        self.conv = nn.Conv2d(in_chnls, out_chnls, kernel_size=ksize, padding=1)
        self.bn = nn.BatchNorm2d(out_chnls)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leakyrelu(x)
        return x


class Darknet19(nn.Module):  # darknet19，参考yolov2
    def __init__(self):
        super(Darknet19, self).__init__()
        self.layer_1 = ConvBnLeakyrelu(3, 32)  # 224 * 224
        self.maxpool = nn.MaxPool2d(2, 2)  # 112 * 112
        self.layer_2 = ConvBnLeakyrelu(32, 64)  # 112 * 112
        self.layer_3 = self.bottleneck_3(64, 128)  # 56 * 56
        self.layer_4 = self.bottleneck_3(128, 256)  # 28 * 28
        self.layer_5 = self.bottleneck_5(256, 512)  # 14 * 14
        self.layer_6 = self.bottleneck_5(512, 1024)  # 7 * 7
        self.layer_7 = nn.Conv2d(1024, 62, 1, stride=1)  # 7 * 7
        self.avgpool = nn.AvgPool2d(7)  # 1 * 1
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():  # 模型参数初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.bias.data.fill_(0)  # 偏差初始为零
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer_1(x)  # 224 * 224
        x = self.maxpool(x)  # 112 * 112
        x = self.layer_2(x)  # 112 * 112
        x = self.maxpool(x)  # 56 * 56
        x = self.layer_3(x)  # 56 * 56
        x = self.maxpool(x)  # 28 * 28
        x = self.layer_4(x)  # 28 * 28
        x = self.maxpool(x)  # 14 * 14
        x = self.layer_5(x)  # 14 * 14
        x = self.maxpool(x)  # 7 * 7
        x = self.layer_6(x)  # 7 * 7
        x = self.layer_7(x)  # 7 * 7
        x = self.avgpool(x)  # 1 * 1
        x = x.view(x.size(0), -1)  # 均值池化后结果进行压平
        x = self.softmax(x)  # 输出softmax分类结果
        return x

    @staticmethod
    def bottleneck_3(in_chnls, out_chnls):  # 三层瓶颈块
        return nn.Sequential(ConvBnLeakyrelu(in_chnls, out_chnls),
                             ConvBnLeakyrelu(out_chnls, int(out_chnls / 2), ksize=1),
                             ConvBnLeakyrelu(int(out_chnls / 2), out_chnls))

    @staticmethod
    def bottleneck_5(in_chnls, out_chnls):  # 五层瓶颈块
        return nn.Sequential(ConvBnLeakyrelu(in_chnls, out_chnls),
                             ConvBnLeakyrelu(out_chnls, int(out_chnls / 2), ksize=1),
                             ConvBnLeakyrelu(int(out_chnls / 2), out_chnls),
                             ConvBnLeakyrelu(out_chnls, int(out_chnls / 2), ksize=1),
                             ConvBnLeakyrelu(int(out_chnls / 2), out_chnls))


class MultiWorks:
    def __init__(self, model_path=None):
        self.start_time = time.time()  # 开始时间，用于打印用时
        self.data_transform = transforms.Compose([transforms.ToTensor()])  # 数据转换，将图片数据转换为tensor
        self.model_path = model_path  # 模型路径，测试、微调、预测任务时需输入

        if not os.path.exists(args.save_directory):  # 新建保存文件夹
            os.makedirs(args.save_directory)

        work = args.work  # 根据输入work对应启动任务
        if work not in ['train', 'test', 'finetune', 'predict']:
            print("args.work should be one of ['train', 'test', 'finetune', 'predict']")
        elif work == 'train':
            self.train()
        elif self.model_path is None:  # 测试、微调、预测任务时需输入模型路径
            print('Please input model_path')
        elif args.work == "finetune":  # 调模型
            self.finetune()
        elif args.work == "test":  # 测试
            precision, recall, mean_precision, mean_recall, mean_loss, max_loss, min_loss = \
                self.test(self.model_path, is_path=True)
            print(f"mean_loss: {mean_loss}  max_loss: {max_loss}  min_loss: {min_loss}\n"
                  f"--precision:\n{precision}\n--recall:\n{recall}\n"
                  f"mean_precision: {mean_precision}  mean_recall: {mean_recall} "
                  f"cost_time: {time.time() - self.start_time}")
            collect_loss = [['epoch_mean_loss', 'epoch_max_loss', 'epoch_min_loss', 'mean_precision', 'mean_recall']]
            precision_head = ['precision' + str(i+1) for i in range(args.class_num)]
            recall_head = ['recall' + str(i+1) for i in range(args.class_num)]
            collect_loss[0].extend(precision_head), collect_loss[0].extend(recall_head)
            precision_list = [i.item() for i in precision]  # 得到各类精度列表形式
            recall_list = [i.item() for i in recall]  # 得到各类召回率列表形式
            info = [mean_loss, max_loss, min_loss, mean_precision, mean_recall]
            info.extend(precision_list), info.extend(recall_list)
            save_loss_path = self.model_path[:-3] + '_test_result.csv'
            self.writelist2csv(collect_loss, save_loss_path)
            print(f'--Save complete!\n--save_loss_path: {save_loss_path}\n')
            print('Test complete!')
        elif work == 'predict':
            self.predict()

    def train(self):
        data_set = DataSet(do_transform=self.data_transform)  # 返回数据集
        class_weight = data_set.class_weight  # 返回类权重
        load_data = DataLoader(data_set, batch_size=args.batch_size, shuffle=True)  # 数据加载器
        print(f"Start Train!  data_set_len: {data_set.__len__()}")  # 打印任务开始及数据集大小
        model = Darknet19().to(device)  # 加载darknet19模型

        criterion = nn.CrossEntropyLoss(torch.tensor(class_weight).to(device))  # 评价器，并输入分类权重
        current_lr = args.lr  # 步长
        optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=args.momentum)  # SGD优化器

        # 采集loss并在最后输出.csv文件
        collect_loss = [['epoch', 'lr', 'epoch_mean_loss', 'epoch_max_loss', 'epoch_min_loss',
                         'precision', 'recall', 'mean_precision', 'mean_recall']]
        epoch_count = []
        loss_record = []
        pre_rec = []
        cost_time_record = []
        for i in range(args.epochs):
            epoch_loss = []  # 每轮loss
            cls_tp, cls_tn, cls_fp, cls_fn = torch.zeros(62) + 0.000001, torch.zeros(62), torch.zeros(62), torch.zeros(
                62)
            for index, (img, label) in enumerate(load_data):
                img = img.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                output = model(img)
                loss = criterion(output, label)
                epoch_loss.append(loss.item())
                loss.backward()  # loss值对模型内参数进行反向传播
                optimizer.step()  # 参数更新

                class_result = torch.argmax(output, dim=1)  # 返回分类结果
                for c in range(args.class_num):  # 对每一类均统计tp、tn、fp、fn
                    cls_tp[c] += torch.sum(class_result[label == c] == c)
                    cls_tn[c] += torch.sum(class_result[label != c] != c)
                    cls_fp[c] += torch.sum(class_result[label != c] == c)
                    cls_fn[c] += torch.sum(class_result[label == c] != c)
            epoch_mean_loss = sum(epoch_loss) / (len(epoch_loss))
            epoch_max_loss = max(epoch_loss)
            epoch_min_loss = min(epoch_loss)
            precision, recall = cls_tp / (cls_tp + cls_fp), cls_tp / (cls_tp + cls_fn)  # 得到各类的精度、召回率
            mean_precision, mean_recall = torch.mean(precision), torch.mean(recall)  # 得到所有类的平均精度和召回率
            _, _, test_mean_precision, test_mean_recall, mean_loss, max_loss, min_loss = self.test(model.state_dict())
            # 供visdom显示
            epoch_count.append(i + 1)
            loss_record.append([epoch_mean_loss, epoch_max_loss, epoch_min_loss, mean_loss, max_loss, min_loss])
            pre_rec.append([mean_precision, mean_recall, test_mean_precision, test_mean_recall])
            cost_time_record.append(time.time() - self.start_time)
            vis.line(X=epoch_count, Y=loss_record, win='chart1', opts=opts1)
            vis.line(X=epoch_count, Y=pre_rec, win='chart2', opts=opts2)
            vis.line(X=epoch_count, Y=cost_time_record, win='chart3', opts=opts3)
            # 采集loss
            collect_loss.append([i, current_lr, epoch_mean_loss, epoch_max_loss, epoch_min_loss,
                                 precision, recall, mean_precision, mean_recall])
        # 模型保存路径
        save_model_path = os.path.join(args.save_directory, time.strftime('%Y%m%d%H%M') + '_train_epoch_' + str(i)
                                       + ".pt")
        # 训练过程保存路径
        save_loss_path = os.path.join(args.save_directory, time.strftime('%Y%m%d%H%M') + '_train_loss.csv')
        torch.save(model.state_dict(), save_model_path)
        self.writelist2csv(collect_loss, save_loss_path)  # 写入.csv文件
        print(f'--Save complete!\n--save_model_path: {save_model_path}\n--save_loss_path: {save_loss_path}')
        print('Train complete!')

    def test(self, input_model, is_path=False):
        data_set = DataSet(is_train=False, do_transform=self.data_transform)  # 返回数据集
        class_weight = data_set.class_weight  # 返回类权重
        load_data = DataLoader(data_set, batch_size=args.test_batch_size, shuffle=False)  # 数据加载器
        with torch.no_grad():  # 关闭梯度计算
            if is_path:
                print(f"Start Test!  len_dataset: {data_set.__len__()}")
            model = Darknet19().to(device)
            if is_path:  # 模型参数加载
                model.load_state_dict(torch.load(input_model))
            else:
                model.load_state_dict(input_model)
            model.eval()  # 关闭参数梯度

            criterion = nn.CrossEntropyLoss(torch.tensor(class_weight).to(device))  # 评价器，并输入分类权重

            epoch_loss = []  # 每轮loss
            cls_tp, cls_tn, cls_fp, cls_fn = torch.zeros(62) + 0.000001, torch.zeros(62), \
                                             torch.zeros(62), torch.zeros(62)
            for index, (img, label) in enumerate(load_data):
                img = img.to(device)
                label = label.to(device)
                output = model(img)
                loss = criterion(output, label)
                epoch_loss.append(loss.item())

                class_result = torch.argmax(output, dim=1)  # 返回分类结果
                for c in range(args.class_num):  # 对每一类均统计tp、tn、fp、fn
                    cls_tp[c] += torch.sum(class_result[label == c] == c)
                    cls_tn[c] += torch.sum(class_result[label != c] != c)
                    cls_fp[c] += torch.sum(class_result[label != c] == c)
                    cls_fn[c] += torch.sum(class_result[label == c] != c)
        precision, recall = cls_tp / (cls_tp + cls_fp), cls_tp / (cls_tp + cls_fn)  # 得到各类的精度、召回率
        mean_precision, mean_recall = torch.mean(precision).item(), torch.mean(recall).item()  # 所有类平均精度召回
        epoch_mean_loss = sum(epoch_loss) / (len(epoch_loss))
        epoch_max_loss = max(epoch_loss)
        epoch_min_loss = min(epoch_loss)
        return precision, recall, mean_precision, mean_recall, epoch_mean_loss, epoch_max_loss, epoch_min_loss

    def finetune(self):
        data_set = DataSet(do_transform=self.data_transform)  # 返回数据集
        class_weight = data_set.class_weight  # 返回类权重
        load_data = DataLoader(data_set, batch_size=args.batch_size, shuffle=True)  # 数据加载器
        print(f"Start Finetune!  len_data_set: {data_set.__len__()}")  # 打印任务开始及数据集大小

        model = Darknet19().to(device)
        model.load_state_dict(torch.load(self.model_path))  # 模型参数加载

        criterion = nn.CrossEntropyLoss(torch.tensor(class_weight).to(device))  # 评价器，并输入分类权重
        current_lr = args.lr  # 步长
        optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=args.momentum)  # SGD优化器
        # 采集loss并在最后输出.csv文件
        collect_loss = [['epoch', 'lr', 'epoch_mean_loss', 'epoch_max_loss', 'epoch_min_loss',
                         'precision', 'recall', 'mean_precision', 'mean_recall']]
        epoch_count = []
        loss_record = []
        pre_rec = []
        cost_time_record = []
        for i in range(args.epochs):
            epoch_loss = []  # 每轮loss
            cls_tp, cls_tn, cls_fp, cls_fn = torch.zeros(62) + 0.000001, torch.zeros(62), torch.zeros(62), torch.zeros(
                62)
            for index, (img, label) in enumerate(load_data):
                img = img.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                output = model(img)
                loss = criterion(output, label)
                epoch_loss.append(loss.item())
                loss.backward()  # loss值对模型内参数进行反向传播
                optimizer.step()  # 参数更新

                class_result = torch.argmax(output, dim=1)  # 返回分类结果
                for c in range(args.class_num):  # 对每一类均统计tp、tn、fp、fn
                    cls_tp[c] += torch.sum(class_result[label == c] == c)
                    cls_tn[c] += torch.sum(class_result[label != c] != c)
                    cls_fp[c] += torch.sum(class_result[label != c] == c)
                    cls_fn[c] += torch.sum(class_result[label == c] != c)

            epoch_mean_loss = sum(epoch_loss) / (len(epoch_loss))
            epoch_max_loss = max(epoch_loss)
            epoch_min_loss = min(epoch_loss)
            precision, recall = cls_tp / (cls_tp + cls_fp), cls_tp / (cls_tp + cls_fn)  # 得到各类的精度、召回率
            mean_precision, mean_recall = torch.mean(precision), torch.mean(recall)  # 得到所有类的平均精度和召回率
            _, _, test_mean_precision, test_mean_recall, mean_loss, max_loss, min_loss = self.test(model.state_dict())
            # 供visdom显示
            epoch_count.append(i + 1)
            loss_record.append([epoch_mean_loss, epoch_max_loss, epoch_min_loss, mean_loss, max_loss, min_loss])
            pre_rec.append([mean_precision, mean_recall, test_mean_precision, test_mean_recall])
            cost_time_record.append(time.time() - self.start_time)
            vis.line(X=epoch_count, Y=loss_record, win='chart1', opts=opts1)
            vis.line(X=epoch_count, Y=pre_rec, win='chart2', opts=opts2)
            vis.line(X=epoch_count, Y=cost_time_record, win='chart3', opts=opts3)
            # 采集loss
            collect_loss.append([i, current_lr, epoch_mean_loss, epoch_max_loss, epoch_min_loss,
                                 precision, recall, mean_precision, mean_recall])

        # 保存模型以及微调过程记录
        save_model_path = self.model_path[:-3] + '_finetune_' + str(i) + ".pt"
        save_loss_path = self.model_path[:-3] + '_finetune_' + str(i) + "_loss.csv"
        torch.save(model.state_dict(), save_model_path)
        self.writelist2csv(collect_loss, save_loss_path)
        print(f'--Save complete!\n--save_model_path: {save_model_path}\n--save_loss_path: {save_loss_path}')
        print('Finetune complete!')

    def predict(self):
        img_path = './traffic-sign/train/00006/00147_00000.png'
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = transforms.ToTensor()(img)
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img).to(device)

        print(f"Start Predict!")  # 打印任务开始
        with torch.no_grad():  # 关闭梯度计算
            model = Darknet19().to(device)
            model.load_state_dict(torch.load(self.model_path))  # 模型参数加载
            model.eval()  # 进行验证模式

            output = model(img)
            class_result = torch.argmax(output, dim=1)  # 返回分类结果
            print(f'--output: {output.data}\n--class_result: {class_result.item()}')

    @staticmethod
    def writelist2csv(list_data, csv_name):  # 列表写入.csv
        with open(csv_name, "w", newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for one_slice in list_data:
                csv_writer.writerow(one_slice)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current Model')
    parser.add_argument('--save-directory', type=str, default='save_model',
                        help='learnt models are saving here')
    parser.add_argument('--class-num', type=int, default=62,
                        help='class num')
    parser.add_argument('--work', type=str, default='test',  # train, eval, finetune, predict
                        help='training, eval, predicting or finetuning')
    args = parser.parse_args()

    # visdom可视化设置
    vis = Visdom(env="traffic-sign-class-1")
    assert vis.check_connection()
    opts1 = {
        "title": 'loss of mean/max/min in epoch',
        "xlabel": 'epoch',
        "ylabel": 'loss',
        "width": 600,
        "height": 400,
        "legend": ['train_mean_loss', 'train_max_loss', 'train_min_loss', 'test_mean_loss', 'test_max_loss',
                   'test_min_loss']
    }
    opts2 = {
        "title": 'precision recall with epoch',
        "xlabel": 'epoch',
        "ylabel": 'precision/recall in percentage',
        "width": 400,
        "height": 300,
        "legend": ['train_precision', 'train_recall', 'test_precision', 'test_recall']
    }
    opts3 = {
        "title": 'cost time with epoch',
        "xlabel": 'epoch',
        "ylabel": 'time in second',
        "width": 400,
        "height": 300,
        "legend": ['cost time']
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    MultiWorks(model_path='save_model/202004171744_train_epoch_9_finetune_9_finetune_9_finetune_2.pt')
