import os
import time
import numpy as np
import get_data
from all_param import *
import bert_torch
import torch
import math
#from transformers import BertTokenizer, BertModel
 
class TextCNN(torch.nn.Module):
    def __init__(self,embed_dim,kernel_num,cnn_layer,learning_rate,class_num,DEVICE):
        super(TextCNN, self).__init__()
        # 初始化第一层卷积核大小分别为（2，embed_dim），（3，embed_dim），（4，embed_dim）的卷积层
        self.conv = [torch.nn.Conv2d(1,kernel_num,(i,embed_dim)).to(DEVICE) for i in range(2,5)]
 
        self.relu = torch.nn.ReLU()
        self.max_pool = torch.nn.MaxPool1d(2,ceil_mode=True)  # 最大池化层
        self.drop = torch.nn.Dropout(learning_rate)
 
 
        # 后续的深层卷积层
        if cnn_layer>1:
            self.conv_add = [torch.nn.Conv1d(int(math.pow(2, i)) * kernel_num,
                                          2 *int(math.pow(2, i)) * kernel_num, 2).to(DEVICE) for i in range(cnn_layer-1)]
 
        # 根据矩阵变化的规律求出最后得到全连接前的矩阵[batch_size,line_dim]里的dim
        line_dim = max_len / 2                      # 由第一层池化操作得到的
        if cnn_layer > 1:
            for i in range(cnn_layer - 1):          # 第二层到第cnn_layer层
                if i%2==0:                          # 偶数层刚好卷积后全部池化
                    line_dim = int((line_dim - 1) / 2)
                if i%2==1:                          # 奇数层卷积后会剩一个没池化到，便多池化一次
                    line_dim = int((line_dim - 1) / 2) + 1
            line_dim = int(math.pow(2, cnn_layer - 1)) * kernel_num * line_dim  # 乘上卷积核个
        # 初始化全连接层
        self.line = torch.nn.Linear(line_dim * 3, class_num)
 
    # 一个cnn结构
    def conv_and_pool(self,input,conv):
        """
        :param input: 输入数据
        :param conv: 卷积层
        :return:
        """
        data = conv(input)          # 卷积  [batch,kernel_num,max_len,1]
        data = data.squeeze(3)      # 降维   [batch,kernel_num,max_len]
        data = self.relu(data)      # relu激活函数
        data = self.max_pool(data)  # 池化    [batch,kernel_num,max_len/2]
        #print(data.shape)
        if cnn_layer>1: # 进入深度卷积层
            for this_layer in range(len(self.conv_add)):  # 例如第二层卷积数据形状
                data = self.conv_add[this_layer](data)  # 卷积  [batch, kernel_num*2, max_len/2-1]
                data = self.relu(data)                 # relu激活函数[batch, kernel_num*2, max_len/2-1]
                data = self.max_pool(data)             # 池化 [batch, kernel_num*2, (max_len/2-1)/2]
                #print(data.shape)
 
        data = torch.reshape(data,shape=(data.shape[0],-1))   # 展开最后一维进行降维
 
        return data
 
    # 用上2，3，4这三个cnn
    def calls(self,input):
        """
        :param input: 输入数据
        :return:
        """
        datas = []
        # 获取三个cnn的结果
        for i in range(len(self.conv)):
            data = self.conv_and_pool(input,self.conv[i])
            datas.append(data)
        # 将结果进行拼接
        for i in range(1,len(datas)):
            datas[0] = torch.cat((datas[0],datas[i]),dim=1)
 
        datas = self.drop(datas[0])    # 防止过拟合
        output = self.line(datas)       # 全连接
 
        return output
 
 
class mymodel(torch.nn.Module):
    def __init__(self, embed_dim, kernel_num, cnn_layer, learning_rate, class_num, Train, DEVICE):
        super(mymodel, self).__init__()
        self.bert = bert_torch.bert(class_num)
        self.cnn = TextCNN(embed_dim,kernel_num,cnn_layer,learning_rate,class_num,DEVICE)
        # none表示不降维，返回和target相同形状；mean表示对一个batch的损失求均值；sum表示对一个batch的损失求和
        self.cross_entropy = torch.nn.CrossEntropyLoss()    # 定义损失函数，交叉熵损失函数
        self.optimizer = torch.optim.Adam(self.parameters(),lr=learning_rate)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)  # 设置学习率下降策略"
 
        #self.drop = torch.nn.Dropout(learning_rate)
        ## 根据矩阵变化的规律求出最后得到全连接前的矩阵[batch_size,line_dim]里的dim
        #line_dim = max_len / 2  # 由第一层池化操作得到的
        #if cnn_layer > 1:
        #    for i in range(cnn_layer - 1):  # 第二层到第cnn_layer层
        #        if i % 2 == 0:  # 偶数层刚好卷积后全部池化
        #            line_dim = int((line_dim - 1) / 2)
        #        if i % 2 == 1:  # 奇数层卷积后会剩一个没池化到，便多池化一次
        #            line_dim = int((line_dim - 1) / 2) + 1
        #    line_dim = int(math.pow(2, cnn_layer - 1)) * kernel_num * line_dim  # 乘上卷积核个数
#
        ## 初始化全连接层
        #self.line = torch.nn.Linear(line_dim * 3, class_num)
#
        self.writing_mode = 'w'
        self.Train = Train
 
    def Training(self, data_path, verify_path, max_len, DEVICE, epoch, batch_size, class_num, save_path):
 
        self.train()
        # 获取数据总数
        with open(data_path, 'r', encoding='utf-8') as file1:
            datas_len = len(file1.readlines())
        file1.close()
        print('一共有{}条数据'.format(datas_len))
 
        # 训练
        all_time_start = time.time()
        bast_acc = 0
        for e in range(epoch):
            this_time_start = time.time()  # 起始时间
            batch_num = datas_len // batch_size  # 可取的批数
            batch_num = 10
 
            all_loss = []
            all_outputs = torch.tensor(np.zeros(shape=(1, class_num)), dtype=torch.float32)
            all_labels = torch.tensor(np.zeros(shape=1), dtype=torch.float32)
 
            # 批训练
            for batch in range(batch_num):
                # 获取数据
                labels, comments = get_data.get_input(data_path, datas_len, batch_size, batch)
                long_labels = torch.tensor(labels, dtype=torch.float32).long()
 
                self.optimizer.zero_grad()  # 1.梯度置零
                outputs, _ = self.bert.calls(comments, max_len, DEVICE)  # 2.模型获得结果
                outputs = outputs.unsqueeze(1)
                #print(outputs.shape)
                outputs = self.cnn.calls(outputs)
                #cls = self.drop(cls)  # 防止过拟合
                #cls = self.line(cls)  # 全连接
                #outputs = torch.softmax(outputs, dim=-1)
 
                loss = self.cross_entropy(outputs.to('cpu'), long_labels)  # 3.计算损失
                #loss.requires_grad_(True)
                loss.backward()  # 4.反向传播
                self.optimizer.step()  # 5.修改参数，w，b
 
                ## 记录遍历一遍数据的总结果
                all_loss.append(loss.item())  # item()返回loss的值
                all_outputs = torch.cat((all_outputs, outputs.to('cpu')), dim=0)
                for i in range(len(labels)):
                    if labels[i] == 0.001:
                        labels[i] = 0
                    else:
                        labels[i] = 1
                labels = torch.tensor(labels, dtype=torch.float32)
                all_labels = torch.cat((all_labels, labels), dim=0)
 
                ## 选择训练最好的参数保存
                #Acc = self.test(verify_path, batch_size, class_num, save_path)
                #if Acc > bast_acc:
                #   bast_acc = Acc
                #   # 保存模型
                #   torch.save(self.state_dict(), save_path + "/model.pth")
                print('\r训练进度{:2d}%， 共有{}批数据， 已完成{:2d}%， 当前损失： {:4f}， ACC: {} '.format(
                        int((e) / epoch * 100), batch_num, int((batch + 1) / batch_num * 100),loss, 'None'), end='')
            # 打印并保存本次训练结果
            if e % 1 == 0:
                torch.save(self,save_path + "/model.pth")
                this_time = time.time() - this_time_start  # 本次耗时
                all_time = time.time() - all_time_start  # 当前总耗时
                predict_value = np.argmax(all_outputs[1:].detach().numpy(), axis=-1)[:, None]  # 预测标签（0或1）
                actual_value = all_labels[1:].detach().numpy()[:, None]  # 实际标签
                result = np.concatenate((predict_value, actual_value), axis=1)  # 标签拼接对比[预测，实际]
                mean_loss = np.array(all_loss).mean()
 
                acc = self.look_and_save_data(result, this_time, save_path, self.writing_mode, self.Train, step=e,
                                   loss=mean_loss, all_time=all_time)
                self.writing_mode = 'a'  # 更改写入模式为追加
 
 
    def test(self, data_path, batch_size, class_num, save_path, test_data_save=False):
 
        self.eval()
        # 获取数据总数
        with open(data_path, 'r', encoding='utf-8') as file1:
            datas_len = len(file1.readlines())
        file1.close()
        print('一共有{}条数据'.format(datas_len))
 
        this_time_start = time.time()  # 起始时间
        batch_num = datas_len // batch_size  # 可取的批数
        all_outputs = torch.tensor(np.zeros(shape=(1, class_num)), dtype=torch.float32)
        all_labels = torch.tensor(np.zeros(shape=1), dtype=torch.float32)
        batch_num = 30
 
        # 批训练
        for batch in range(batch_num):
            # 获取数据
            labels, comments = get_data.get_input(data_path, datas_len, batch_size, batch)
            labels = torch.tensor(labels, dtype=torch.float32)
 
            with torch.no_grad():  # 不进行梯度计算，节省内存
                outputs, _ = self.bert.calls(comments, max_len, DEVICE)  # 2.模型获得结果
                outputs = self.cnn.calls(outputs.unsqueeze(1))
                #cls = self.drop(cls)  # 防止过拟合
                #cls = self.line(cls)  # 全连接
                #outputs = torch.softmax(outputs, dim=-1)
 
            # 记录遍历一遍数据的总结果
            all_outputs = torch.cat((all_outputs, outputs.to('cpu')), dim=0)
            for i in range(len(labels)):
                if labels[i] == 0.001:
                    labels[i] = 0
                else:
                    labels[i] = 1
            labels = torch.tensor(labels, dtype=torch.float32)
            all_labels = torch.cat((all_labels, labels), dim=0)
            if test_data_save != False:
                print('\r共有{}批数据， 测试进度{:2d}% '.format(batch_num, int((batch + 1) / batch_num * 100)), end='')
 
        this_time = time.time() - this_time_start  # 本次耗时
        all_outputs = np.argmax(all_outputs[1:].detach().numpy(), axis=-1)[:, None]  # 预测标签（0或1）
        all_labels = all_labels[1:].detach().numpy()[:, None]  # 实际标签
        all_outputs = np.concatenate((all_outputs, all_labels), axis=1)  # 标签拼接对比[预测，实际]
        # 计算评价指标并保存训练情况
        Acc = self.look_and_save_data(all_outputs, this_time, save_path, self.writing_mode, test_data_save=test_data_save)
 
        return Acc
 
 
    # 打印和保存训练过程或预测结果
    def look_and_save_data(self, result, this_time, save_path, writing_mode, Train=False, step=None, loss=None,
                           all_time=None, test_data_save=False):
        # 计算P、R、F1、Accuracy
        TP = len([i for i in result if i.sum() == 2])
        TN = len([i for i in result if i.sum() == 0])
        FP = len([i for i in result if (i[0] - i[1]) == 1])
        FN = len([i for i in result if (i[0] - i[1]) == -1])
        P = (TP + 0.0001) / (TP + FP + 0.0001)
        R = (TP + 0.0001) / (TP + FN + 0.0001)
        F1 = (2 * P * R + 0.00001) / (P + R + 0.00001)
        Accuracy = (TP + TN) / len(result)
        # 输出并保存结果
        if Train == True:  # 训练模式
            # 打印并保存训练过程
            print("\tstep: {:3}  |  mean_loss: {:3f}  |  time: {:3f}m  |  train_data_Acc: {:3f}  |".format(
                step, loss, this_time / 60, Accuracy))
            # 保存训练过程的数据
            with open(save_path + '/train_process.txt', writing_mode, encoding='utf-8') as file:
                file.write(
                    "step: {:3} | mean_loss: {:3f} | time: {:3f}m | P: {:3f} | R: {:3f} | F1: {:3f} | train_data_Acc: {:3f} |\n".format(
                        step, loss, all_time / 60, P, R, F1, Accuracy))
            file.close()
        ## 保存模型
        # torch.save(model.state_dict(), save_path+"/model.pth")
        else:  # 预测模式
            if test_data_save == True:
                print("P: {:3f} | R: {:3f} | F1: {:3f} | Accuracy: {:3f} | time: {:3f}m |\n".format(
                    P, R, F1, Accuracy, this_time / 60))
                with open(save_path + '/test_result.txt', writing_mode, encoding='utf-8') as file:
                    file.write("P: {:3f} | R: {:3f} | F1: {:3f} | Accuracy: {:3f} | time: {:3f}m |\n".format(
                        P, R, F1, Accuracy, this_time / 60))
                file.close()
        return Accuracy
 
 
 
 
if __name__ == '__main__':
    #tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')  # 加载base模型的对应的切词器
    #model = BertModel.from_pretrained('bert-base-chinese')
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('GPU: ', '可用' if str(DEVICE) == "cuda" else "不可用")     # 查看GPU是否可用
    print('torch版本: ', torch.__version__)                    # 查看torch版本
    print('GPU数量: ', torch.cuda.device_count())              # 查看GPU数量
    print('GPU索引号: ', torch.cuda.current_device() if torch.cuda.is_available() else 'N/A')          # 查看GPU索引号
    print('GPU名称: ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')          # 根据索引号得到GPU名称
 
    # 获取数据集个数
    save_path = 'model_data/balanced_bert_output_CNN_in_50_3_label'
    os.makedirs(save_path, exist_ok=True)  # 创建保存文件目录
    train_path = 'data_set/douban_comment/balanced/balanced_train.txt'
    test_path = 'data_set/douban_comment/balanced/balanced_test.txt'
    # verify_path = 'data_set/douban_comment/balanced/balanced_verify.txt'
    if Train == True:
        model = mymodel(word2vec_size, kernel_num, cnn_layer, learning_rate, class_num, Train, DEVICE).to(DEVICE)
        model.Training(train_path, verify_path, max_len, DEVICE, steps, batch_size, class_num, save_path)
 
        # 自行测试
        Train = False
        model.test(test_path, batch_size, class_num, save_path, test_data_save=True)
        model.test(test_path, batch_size, class_num, save_path, test_data_save=True)
    else:
        model = torch.load(save_path + "/model.pth")  # 加载模型参数
        model.test(test_path, batch_size, class_num, save_path, test_data_save=True)