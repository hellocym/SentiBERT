from transformers import BertModel, BertTokenizer
import torch
 
#print(torch.cuda.is_available())        # 查看GPU是否可用
#print(torch.cuda.device_count())        # 查看GPU数量
#print(torch.cuda.current_device())      # 查看GPU索引号
#print(torch.cuda.get_device_name(0))    # 根据索引号得到GPU名称
 
 
class bert(torch.nn.Module):
    def __init__(self):
        super(bert, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm')  # Bert分词器
        self.BERT = BertModel.from_pretrained('hfl/chinese-bert-wwm') # Bert模型,放GPU上
 
    def calls(self,input_list):
        batch_tokenized = self.tokenizer.batch_encode_plus(input_list, add_special_tokens=True,
                                                           max_length=max_len, padding='max_length',
                                                           truncation=True)
 
        input_ids = torch.tensor(batch_tokenized['input_ids'])
        attention_mask = torch.tensor(batch_tokenized['attention_mask'])
        #with torch.no_grad():
        hidden_outputs = self.BERT(input_ids, attention_mask=attention_mask)
        outputs = hidden_outputs[0]  # [0]表示输出结果(last_hidden_state部分)，[:,0,:]表示[CLS]对应的结果
        cls = outputs[:, 0, :]
        return outputs, cls
 
 
if __name__ == '__main__':
    import get_data
    import numpy as np
    import os
    import time
    from  all_param import *
 
    def train(BERT, data_path, epoch, batch_size, class_num, optimizer, line, cross_entropy, save_path, writing_mode, Train):
        # 获取数据总数
        with open(data_path, 'r', encoding='utf-8') as file1:
            datas_len = len(file1.readlines())
        file1.close()
        print('一共有{}条数据'.format(datas_len))
 
        # 训练
        all_time_start = time.time()
        torch.cuda.empty_cache()
        for e in range(epoch):
            this_time_start = time.time()  # 起始时间
            batch_num = datas_len // batch_size  # 可取的批数
            batch_num = 2
 
            all_loss = []
            all_outputs = torch.tensor(np.zeros(shape=(1, class_num)), dtype=torch.float32)
            all_labels = torch.tensor(np.zeros(shape=(1)), dtype=torch.float32)
 
            # 批训练
            for batch in range(batch_num):
                # 获取数据
                labels, comments = get_data.get_input(data_path, datas_len, batch_size, batch)
 
                labels = torch.tensor(labels, dtype=torch.float32).long()
 
                optimizer.zero_grad()  # 1.梯度置零
                _, cls = BERT.calls(comments)  # 2.模型获得结果
                cls = line(cls)
                #cls = torch.softmax(cls, dim=-1)
                loss = cross_entropy(cls, labels)  # 3.计算损失
 
                loss.requires_grad_(True)
                loss.backward()  # 4.反向传播
                optimizer.step()  # 5.修改参数，w，b
                print('\r共有{}批数据，第 {:3} 批数据,当前损失： {:4f} '.format(batch_num, batch, loss), end='')
 
                ## 记录遍历一遍数据的总结果
                all_loss.append(loss.item())  # item()返回loss的值
                all_outputs = torch.cat((all_outputs, cls), dim=0)
                all_labels = torch.cat((all_labels, labels), dim=0)
 
            # 打印并保存本次训练结果
            if e % 1 == 0:
                this_time = time.time() - this_time_start  # 本次耗时
                all_time = time.time() - all_time_start   # 当前总耗时
                predict_value = np.argmax(all_outputs[1:].detach().numpy(), axis=-1)[:, None]  # 预测标签（0或1）
                actual_value = all_labels[1:].detach().numpy()[:, None]  # 实际标签
                result = np.concatenate((predict_value, actual_value), axis=1)  # 标签拼接对比[预测，实际]
                look_and_save_data(BERT, result, this_time, save_path, writing_mode, Train, step=e,
                                        loss=np.array(all_loss).mean(), all_time=all_time)
                writing_mode = 'a'  # 更改写入模式为追加
 
    def test(BERT, data_path, batch_size, class_num, save_path, writing_mode, Train):
        # 获取数据总数
        with open(data_path, 'r', encoding='utf-8') as file1:
            datas_len = len(file1.readlines())
        file1.close()
        print('一共有{}条数据'.format(datas_len))
 
        BERT.load_state_dict(torch.load(save_path+"/model.ckpt"))
        BERT.eval()
        this_time_start = time.time()  # 起始时间
        batch_num = datas_len // batch_size  # 可取的批数
        all_outputs = torch.tensor(np.zeros(shape=(1, class_num)), dtype=torch.float32)
        all_labels = torch.tensor(np.zeros(shape=(1, class_num)), dtype=torch.float32)
 
        # 批训练
        for batch in range(batch_num):
            # 获取数据
            labels, comments = get_data.get_input(data_path, datas_len, batch_size, batch)
 
            labels = torch.tensor(labels, dtype=torch.float32)
 
            outputs, cls = BERT.call(comments)  # 2.模型获得结果
            cls = line(cls)
            cls = torch.softmax(cls, dim=-1)
 
            # 记录遍历一遍数据的总结果
            all_outputs = torch.cat((all_outputs, cls), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)
            print('\r共有{}批数据， 第 {:3} 批数据'.format(batch_num, batch+1), end='')
 
        this_time = time.time() - this_time_start  # 本次耗时
        predict_value = np.argmax(all_outputs[1:], axis=-1)[:, None]  # 预测标签（0或1）
        actual_value = np.argmax(all_labels[1:], axis=-1)[:, None]  # 实际标签
        result = np.concatenate((predict_value, actual_value), axis=1)  # 标签拼接对比[预测，实际]
        look_and_save_data(BERT, result, this_time, save_path, writing_mode, Train)
 
 
    # 打印和保存训练过程或预测结果
    def look_and_save_data(model, result, this_time, save_path,writing_mode, Train,
                           step=None, loss=None, all_time=None):
        # 计算P、R、F1、Accuracy
        TP = len([i for i in result if i.sum() == 2])
        TN = len([i for i in result if i.sum() == 0])
        FP = len([i for i in result if (i[0] - i[1]) == 1])
        FN = len([i for i in result if (i[0] - i[1]) == -1])
        P = (TP + 0.0001) / (TP + FP + 0.0001)
        R = (TP + 0.0001) / (TP + FN + 0.0001)
        F1 = (2 * P * R + 0.00001) / (P + R + 0.00001)
        Accuracy = (TP + TN) / len(result)
 
        os.makedirs(save_path, exist_ok=True)  # 创建文件目录
        # 输出并保存结果
        if Train == True:  # 训练模式
            # 打印并保存训练过程
            print("\tstep: {:3}  |  mean_loss: {:3f}  |  time: {:3f}m  |  Accuracy: {:3f}  |".format(
                step, loss, this_time / 60, Accuracy))
            # 保存训练过程的数据
            with open(save_path+'/train_process.txt', writing_mode, encoding='utf-8') as file:
                file.write(
                    "step: {:3} | mean_loss: {:3f} | time: {:3f}m | P: {:3f} | R: {:3f} | F1: {:3f} | Accuracy: {:3f} |\n".format(
                        step, loss, all_time / 60, P, R, F1, Accuracy))
            file.close()
 
            # 保存模型
            torch.save(model.state_dict(), save_path+"/model.ckpt")
 
        else:  # 预测模式
            print("P: {:3f} | R: {:3f} | F1: {:3f} | Accuracy: {:3f} | time: {:3f}m |\n".format(
                    P, R, F1, Accuracy, this_time / 60))
            with open(save_path+'/test_result.txt', writing_mode, encoding='utf-8') as file:
                file.write("P: {:3f} | R: {:3f} | F1: {:3f} | Accuracy: {:3f} | time: {:3f}m |\n".format(
                    P, R, F1, Accuracy, this_time / 60))
            file.close()
 
    # 初始化交叉熵和优化器
    bert = bert()
    line = torch.nn.Linear(768, class_num)
    cross_entropy = torch.nn.CrossEntropyLoss()  # 定义损失函数，交叉熵损失函数
    optimizer = torch.optim.Adam(bert.parameters(),lr=learning_rate)
    writing_mode = 'w'  # 初始写入模式为覆盖
    save_path = './model_data/cg'
    # 模型参数初始化
 
    if Train == True:  # 模型训练
        train(bert, 'data_set/douban_comment/balanced/balanced_train.txt', steps, batch_size, class_num,
              optimizer, line, cross_entropy, save_path, writing_mode, Train)
 
    else:  # 测试模型
        # 模型参数初始化
        test(bert, 'data_set/douban_comment/balanced/balanced_test.txt', batch_size, class_num,
             save_path, writing_mode, Train)
 
 
 