import numpy as np
from all_param import *
 
def word2vec_index(file_path):
    """
    :param file_path: 词向量文件路径
    :return word2vector: 字到向量的字典
    :return word2index: 字到词袋表示的字典
    :return index2word: 词袋表示到字的字典
    """
    word2vector = {}
    word2index = {}
    index2word = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        index = 1
        data = file.readlines()[1:]
        for line in data:
            line = line.replace('\n', '')
            line = line.split(' ')
            word = line[0]
            vector = np.array(line[1:], dtype=float)
            #建立索引
            word2vector[word] = vector
            word2index[word] = index
            index2word[index] = word
            index +=1
 
        # 加入填充符
        word2vector['<pad>'] = np.zeros(shape=(word2vec_size))
        word2index['<pad>'] = 0
        index2word[0] = '<pad>'
 
    return word2vector, word2index, index2word
 
 
def data_processing(path, data_len, word2vector, word2index, data_batch, data_start_site):
    """
    :param path: 数据集路径
    :param data_len: 数据数
    :param word2vector: 转词向量字典
    :param word2index: 转词词袋表示字典
    :param data_batch: 一次取的数据数
    :param data_start_site: 开始取的数据位置
    :return comment2vector: 评论向量表示
    :return comment2index: 评论词袋表示
    :return labels: 标签（独热编码）
    """
    with open(path, 'r', encoding='utf-8') as file1:
        data = file1.readlines()
        if data_start_site + data_batch > data_len: # 选取数据下标超出列表的长度但小于所取的数据批数时
            end_site = data_start_site + data_batch - data_len   # 应取数据的末尾位置
            data = data[data_start_site:] + data[:end_site]
        else:
            end_site = data_start_site + data_batch       # 应取数据的末尾位置
            data = data[data_start_site:end_site]
    file1.close()
    #初始化向量空间和词袋空间
    comment2vector = np.zeros(shape=(len(data), max_len, word2vec_size))
    comment2index = np.zeros(shape=(len(data), max_len))
    labels = np.zeros(shape=(len(data), class_num), dtype=float)
    #遍历每一条评论
    for i in range(len(data)):
        comment = data[i][2:]   # 获取评论
        comment = comment.replace('\n', '')
        comment = comment.split(' ')
        comment = [i for i in comment if i !='']    # 去除列表里所有空元素
 
        for word in range(max_len):    #对评论进行数值转换
            if word > len(comment) - 1:                        #评论长度短需要填充时
                continue
            else:                                           #正常数值转换时
                comment2vector[i][word] = word2vector[comment[word]]   #向量转换
                comment2index[i][word] = word2index[comment[word]]     #词袋转换
 
        label = int(data[i][:1])  # 获取标签
        # 独热编码
        labels[i][label] = 1
 
        # 标签平滑
        for zero in range(len(labels[i])):
            if labels[i][zero] == 0:
                labels[i][zero] = 0.0000001
            else:
                labels[i][zero] = 0.9999999
    return comment2vector, comment2index, labels
 
 
 
 
 
if __name__ == '__main__':
    word2vector, word2index, index2word = word2vec_index(
        'word2vec/douban_comment/fen_ci128/balanced/balanced_data.vector')  # 加载词向量
    # 获取数据集个数
    with open('data_set/douban_comment/balanced/balanced_train.txt', 'r', encoding='utf-8') as file1:
        datas_len = len(file1.readlines())
    file1.close()
    print('一共有{}条数据'.format(datas_len))
 
    # 分批次输入数据集
    #batch_num = datas_len // batch_size  # 可分的批次数
    batch_num = 1
    for i in range(batch_num+1):
        comment_vector, comment_index, labels = data_processing(
            'data_set/douban_comment/balanced/balanced_train.txt', datas_len,word2vector, word2index, batch_size, i * batch_size)
        print(labels)
 