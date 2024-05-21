word2vec_size = 768         # 词向量维度
 
max_len = 250               # 最大句子长度
 
batch_size = 16             # 一次训练批数
 
head_num = 8                # 多头个数, 必须小于词向量维度，（head_dim=word2vec_size//head_num）
 
transformer_layer = 1               # 编码器（解码器）层数
 
class_num = 2               # 分类的类别数
 
learning_rate = 1e-5       # 学习率
 
steps = 10                   # 训练次数
 
Train = True           # 是否选择训练模式，True为训练模式， False为预测模式
 
cnn_layer = 3         # CNN层数
 
kernel_num = 32         # 卷积核个数