# -*- coding: utf-8 -*-
"""
lstm_text_generator_word_level.py

该模块实现了一个基于 NumPy 的词级别文本生成器，使用 LSTM 模型。
代码包含训练、模型保存、模型加载和测试功能，适用于教学和理解 LSTM 的工作原理。

Created on Mon Nov 25 11:17:34 2019
@author: lizhenping
"""

import numpy as np
import pickle
import os
import re
from collections import Counter

class Tokenizer:
    """
    词级别的分词器，负责将文本转换为词语索引序列，以及索引序列转换为文本。
    """
    def __init__(self, text, max_vocab_size=None):
        self.special_tokens = ['<PAD>', '<UNK>', '<EOS>']  # 特殊标记
        self.max_vocab_size = max_vocab_size
        self.build_vocab(text)
    
    def build_vocab(self, text):
        # 使用正则表达式分词，保留标点符号
        words = re.findall(r'\w+|[^\s\w]+', text)
        word_counts = Counter(words)
        
        if self.max_vocab_size:
            # 根据词频排序，取前 max_vocab_size 个词
            most_common = word_counts.most_common(self.max_vocab_size - len(self.special_tokens))
            vocab = self.special_tokens + [word for word, _ in most_common]
        else:
            vocab = self.special_tokens + list(word_counts.keys())
            
        self.word_to_ix = {word: i for i, word in enumerate(vocab)}
        self.ix_to_word = {i: word for i, word in enumerate(vocab)}
        self.vocab_size = len(vocab)

    def text_to_sequence(self, text):
        # 将文本转换为词语索引序列
        words = re.findall(r'\w+|[^\s\w]+', text)
        sequence = []
        for word in words:
            sequence.append(self.word_to_ix.get(word, self.word_to_ix['<UNK>']))
        sequence.append(self.word_to_ix['<EOS>'])
        return sequence
    
    def sequence_to_text(self, sequence):
        # 将词语索引序列转换为文本
        words = [self.ix_to_word.get(idx, '<UNK>') for idx in sequence]
        text = ''.join(words)  # 对于中文不需要空格分隔
        return text

class Module:
    """
    模块基类，所有神经网络模块的父类。
    提供参数管理、梯度清零、参数保存和加载功能。
    """
    def __init__(self):
        self.parameters = []
        self.gradients = []
        
    def zero_grad(self):
        """
        将所有参数的梯度清零。
        """
        for grad in self.gradients:
            grad.fill(0)
            
    def init_weights(self, init_range):
        """
        初始化模型权重，服从[-init_range, init_range]的均匀分布。
        """
        for param in self.parameters:
            param[:] = np.random.uniform(-init_range, init_range, size=param.shape)
            
    def apply_gradients(self, optimizer):
        """
        将优化器提供的梯度应用于模型参数。
        """
        for param, grad in zip(self.parameters, self.gradients):
            optimizer.apply_gradient(param, grad)

class LSTMCell(Module):
    """
    实现单个 LSTM 单元，包括前向和反向传播。
    """
    def __init__(self, input_size, hidden_size, init_range=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 初始化权重和偏置
        self.W_ih = np.random.uniform(-init_range, init_range, (4 * hidden_size, input_size))
        self.W_hh = np.random.uniform(-init_range, init_range, (4 * hidden_size, hidden_size))
        self.b_ih = np.zeros((4 * hidden_size, 1))
        self.b_hh = np.zeros((4 * hidden_size, 1))
        
        # 存储参数和对应的梯度
        self.parameters = [self.W_ih, self.W_hh, self.b_ih, self.b_hh]
        self.gradients = [np.zeros_like(param) for param in self.parameters]
        
    def forward(self, x, h_prev, c_prev):
        """
        前向传播

        参数:
        - x: 当前时间步的输入，形状为 (input_size, 1)
        - h_prev: 前一时间步的隐藏状态，形状为 (hidden_size, 1)
        - c_prev: 前一时间步的细胞状态，形状为 (hidden_size, 1)

        返回:
        - h_next: 当前时间步的隐藏状态
        - c_next: 当前时间步的细胞状态
        """
        self.x = x
        self.h_prev = h_prev
        self.c_prev = c_prev
        
        # 计算门和候选值
        gates = np.dot(self.W_ih, x) + self.b_ih + np.dot(self.W_hh, h_prev) + self.b_hh
        self.i_gate = sigmoid(gates[0:self.hidden_size])
        self.f_gate = sigmoid(gates[self.hidden_size:2*self.hidden_size])  
        self.o_gate = sigmoid(gates[2*self.hidden_size:3*self.hidden_size])
        self.g_gate = tanh(gates[3*self.hidden_size:4*self.hidden_size])
        
        # 计算细胞状态和隐藏状态
        self.c_next = self.f_gate * c_prev + self.i_gate * self.g_gate
        self.h_next = self.o_gate * tanh(self.c_next)
        
        return self.h_next, self.c_next
        
    def backward(self, dh_next, dc_next, grad_clip=5):
        """
        反向传播

        参数:
        - dh_next: 当前时间步的隐藏状态梯度
        - dc_next: 当前时间步的细胞状态梯度
        - grad_clip: 梯度裁剪阈值，默认为 5

        返回:
        - dx: 对输入 x 的梯度
        - dh_prev: 对前一隐藏状态 h_prev 的梯度
        - dc_prev: 对前一细胞状态 c_prev 的梯度
        """
        # 计算梯度
        do = dh_next * tanh(self.c_next) * dsigmoid(self.o_gate)
        dc = dh_next * self.o_gate * dtanh(tanh(self.c_next)) + dc_next
        df = dc * self.c_prev * dsigmoid(self.f_gate)  
        di = dc * self.g_gate * dsigmoid(self.i_gate)
        dg = dc * self.i_gate * dtanh(self.g_gate)
        
        # 拼接各个门的梯度
        d_gates = np.concatenate((di, df, do, dg), axis=0)
        
        # 计算权重和偏置的梯度
        dW_ih = np.dot(d_gates, self.x.T)
        dW_hh = np.dot(d_gates, self.h_prev.T) 
        db_ih = d_gates
        db_hh = d_gates
        
        # 裁剪梯度，避免梯度爆炸
        dW_ih = np.clip(dW_ih, -grad_clip, grad_clip)
        dW_hh = np.clip(dW_hh, -grad_clip, grad_clip)
        
        # 计算输入、前一隐藏状态和前一细胞状态的梯度
        dx = np.dot(self.W_ih.T, d_gates)
        dh_prev = np.dot(self.W_hh.T, d_gates)
        dc_prev = dc * self.f_gate
        
        # 存储梯度
        self.gradients[0] += dW_ih  
        self.gradients[1] += dW_hh
        self.gradients[2] += np.sum(db_ih, axis=1, keepdims=True)
        self.gradients[3] += np.sum(db_hh, axis=1, keepdims=True)
        
        return dx, dh_prev, dc_prev

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    # y = sigmoid(x)
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def dtanh(y):
    # y = tanh(x)
    return 1 - y ** 2

class LSTMLayer(Module):
    """
    LSTM 层，由多个 LSTM 单元组成。
    """
    def __init__(self, input_size, hidden_size, num_layers=1, init_range=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cells = [LSTMCell(input_size, hidden_size, init_range) if i == 0 
                           else LSTMCell(hidden_size, hidden_size, init_range) 
                           for i in range(num_layers)]
        
        # 存储参数和梯度
        self.parameters = []
        self.gradients = []
        for lstm_cell in self.lstm_cells:
            self.parameters.extend(lstm_cell.parameters)
            self.gradients.extend(lstm_cell.gradients)
            
    def forward(self, x, h0, c0):
        """
        前向传播

        参数:
        - x: 输入序列，形状为 (embedding_dim, seq_len)
        - h0: 初始隐藏状态，形状为 (hidden_size, batch_size)
        - c0: 初始细胞状态，形状为 (hidden_size, batch_size)

        返回:
        - outputs: 每个时间步的输出，形状为 (hidden_size, seq_len)
        - (hn, cn): 最后一个时间步的隐藏状态和细胞状态，形状均为 (hidden_size, batch_size)
        """
        seq_len = x.shape[1]
        hn = h0
        cn = c0 
        outputs = []
        
        for t in range(seq_len):
            xt = x[:, t:t+1]  # 当前时间步的输入
            for i, lstm_cell in enumerate(self.lstm_cells):
                hn, cn = lstm_cell.forward(xt, hn, cn)
                xt = hn  # 当前层的输出作为下一层的输入
            outputs.append(hn)
            
        outputs = np.concatenate(outputs, axis=1)  # 拼接各时间步的输出  
        return outputs, (hn, cn)
        
    def backward(self, doutputs, dhn, dcn, grad_clip=5): 
        """
        反向传播

        参数:
        - doutputs: 每个时间步的输出梯度，形状为 (hidden_size, seq_len)
        - dhn: 最后一个时间步的隐藏状态梯度，形状为 (hidden_size, 1)
        - dcn: 最后一个时间步的细胞状态梯度，形状为 (hidden_size, 1)
        - grad_clip: 梯度裁剪阈值，默认为 5

        返回:
        - dx: 输入序列的梯度，形状为 (embedding_dim, seq_len)
        """
        seq_len = doutputs.shape[1] 
        dh_next = dhn
        dc_next = dcn
        dx = []
        
        for t in reversed(range(seq_len)):
            dh = doutputs[:, t:t+1] + dh_next
            for i in reversed(range(self.num_layers)):
                lstm_cell = self.lstm_cells[i]
                dx_step, dh_prev, dc_prev = lstm_cell.backward(dh, dc_next, grad_clip)
                dh = dh_prev
                dc_next = dc_prev
            dx.insert(0, dx_step)  # 在列表开头插入当前时间步的输入梯度
            dh_next = dh_prev

        dx = np.concatenate(dx, axis=1)  # 拼接各时间步的输入梯度
        return dx

class Embedding(Module):
    """
    嵌入层，将词语索引转换为词向量。
    """
    def __init__(self, vocab_size, embedding_dim, init_range=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # 初始化嵌入矩阵
        self.embedding_matrix = np.random.uniform(-init_range, init_range, (vocab_size, embedding_dim))
        
        # 存储参数和梯度
        self.parameters = [self.embedding_matrix]
        self.gradients = [np.zeros_like(self.embedding_matrix)]
        
    def forward(self, inputs):
        """
        前向传播

        参数:
        - inputs: 词语索引序列，形状为 (seq_len, batch_size)

        返回:
        - outputs: 词向量序列，形状为 (embedding_dim, seq_len * batch_size)
        """
        self.inputs = inputs
        outputs = self.embedding_matrix[inputs]  # shape: (seq_len, batch_size, embedding_dim)
        outputs = outputs.transpose(2, 0, 1)  # 转换为 (embedding_dim, seq_len, batch_size)
        return outputs.reshape(self.embedding_dim, -1)  # (embedding_dim, seq_len * batch_size)
    
    def backward(self, doutputs):
        """
        反向传播

        参数:
        - doutputs: 词向量序列的梯度，形状为 (embedding_dim, seq_len * batch_size)

        返回:
        - dinputs: 输入的梯度，形状为 (seq_len, batch_size)
        """
        doutputs = doutputs.reshape(self.embedding_dim, -1, 1)  # (embedding_dim, seq_len * batch_size, 1)
        self.gradients[0].fill(0)  # 清零梯度
        for i in range(doutputs.shape[1]):
            idx = self.inputs[i]
            self.gradients[0][idx] += doutputs[:, i, 0]
        # 返回的梯度不传递给前面的层
        return None

class Linear(Module):
    """
    线性层，实现 y = Wx + b。
    """
    def __init__(self, input_size, output_size, bias=True, init_range=0.1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = bias

        # 初始化权重和偏置
        self.W = np.random.uniform(-init_range, init_range, (output_size, input_size))
        self.b = np.zeros((output_size, 1)) if self.use_bias else None
        
        # 存储参数和梯度
        self.parameters = [self.W] if not self.use_bias else [self.W, self.b] 
        self.gradients = [np.zeros_like(param) for param in self.parameters]
        
    def forward(self, inputs):
        """
        前向传播

        参数:
        - inputs: 输入，形状为 (input_size, batch_size)

        返回:
        - outputs: 输出，形状为 (output_size, batch_size)
        """
        self.inputs = inputs
        outputs = np.dot(self.W, inputs)
        if self.use_bias:
            outputs += self.b
        return outputs
    
    def backward(self, doutputs):
        """
        反向传播

        参数:
        - doutputs: 输出的梯度，形状为 (output_size, batch_size)

        返回:
        - dinputs: 输入的梯度，形状为 (input_size, batch_size)
        """
        dinputs = np.dot(self.W.T, doutputs)
        dW = np.dot(doutputs, self.inputs.T)
        self.gradients[0] += dW
        if self.use_bias:
            db = np.sum(doutputs, axis=1, keepdims=True)
            self.gradients[1] += db
        return dinputs

class CrossEntropyLoss:
    """
    交叉熵损失函数
    """
    def __init__(self):
        pass

    def forward(self, inputs, targets, reduction='mean'):
        """
        前向计算损失值

        参数:
        - inputs: 模型的输出，形状为 (num_classes, batch_size)
        - targets: 目标类别索引，形状为 (batch_size,)
        - reduction: 损失归约方式，可选 'mean' 或 'sum'，默认为 'mean'

        返回:
        - loss: 标量，平均或求和后的损失值
        """
        self.inputs = inputs
        self.targets = targets
        num_classes, batch_size = inputs.shape

        # 计算每个样本的对数 softmax
        shifted_logits = inputs - np.max(inputs, axis=0, keepdims=True)
        log_probs = shifted_logits - np.log(np.sum(np.exp(shifted_logits), axis=0, keepdims=True))
        self.probs = np.exp(log_probs)
        loss = -log_probs[targets, range(batch_size)]
        if reduction == 'mean':
            return np.mean(loss)
        elif reduction == 'sum':
            return np.sum(loss)

    def backward(self):
        """
        反向传播，计算输入的梯度

        返回:
        - dinputs: 模型输出的梯度，形状与 inputs 相同
        """
        batch_size = self.inputs.shape[1]
        dinputs = self.probs.copy()
        dinputs[self.targets, range(batch_size)] -= 1
        dinputs /= batch_size
        return dinputs

class AdamOptimizer:
    """
    Adam 优化器
    """
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = [np.zeros_like(param) for param in parameters]  # 一阶矩
        self.v = [np.zeros_like(param) for param in parameters]  # 二阶矩
        self.t = 0  # 时间步
        
    def apply_gradient(self, param, grad):
        """
        应用梯度，更新参数

        参数:
        - param: 待更新的参数
        - grad: 参数的梯度
        """
        self.t += 1
        idx = self.parameters.index(param)
        
        # 应用权重衰减
        if self.weight_decay != 0:
            grad += self.weight_decay * param
        
        # 更新一阶矩和二阶矩估计
        self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * grad
        self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (grad ** 2)
        
        # 修正一阶矩和二阶矩的偏差
        m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
        v_hat = self.v[idx] / (1 - self.beta2 ** self.t)
        
        # 更新参数
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class LanguageModel(Module):
    """
    语言模型，包括词嵌入层、LSTM 层和线性输出层。
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, init_range=0.1, bias=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = Embedding(vocab_size, embedding_dim, init_range)
        self.lstm = LSTMLayer(embedding_dim, hidden_size, num_layers, init_range)
        self.output_layer = Linear(hidden_size, vocab_size, bias, init_range)
        
        # 存储参数和梯度
        self.parameters = self.embedding.parameters + self.lstm.parameters + self.output_layer.parameters
        self.gradients = self.embedding.gradients + self.lstm.gradients + self.output_layer.gradients
        
    def forward(self, inputs, h0, c0):
        """
        前向传播

        参数:
        - inputs: 词语索引序列，形状为 (seq_len, batch_size)
        - h0: 初始隐藏状态，形状为 (hidden_size, batch_size)
        - c0: 初始细胞状态，形状为 (hidden_size, batch_size)

        返回:
        - outputs: 每个时间步的输出，形状为 (vocab_size, seq_len * batch_size)
        - (hn, cn): 最后一个时间步的隐藏状态和细胞状态，形状均为 (hidden_size, batch_size)
        """
        embeddings = self.embedding.forward(inputs)
        lstm_outputs, (hn, cn) = self.lstm.forward(embeddings, h0, c0)
        outputs = self.output_layer.forward(lstm_outputs)
        return outputs, (hn, cn)
        
    def backward(self, doutputs, dh, dc):  
        """
        反向传播

        参数:
        - doutputs: 每个时间步的输出梯度，形状为 (vocab_size, seq_len * batch_size)
        - dh: 最后时间步的隐藏状态梯度，形状为 (hidden_size, batch_size)
        - dc: 最后时间步的细胞状态梯度，形状为 (hidden_size, batch_size)

        返回:
        - 无需返回输入梯度
        """
        d_lstm_outputs = self.output_layer.backward(doutputs)
        d_embeddings = self.lstm.backward(d_lstm_outputs, dh, dc)
        if d_embeddings is not None:
            self.embedding.backward(d_embeddings)

class LanguageModelTrainer:
    """
    语言模型训练器
    """
    def __init__(self, model, loss_func, optimizer):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer

    def train(self, inputs, targets, h0, c0, grad_clip=5):
        """
        训练一个批次的数据

        参数:
        - inputs: 输入词语索引序列，形状为 (seq_len, batch_size)
        - targets: 目标词语索引序列，形状为 (seq_len, batch_size)
        - h0: 初始隐藏状态，形状为 (hidden_size, batch_size)
        - c0: 初始细胞状态，形状为 (hidden_size, batch_size)
        - grad_clip: 梯度裁剪阈值，默认为 5

        返回:
        - loss: 标量，平均损失值
        """
        self.model.zero_grad()
        outputs, (hn, cn) = self.model.forward(inputs, h0, c0)
        loss = self.loss_func.forward(outputs, targets.flatten())
        dloss = self.loss_func.backward()
        dh = np.zeros_like(h0)
        dc = np.zeros_like(c0)
        self.model.backward(dloss, dh, dc)
        
        # 应用梯度
        for param, grad in zip(self.model.parameters, self.model.gradients):
            grad = np.clip(grad, -grad_clip, grad_clip)
            self.optimizer.apply_gradient(param, grad)
        return loss

def generate_dummy_data():
    text = '你好 世界 。 机器学习 是 有趣 的 。 你好 机器 。'
    return text

def train_model():
    # 生成数据
    text = generate_dummy_data()
    tokenizer = Tokenizer(text)
    data = tokenizer.text_to_sequence(text)
    
    # 初始化模型和优化器
    vocab_size = tokenizer.vocab_size
    embedding_dim = 10
    hidden_size = 20
    num_layers = 1
    model = LanguageModel(vocab_size, embedding_dim, hidden_size, num_layers)
    loss_func = CrossEntropyLoss()
    optimizer = AdamOptimizer(model.parameters)
    trainer = LanguageModelTrainer(model, loss_func, optimizer)
    
    # 准备数据
    seq_len = 5
    batch_size = 1
    inputs = np.array([data[:-1]]).T  # (seq_len, batch_size)
    targets = np.array([data[1:]]).T  # (seq_len, batch_size)
    h0 = np.zeros((hidden_size, batch_size))
    c0 = np.zeros((hidden_size, batch_size))
    
    # 训练
    epochs = 100
    for epoch in range(epochs):
        loss = trainer.train(inputs, targets, h0, c0)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss:.4f}')
    
    return model, tokenizer

def generate_text(model, tokenizer, seed_text, max_len=20, temperature=1.0):
    generated_text = seed_text
    h = np.zeros((model.lstm.hidden_size, 1)) 
    c = np.zeros((model.lstm.hidden_size, 1))
    
    # 使用种子文本初始化隐藏状态和细胞状态
    seed_sequence = tokenizer.text_to_sequence(seed_text)
    for word_id in seed_sequence[:-1]:
        inputs = np.array([word_id]).reshape(1, 1)  # (seq_len=1, batch_size=1)
        _, (h, c) = model.forward(inputs, h, c)
    
    last_word_id = seed_sequence[-1]
    
    for _ in range(max_len):
        inputs = np.array([last_word_id]).reshape(1, 1)  # (seq_len=1, batch_size=1)
        outputs, (h, c) = model.forward(inputs, h, c)
        logits = outputs[:, -1]  # (vocab_size,)
        probs = np.exp(logits / temperature)
        probs /= np.sum(probs)
        next_word_id = np.random.choice(len(probs), p=probs)
        next_word = tokenizer.ix_to_word[next_word_id]
        if next_word == '<EOS>':
            break
        generated_text += next_word
        last_word_id = next_word_id
    return generated_text

# 主程序
if __name__ == "__main__":
    # 训练模型
    model, tokenizer = train_model()
    
    # 示例生成
    seed_text = '你好'
    generated = generate_text(model, tokenizer, seed_text)
    print(f'Generated Text: {generated}')
