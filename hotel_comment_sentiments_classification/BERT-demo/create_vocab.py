#!/usr/bin/env python
# coding: utf-8

# In[3]:


"""
    制作vocab.txt, 语料中出现的word
"""
from collections import Counter
import pandas as pd

# 计算每个词出现的次数
word_counter = Counter()
f = pd.read_csv('./data/train_set.csv', sep='\t', encoding='utf-8')
data = f['text'].tolist()
for text in data:
    words = text.split()
    for word in words:
        word_counter[word] += 1

words = word_counter.keys()


# In[4]:


words


# In[7]:


# 打开文件，采用写入模式
# 若文件不存在,创建，若存在，清空并写入
my_open = open('./bert-mini/vocab.txt', 'w')
               
extra_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
my_open.writelines("\n".join(extra_tokens))
# 换行
my_open.writelines('\n')
my_open.writelines("\n".join(words))
my_open.close()

