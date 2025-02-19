# -*- coding: utf-8 -*-
# @Time    : 2021/7/7 17:22
# @FileName: 矢量化代码.py
# @Software: PyCharm
# from gensim.models import Word2Vec
import re
import os
import warnings
import numpy as np
# from gensim.models import Word2Vec
import time


# import fasttext.util
# fasttext.util.download_model('en', if_exists='ignore')  # English
# ft = fasttext.load_model('cc.en.300.bin')
# fasttext.util.reduce_model(ft, 100)
# sentenses = ['this is a test', 'hello world']
# for sentense in sentenses:
#     vec = ft.get_sentence_vector(sentense)
#     print(vec.shape)
#     print(ft.get_sentence_vector(sentense))
#
warnings.filterwarnings("ignore")

operators3 = {'<<=', '>>='}
operators2 = {
    '->', '++', '--',
    '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=',
    '-=', '*=', '/=', '%=', '&=', '^=', '|='
}
operators1 = {
    '(', ')', '[', ']', '.',
    '+', '-', '*', '&', '/',
    '%', '<', '>', '^', '|',
    '=', ',', '?', ':', ';',
    '{', '}'
}

"""
Functionality to train Word2Vec model and vectorize fragments
Trains Word2Vec model using list of tokenized fragments
Uses trained model embeddings to create 2D fragment vectors
"""


class FragmentVectorizer:
    def __init__(self, vector_length):
        self.fragments = []
        self.vector_length = vector_length
        self.forward_slices = 0
        self.backward_slices = 0
        self.fragment_length = []

    """
    Takes a line of solidity code (string) as input
    Tokenizes solidity code (breaks down into identifier, variables, keywords, operators)
    Returns a list of tokens, preserving order in which they appear
    """

    @staticmethod
    def tokenize(line):
        tmp, w = [], []
        i = 0
        while i < len(line):
            # Ignore spaces and combine previously collected chars to form words
            if line[i] == ' ':
                tmp.append(''.join(w))
                tmp.append(line[i])
                w = []
                i += 1
            # Check operators and append to final list
            elif line[i:i + 3] in operators3:
                tmp.append(''.join(w))
                tmp.append(line[i:i + 3])
                w = []
                i += 3
            elif line[i:i + 2] in operators2:
                tmp.append(''.join(w))
                tmp.append(line[i:i + 2])
                w = []
                i += 2
            elif line[i] in operators1:
                tmp.append(''.join(w))
                tmp.append(line[i])
                w = []
                i += 1
            # Character appended to word list
            else:
                w.append(line[i])
                i += 1
        # Filter out irrelevant strings
        res = list(filter(lambda c: c != '', tmp))
        return list(filter(lambda c: c != ' ', res))

    """
    Tokenize entire fragment
    Tokenize each line and concatenate to one long list
    """

    @staticmethod
    def tokenize_fragment(fragment):
        tokenized = []
        function_regex = re.compile('FUN(\d)+')
        backwards_slice = False
        for line in fragment:
            tokens = FragmentVectorizer.tokenize(line)
            tokenized += tokens
            if len(list(filter(function_regex.match, tokens))) > 0:
                backwards_slice = True
            else:
                backwards_slice = False
        return tokenized, backwards_slice

    """
    Add input fragment to model
    Tokenize fragment and buffer it to list
    """

    def add_fragment(self, fragment):
        tokenized_fragment, backwards_slice = FragmentVectorizer.tokenize_fragment(fragment)
        self.fragments.append(tokenized_fragment)
        if backwards_slice:
            self.backward_slices += 1
        else:
            self.forward_slices += 1

    """
    Uses Word2Vec to create a vector for each fragment
    Gets a vector for the fragment by combining token embeddings
    Number of tokens used is min of number_of_tokens and 100
    """

    def vectorize(self, fragment):
        # 这边可以进行改进，这边固定的维度就是100了，我应该先统计每个的长度，然后用中间的大部分来确定这个数字的大小.这边真的就是实锤了
        tokenized_fragment, backwards_slice = FragmentVectorizer.tokenize_fragment(fragment)
        vectors = np.zeros(shape=(100, self.vector_length))
        if backwards_slice:
            for i in range(min(len(tokenized_fragment), 100)):
                vectors[100 - 1 - i] = self.embeddings[tokenized_fragment[len(tokenized_fragment) - 1 - i]]
        else:
            self.fragment_length.append(len(tokenized_fragment))
            for i in range(min(len(tokenized_fragment), 100)):
                vectors[i] = self.embeddings[tokenized_fragment[i]]
        return vectors

    """
    Done adding fragments, then train Word2Vec model
    Only keep list of embeddings, delete model and list of fragments
    """

    def train_model(self):
        # Set min_count to 1 to prevent out-of-vocabulary errors
        # gensim中有关word2vec模型的参数：negative如果大于0，使用负采样,int为负数，指定应该绘制多少噪音词，默认是5
        model = Word2Vec(self.fragments,min_count=1, sg=1)
        self.embeddings = model.wv
        del model
        del self.fragments
        # return model

def parse_file(filename):
    with open(filename, "r", encoding="utf8") as file:
        fragment = []
        fragment_val = 0
        for line in file:
            stripped = line.strip()#去除首尾的空格
            if '.sol' not in stripped:
                if not stripped:
                    continue
                if "-" * 33 in line and fragment:
                    yield fragment, fragment_val
                    fragment = []
                elif stripped.split()[0].isdigit():
                    if fragment:
                        if stripped.isdigit():
                            fragment_val = int(stripped)
                        else:
                            fragment.append(stripped)
                else:
                    fragment.append(stripped)
    return fragment

if __name__ == '__main__':
    # start_Time = time.clock()
    vectorizer = FragmentVectorizer(100)
    with open('dao2.sol/test.sol', "r", encoding="utf8") as file:
        content = ''.join(file.readlines())
        w1 = vectorizer.tokenize_fragment(content)
        w2 = vectorizer.tokenize(content)
        print(w1)
        print(w2)
    # filepath='../codeFragment_extract/output_repeat_2.txt'
    #######判断一个code_fragment里面有多少代码token
    # filepath = '../data/ast/ast.txt'
    # fragment = parse_file(filepath)
    # codeSlicing_list = []
    # codeLengthList = []
    # for i,value in fragment:
    #     tokenize_fragment = vectorizer.tokenize_fragment(i)
    #     codeSlicing_list.append(tokenize_fragment[0])
    #     codeLengthList.append(len(tokenize_fragment[0]))
    # print(codeLengthList)
    #######判断一个code_fragment里面有多少代码token
    # filepath = '../data/code_slicing/code_slicing.txt'
    # fragment = parse_file(filepath)
    # codeSlicing_list = []
    # for i,value in fragment:
    #     tokenize_fragment = vectorizer.tokenize_fragment(i)
    #     codeSlicing_list.append(tokenize_fragment[0])
    # model = Word2Vec(codeSlicing_list, min_count=1, sg=1)
    # if os.path.exists('./vector_codeSlicing_normalized.txt'):
    #     pass
    # else:
    #     model.wv.save_word2vec_format('./vector_codeSlicing_normalized.txt')
    # end_Time = time.clock()
    # print(end_Time - start_Time)
