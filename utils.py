import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from time import time
import torch.utils.data as Data
from collections import defaultdict


def letter_transform(word: str):
    word = word.lower()
    a = 97
    x = []
    y = []
    for i in range(len(word)):
        x_one_hot = [0] * 28  # 26 + '-' + eof
        y_one_hot = [0] * 28
        # x
        if i < len(word):
            if word[i] == ' ' or word[i] == '-' or word[i] == '\'':
                x_one_hot[26] = 1
            else:
                x_one_hot[ord(word[i]) - a] = 1
        # y
        if i < len(word) - 1:
            if word[i + 1] == ' ' or word[i + 1] == '-' or word[i + 1] == '\'':
                y_one_hot[26] = 1
            else:
                y_one_hot[ord(word[i + 1]) - a] = 1
        elif i == len(word) - 1:
            y_one_hot[27] = 1
        x.append(x_one_hot)
        y.append(y_one_hot)
    return x, y


def vec2word(vec):
    name = ''
    for i in vec:
        if 0 <= i < 26:
            name += chr(i + 97)
        elif i == 26:
            name += '-'
        elif i == 27:
            name += '#'
    return name


def load_data():
    # files = ['pet.txt']
    files = ['female.txt', 'male.txt', 'pet.txt']
    # files = ['same_len.txt']

    x_data = defaultdict(list)
    y_data = defaultdict(list)
    # length = []
    for file in files:
        f = open('baby names/{}'.format(file), 'r')
        for line in f.readlines():
            line = line.strip('\n')
            if line == '' or '#' in line:
                continue
            line = line.split('\t')
            name = line[0]
            x, y = letter_transform(name)
            x_data[len(name)].append(x)
            y_data[len(name)].append(y)
            # length.append(len(line[0]))
        f.close()

    return x_data, y_data, x_data.keys()
    # return x_data, y_data, length


def get_entroy_loss(output, labels):  # (batch, dim)
    # masks = masks.tolist()
    target = torch.argmax(labels, dim=1)
    entroy = nn.CrossEntropyLoss(reduction='sum')
    loss = entroy(output, target)
    return loss





