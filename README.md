# Neural Text Process lib

## Introduction

A neural text process python lib for sequence tagging data generating.

Support feature template which used to extract context-based feature from text.

## Suffix

Support feature templates suffixes enabled or disabled.

For example, there are few context-based feature templates:

```
# Fields
w y
# Templates
w:-2
w:-1
w: 0
w: 1
w: 2
```

Given the sentence "我爱北京天安门。", and current character is now "北", then it will extract features for "北" as:

1. Suffix enbaled

```
w[-2]:我
w[-1]:爱
w[0]:北
w[1]:京
w[2]:天
```

2. Suffix disabled

```
我
爱
北
京
天
```

Disabled suffixes can be used to extract raw word from a window.

## History

- **2017-09-25 ver 0.1.3**
  - Add new method to return the size of feature templates
  - Replace both 'START'&'END' tag with '</s>'
- **2017-09-12 ver 0.1.2**
  - label2idx's index starts from 1
  - Index of unknow words or labels will be 0
- **2017-09-04 ver 0.1.1**
  - Index of feature ‘OOV’ set to default 0
  - label2idx's index starts from 0
- **2017-08-26 ver 0.1.0**
  - First version