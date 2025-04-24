"""Byte pair encoding utilities (Adapted from the official GPT-2 GitHub repository)"""
import json
import os
import regex as re
import requests
import sys

from functools import lru_cache
from tqdm import tqdm

# 定义 _get_encoder 函数，用于下载编码器和词汇表
def _get_encoder(subdir):
    print("Downloading encoder and vocab to ", subdir)
    for filename in ['encoder.json', 'vocab.bpe']:
        r = requests.get("https://openaipublic.blob.core.windows.net/gpt-2/" + subdir + "/" + filename, stream=True)
        with open(os.path.join(subdir, filename), 'wb') as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)

# 定义 bytes_to_unicode 函数，用于将字节编码映射到 Unicode 字符
@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.

    返回 utf-8 字节列表和相应的 unicode 字符串列表。
    可逆 bpe 编码适用于 unicode 字符串。
    这意味着如果要避免 UNK，就需要在词汇表中包含大量的 unicode 字符。
    当你使用类似 10B 标记的数据集时，你最终需要大约 5K 才能达到不错的覆盖率。
    这在正常的 32K bpe 词汇表中占了很大比例。
    为了避免这种情况，我们需要在 utf-8 字节和 unicode 字符串之间建立查找表。
    并避免映射到 bpe 代码所使用的空白/控制字符。
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

# 定义 get_pairs 函数，用于获取单词中的符号对:两两相邻的符号组成的集合
def get_pairs(word):
    """Return set of symbol pairs in a word. 返回单词中的符号对集合。

    Word is represented as tuple of symbols (symbols being variable-length strings). 单词用符号元组表示（符号是长度可变的字符串）。
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


# 定义 Encoder 类，用于编码和解码
class Encoder:
    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder  # 编码器，用于将字符映射到整数编码
        self.decoder = {v:k for k,v in self.encoder.items()}  # 解码器，用于将整数编码映射回字符
        self.errors = errors  # 在解码时如何处理错误
        self.byte_encoder = bytes_to_unicode()  # 将字节编码映射到 Unicode 字符的编码器
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}  # 将 Unicode 字符映射回字节编码的解码器
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))  # BPE 合并的排名字典
        self.cache = {}  # 缓存，用于存储已处理过的 token

        # 应该添加 re.IGNORECASE，以便可以对缩写的大写版本进行 BPE 合并
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")  # 正则表达式模式，用于标记文本中的单词、数字和标点符号

    # 定义 bpe 函数，用于对 token 进行 BPE 编码
    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]  # 如果 token 已在缓存中，则直接返回其处理结果
        word = tuple(token)
        pairs = get_pairs(word)  # 获取 token 中所有字符的相邻字符对

        if not pairs:  # 如果没有相邻字符对，直接返回 token
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))  # 找到排名最小的相邻字符对
            if bigram not in self.bpe_ranks:  # 如果该相邻字符对不在 BPE 合并字典中，停止循环
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:  # 如果 token 中只有一个字符，停止循环
                break
            else:
                pairs = get_pairs(word)  # 更新相邻字符对列表
        word = ' '.join(word)  # 将字符列表连接为字符串
        self.cache[token] = word  # 将处理结果存入缓存
        return word

    # 定义 encode 函数，用于对文本进行编码
    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):  # 使用正则表达式模式对文本进行分词
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))  # 将 token 编码为字节，并使用 byte_encoder 进行编码
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))  # 对编码后的 token 进行 BPE 编码，并添加到 bpe_tokens 中
        return bpe_tokens  # 返回 BPE 编码后的 token 列表

    # 定义 decode 函数，用于对 token 进行解码
    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])  # 将 token 解码为文本
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)  # 将字节编码解码为 Unicode 字符
        return text  # 返回解码后的文本


# 定义 get_encoder 函数，用于获取编码器
def get_encoder(model_name):
    # 创建子目录
    subdir = os.path.join("models", model_name)
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    if not os.path.exists(os.path.join(subdir, 'encoder.json')):
        _get_encoder(subdir)

    
    subdir = subdir.replace('\\','/') # needed for Windows

    # 读取编码器和词汇表
    with open(os.path.join(subdir, 'encoder.json'), 'r') as f:
        encoder = json.load(f)
    with open(os.path.join(subdir, 'vocab.bpe'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    # 读取BPE合并
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]] # 读取BPE合并,具体请参考GPT-2论文？
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )

# 使用 get_encoder 函数获取编码器
enc = get_encoder('124M')

# 定义 crop_prompt 函数，用于截取提示信息
def crop_prompt(prompt: str):
    global enc # 使用全局变量enc, 用于编码和解码

    # 如果提示信息长度大于2048，则截取前2048个字符
    cropped_prompt = enc.decode(enc.encode(prompt)[:2048]) # 编码、截取、解码
    return cropped_prompt

# 定义 crop 函数，用于截取提示信息
def crop(s):
    # 如果提示信息长度大于2048，则截取前2048个字符
    prompt = crop_prompt(s)
    return prompt

