#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os # 导入os模块，用于操作系统功能，如文件路径
os.environ['CUDA_VISIBLE_DEVICES'] = "5" # 设置CUDA_VISIBLE_DEVICES环境变量，指定使用的GPU编号

# 加载模型---------------------------------------------------------------------------

from pathlib import Path # 导入路径操作库
from typing import Annotated, Union # 导入类型注解支持库

import typer # 导入命令行应用构建库
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM # 从peft库导入模型类
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
) # 从transformers库导入模型和分词器类

ModelType = Union[PreTrainedModel, PeftModelForCausalLM] # 定义模型类型的联合类型
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast] # 定义分词器类型的联合类型

# 定义函数解析路径
def _resolve_path(path: Union[str, Path]) -> Path:
    # 将路径展开为绝对路径
    return Path(path).expanduser().resolve()

# 定义加载模型和分词器的函数
def load_model_and_tokenizer(model_dir: Union[str, Path]) -> tuple[ModelType, TokenizerType]:
    model_dir = _resolve_path(model_dir) # 解析模型目录路径
    # 如果adapter配置文件存在
    if (model_dir / 'adapter_config.json').exists():
        # 加载Peft模型
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        # 获取分词器目录
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        # 加载transformers模型
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        # 设置分词器目录为模型目录
        tokenizer_dir = model_dir
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=True
    )
    return model, tokenizer # 返回模型和分词器


model_dir = "/workspace/dujh22/ce_finetune/output_single/checkpoint-3000/"
model, tokenizer = load_model_and_tokenizer(model_dir) # 加载模型和分词器
    
# -----------------------------------------------------------------------------------


# 导入所需的库
import argparse  # 用于解析命令行参数

import os  # 用于处理文件和目录
import sys
import numpy as np  # 用于数学运算
import pandas as pd  # 用于数据处理
import time  # 用于处理时间相关的任务

from crop import crop  # 从 crop 模块导入 crop 函数
from tqdm import tqdm # 进度条

import concurrent.futures # 并发测试

choices = ["A", "B", "C", "D"]  # 定义多项选择题的选项

# 定义 softmax 函数，用于计算概率分布
def softmax(x):
    # 将输入的x减去x中的最大值，以提高计算的数值稳定性
    z = x - max(x)
    # 对z中的每个元素进行指数运算
    numerator = np.exp(z)
    # 计算指数运算结果的总和，作为分母
    denominator = np.sum(numerator)
    # 计算softmax值，即指数运算结果除以总和
    softmax = numerator/denominator
    # 返回softmax计算结果
    return softmax


# 格式化主题名称，将下划线分隔的字符串转换为带空格的字符串
def format_subject(subject):
    # 使用下划线将字符串分割成列表
    l = subject.split("_")
    # 初始化一个空字符串
    s = ""
    # 遍历分割后的列表
    for entry in l:
        # 将每个部分添加到字符串s中，并在前面加一个空格
        s += " " + entry
    # 返回格式化后的字符串
    return s


# 格式化多项选择题的示例
def format_example(df, idx, include_answer=True):
    # 从DataFrame中获取特定索引行的题干
    prompt = df.iloc[idx, 0]
    # 计算选项的数量（DataFrame的列数减去2，通常是题干和答案列）
    k = df.shape[1] - 2
    # 遍历所有选项
    for j in range(k):
        # 将每个选项附加到题干字符串上
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    # 添加答案的提示
    prompt += "\nAnswer:"
    # 如果包含答案，则将答案附加到字符串上
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    # 返回格式化后的多项选择题
    return prompt


# 生成包含多项选择题的提示信息
def gen_prompt(train_df, subject, k=-1):
    # 使用format_subject函数格式化主题名称，并构建提示信息的开头部分
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    # 如果k为-1，表示需要使用数据集中所有的问题
    if k == -1:
        k = train_df.shape[0]
    # 遍历数据集中的问题，数量由k决定
    for i in range(k):
        # 使用format_example函数格式化每个问题，并添加到提示信息中
        prompt += format_example(train_df, i)
    # 返回最终构建的包含多项选择题的提示信息
    return prompt

# 并发eval函数
def process_question(i, args, subject, dev_df, test_df):
    # 这里是处理单个问题的逻辑
    k = args.ntrain
    prompt_end = format_example(test_df, i, include_answer=False)
    train_prompt = gen_prompt(dev_df, subject, k)
    prompt = train_prompt + prompt_end

    while crop(prompt) != prompt:
        k -= 1
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

    label = test_df.iloc[i, test_df.shape[1]-1]

    pred = 'N'
    for i in range(10):
        try:
            # 调用上述模型
            response, _ = model.chat(tokenizer, prompt) # 使用模型和分词器生成响应
            if response != "":
                pred = response[0]
                if pred in choices:
                    print(f"Question {i} - Prediction: {pred} - Label: {label}")
                    break
                else:
                    continue 
            else:
                continue       
        except Exception as e:
            print("An error occurred:", str(e))
            time.sleep(1)  # 需要import time
            continue
        
    # 计算预测结果和标签的准确率
    cor = pred == label
    return cor

def eval_concurrent(args, subject, dev_df, test_df):
    cors = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        # 将处理函数所需的参数直接传递给它
        futures = [executor.submit(process_question, i, args, subject, dev_df, test_df) for i in range(test_df.shape[0])]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Processing'):
            cors.append(future.result())

    acc = np.mean(cors)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return np.array(cors), acc


# 定义 main 函数，用于处理主要的逻辑
def main(args):
    models = args.model # 从命令行参数中获取AI引擎
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f]) # 获取所有主题名称

    # 如果保存目录不存在，则创建
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    # 为每个引擎创建一个结果目录
    for model in models:
        if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(model))):
            os.mkdir(os.path.join(args.save_dir, "results_{}".format(model)))

    # 打印主题和引擎
    print(subjects)
    # 打印命令行参数
    print(args)

    # 遍历所有引擎
    for model in models:
        print(model)
        all_cors = []
        all_acc = {}

        # 遍历所有主题
        for subject in subjects:
            # 读取训练集、开发集和测试集
            dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
            test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

            # 评估模型在测试集上的表现
            cors, acc = eval_concurrent(args, subject, dev_df, test_df)
            all_cors.append(cors) # 将准确率添加到列表中
            all_acc[subject] = acc # 将准确率添加到字典中

            # 将结果保存到CSV文件中
            test_df["{}_correct".format(model)] = cors # 将准确率添加到DataFrame中
            test_df.to_csv(os.path.join(args.save_dir, "results_{}".format(model), "{}.csv".format(subject)), index=None) # 保存结果到CSV文件

        weighted_acc = np.mean(np.concatenate(all_cors)) # 计算加权平均准确率
        print("Average accuracy: {:.3f}".format(weighted_acc)) # 打印加权平均准确率
        all_acc["average"] = weighted_acc # 将加权平均准确率添加到字典中

        # 将准确率字典输出到txt文件中汇总
        with open(os.path.join(args.save_dir, "results_{}".format(model), "accuracy.txt"), "w") as f:
            f.write(str(all_acc))

# 当这个脚本作为主程序运行时，下面的代码块将被执行
if __name__ == "__main__":
    # 创建一个解析命令行参数的解析器
    parser = argparse.ArgumentParser()
    # 添加一个命令行参数来指定训练集中使用的问题数量，默认值为5
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    # 添加一个命令行参数来指定数据存储的目录，默认为"data"
    parser.add_argument("--data_dir", "-d", type=str, default="/workspace/dujh22/MMLU/data")
    # 添加一个命令行参数来指定结果存储的目录，默认为"results"
    parser.add_argument("--save_dir", "-s", type=str, default="/workspace/dujh22/MMLU/results")
    # 添加一个命令行参数来指定使用的模型名称
    parser.add_argument("--model", "-m", type=list, default=["chatglm3-6b-sft"])
    # 解析命令行参数
    args = parser.parse_args()
    # 调用main函数，传入解析得到的参数
    main(args)