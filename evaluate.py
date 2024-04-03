# 导入所需的库
import argparse  # 用于解析命令行参数
import openai  # 导入 OpenAI 库
import os  # 用于处理文件和目录
import numpy as np  # 用于数学运算
import pandas as pd  # 用于数据处理
import time  # 用于处理时间相关的任务

from crop import crop  # 从 crop 模块导入 crop 函数

openai.api_key = "INSERTYOURKEYHERE"  # 设置 OpenAI 的 API 密钥
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

def eval(args, subject, engine, dev_df, test_df):
    # 初始化正确答案的列表和所有概率的列表
    cors = []
    all_probs = []
    # 获取可能的答案选项
    answers = choices[:test_df.shape[1]-2]

    # 遍历测试数据集中的每一个问题
    for i in range(test_df.shape[0]):
        # 获取提示信息，并确保其长度适合
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        # 如果提示信息太长，则缩减训练集问题数量，直至长度合适
        while crop(prompt) != prompt:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        # 获取正确答案标签
        label = test_df.iloc[i, test_df.shape[1]-1]

        # 循环直至成功获取模型的预测结果
        while True:
            try:
                c = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    max_tokens=1,
                    logprobs=100,
                    temperature=0,
                    echo=True
                )
                break
            except:
                print("pausing")
                time.sleep(1)
                continue

        # 获取每个答案的对数概率
        lprobs = []
        for ans in answers:
            try:
                lprobs.append(c["choices"][0]["logprobs"]["top_logprobs"][-1][" {}".format(ans)])
            except:
                print("Warning: {} not found. Artificially adding log prob of -100.".format(ans))
                lprobs.append(-100)
        # 预测答案
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(lprobs)]
        # 计算概率
        probs = softmax(np.array(lprobs))

        # 判断预测是否正确
        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    # 计算平均准确率
    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


# 定义 eval 函数，用于评估多项选择题在指定的测试集（test_df）上的表现。
def eval(args, subject, engine, dev_df, test_df):
    # 初始化正确答案的列表和所有概率的列表
    cors = []
    all_probs = []
    # 获取可能的答案选项
    answers = choices[:test_df.shape[1]-2]

    # 遍历测试数据集中的每一个问题
    for i in range(test_df.shape[0]):
        # 获取提示信息，并确保其长度适合
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        # 如果提示信息太长，则缩减训练集问题数量，直至长度合适
        while crop(prompt) != prompt:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        # 获取正确答案标签
        label = test_df.iloc[i, test_df.shape[1]-1]

        # 循环直至成功获取模型的预测结果
        while True:
            try:
                c = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    max_tokens=1,
                    logprobs=100,
                    temperature=0,
                    echo=True
                )
                break
            except:
                print("pausing")
                time.sleep(1)
                continue

        # 获取每个答案的对数概率
        lprobs = []
        for ans in answers:
            try:
                lprobs.append(c["choices"][0]["logprobs"]["top_logprobs"][-1][" {}".format(ans)])
            except:
                print("Warning: {} not found. Artificially adding log prob of -100.".format(ans))
                lprobs.append(-100)
        # 预测答案
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(lprobs)]
        # 计算概率
        probs = softmax(np.array(lprobs))

        # 判断预测是否正确
        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    # 计算平均准确率
    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


# 当这个脚本作为主程序运行时，下面的代码块将被执行
if __name__ == "__main__":
    # 创建一个解析命令行参数的解析器
    parser = argparse.ArgumentParser()
    # 添加一个命令行参数来指定训练集中使用的问题数量，默认值为5
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    # 添加一个命令行参数来指定数据存储的目录，默认为"data"
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    # 添加一个命令行参数来指定结果存储的目录，默认为"results"
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    # 添加一个命令行参数来指定使用的AI引擎，默认包含"davinci", "curie", "babbage", "ada"这四种
    parser.add_argument("--engine", "-e", choices=["davinci", "curie", "babbage", "ada"],
                        default=["davinci", "curie", "babbage", "ada"], nargs="+")
    # 解析命令行参数
    args = parser.parse_args()
    # 调用main函数，传入解析得到的参数
    main(args)