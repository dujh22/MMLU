# 导入所需的库
import config
import argparse  # 用于解析命令行参数
import openai  # 导入 OpenAI 库

import os  # 用于处理文件和目录
import sys
import numpy as np  # 用于数学运算
import pandas as pd  # 用于数据处理
import time  # 用于处理时间相关的任务

from crop import crop  # 从 crop 模块导入 crop 函数
from tqdm import tqdm # 进度条

import concurrent.futures # 并发测试

openai.api_key = config.GPT_API_KEY  # 设置 OpenAI 的 API 密钥
openai.api_base = config.GPT_BASE_URL # 原始代码中是不需要设置这一行的
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

# 定义 eval 函数，用于评估多项选择题在指定的测试集（test_df）上的表现。(这个函数是单点测试，没有并发)
def eval_single(args, subject, model, dev_df, test_df):
    # 初始化正确答案的列表
    cors = []

    # 遍历测试数据集中的每一个问题
    for i in tqdm(range(test_df.shape[0]), desc='Processing'):
    # for i in range(test_df.shape[0]):
        # 获取提示信息，并确保其长度适合
        k = args.ntrain # 从命令行参数中获取训练集问题数量
        prompt_end = format_example(test_df, i, include_answer=False) # 获取问题的提示信息
        train_prompt = gen_prompt(dev_df, subject, k) # 生成训练集的提示信息
        prompt = train_prompt + prompt_end # 将训练集和测试集的提示信息合并

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
                c = openai.ChatCompletion.create(
                    model=model, 
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1,
                    # logprobs=True, # 模型不支持
                    temperature=0,
                ) # model: 使用的模型名称。prompt: 提示文本，模型将基于此生成文本。max_tokens: 模型生成的最大令牌数。logprobs: 是否返回输出令牌的对数概率。如果为 true，则返回的每个输出标记的对数概率。temperature: 控制输出随机性的参数。值越高，输出越随机；值越低，输出越确定。echo: 如果为True，响应将包含输入提示：输出请求和响应日志
                break
            except Exception as e:
                print("An error occurred:", str(e))
                print("Exception type:", sys.exc_info()[0])
                print("Exception value:", sys.exc_info()[1])
                print("pausing")
                time.sleep(1)
                continue

        # 判断预测是否正确
        pred = c.choices[0].message.content
        cor = pred == label
        # print("id:{},pred:{}, label:{}, cor:{}".format(i, pred, label, cor))
        cors.append(cor)

    # 计算平均准确率
    acc = np.mean(cors)
    cors = np.array(cors)

    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc # 返回准确率、平均准确率和概率

# 并发eval函数
def process_question(i, args, subject, model, dev_df, test_df):
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

    # while True:
    for i in range(2):
        try:
            c = openai.ChatCompletion.create(
                model=model, 
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                temperature=0,
            )
            pred = c.choices[0].message.content
            break
        except Exception as e:
            # print("An error occurred:", str(e))
            # time.sleep(1)  # 需要import time
            pred = 'N'
            continue
   
    cor = pred == label
    return cor

def eval_concurrent(args, subject, model, dev_df, test_df):
    cors = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # 将处理函数所需的参数直接传递给它
        futures = [executor.submit(process_question, i, args, subject, model, dev_df, test_df) for i in range(test_df.shape[0])]
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
            cors, acc = eval_concurrent(args, subject, model, dev_df, test_df)
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
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    # 添加一个命令行参数来指定结果存储的目录，默认为"results"
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    # 添加一个命令行参数来指定使用的模型名称
    parser.add_argument("--model", "-m", type=list, default=["gpt-3.5-turbo"])
    # 解析命令行参数
    args = parser.parse_args()
    # 调用main函数，传入解析得到的参数
    main(args)