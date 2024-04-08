import os
import json
import csv

# 定义存储数据的字典
results_summary = {}

# 路径到results文件夹
results_dir = 'results'

# 遍历results文件夹内的所有子文件夹
for subdir in os.listdir(results_dir):
    subdir_path = os.path.join(results_dir, subdir)
    if os.path.isdir(subdir_path):
        # 读取accuracy.txt文件
        accuracy_file_path = os.path.join(subdir_path, 'accuracy.txt')
        with open(accuracy_file_path, 'r', encoding='utf-8') as f:
            # 读取文件内容并将单引号替换为双引号
            file_content = f.read().replace("'", '"')
            # 使用json.loads解析修正后的字符串
            data = json.loads(file_content)
            model_name = subdir.split('_')[-1]  # 假设模型名称是路径的最后一部分
            for subject, accuracy in data.items():
                if subject not in results_summary:
                    results_summary[subject] = {}
                results_summary[subject][model_name] = accuracy

# 写入CSV文件
csv_file_path = os.path.join(results_dir, 'results_summary.csv')
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    # 写入标题行：科目和各模型名称
    headers = ['Subject'] + sorted(list(results_summary[next(iter(results_summary))].keys()))
    # iter(results_summary)是获取字典的键，next(iter(results_summary))是获取第一个键
    # results_summary[next(iter(results_summary))].keys()是获取第一个键对应的值的键
    writer.writerow(headers)
    
    # 写入每一行的数据
    for subject, accuracies in sorted(results_summary.items()):
        row = [subject] + [accuracies.get(model_name, '') for model_name in headers[1:]]
        writer.writerow(row)

print(f"汇总完成，数据已保存到'{csv_file_path}'")
