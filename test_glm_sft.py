#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os # 导入os模块，用于操作系统功能，如文件路径
os.environ['CUDA_VISIBLE_DEVICES'] = "5" # 设置CUDA_VISIBLE_DEVICES环境变量，指定使用的GPU编号

# 加载模型---------------------------------------------------------------------------


from pathlib import Path # 导入路径操作库
from typing import Annotated, Union # 导入类型注解支持库

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
prompt = "The following are multiple choice questions (with answers) about college physics\n\nFor which of the following thermodynamic processes is the increase in the internal energy of an ideal gas equal to the heat added to the gas?\nA. Constant temperature\nB. Constant volume\nC. Constant volume\nD. Adiabatic\nAnswer:B\n\nThe following are multiple choice questions (with answers) about college physics\n\nA refracting telescope consists of two converging lenses separated by 100 cm. The eye-piece lens has a focal length of 20 cm. The angular magnification of the telescope is\nA. 4\nB. 5\nC. 6\nD. 20\nAnswer:"
attempts = 0
while attempts < 10:
    try:
        # 调用上述模型
        response, _ = model.chat(tokenizer, prompt) # 使用模型和分词器生成响应
        
        # 添加GLM响应到数据并返回
        if response != "":
            print(response) # 打印响应
            print("提取第一个答案：")
            print(response[0]) # 打印响应
            break
    except Exception as e:
        print(f"在处理 {prompt} 时遇到异常：{e}")
        attempts += 1
        print(f"重试次数 {attempts}/10")