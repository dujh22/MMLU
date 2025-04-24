# V1版本-------------------------------------------------------------------

# from openai import OpenAI
# import config


# client = OpenAI(api_key=config.GPT_API_KEY)

# prompt = ["你好"]
# response = client.completions.create(
#   model="gpt-3.5-turbo-instruct",
#   prompt=prompt,
#   max_tokens=7,
#   temperature=0
# )

# print(response)
# print("\n\n")
# for i in range(len(response.choices)):
#     print(response.choices[i].text, end="")

# V2版本-------------------------------------------------------------------
import openai  # 导入 OpenAI 库
import config
openai.api_key = config.GPT_API_KEY  # 设置 OpenAI 的 API 密钥
openai.api_base = config.GPT_BASE_URL # 原始代码中是不需要设置这一行的

c = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", 
                    messages=[{"role": "user", "content": "The following are multiple choice questions (with answers) about  abstract algebra.\n\nFind the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\nA. 0\nB. 4\nC. 2\nD. 6\nAnswer:"}],
                    max_tokens=1,
                    # logprobs=True,
                    temperature=0,
                )

print(c.choices[0].message.content)