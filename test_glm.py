# 适用版本 openai == 1.14.3
from openai import OpenAI
import config

client = OpenAI(api_key=config.GLM_API_KEY, base_url=config.GLM_BASE_URL)

stream = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "The following are multiple choice questions (with answers) about college physics\n\nFor which of the following thermodynamic processes is the increase in the internal energy of an ideal gas equal to the heat added to the gas?\nA. Constant temperature\nB. Constant volume\nC. Constant volume\nD. Adiabatic\nAnswer:B\n\nThe following are multiple choice questions (with answers) about college physics\n\nA refracting telescope consists of two converging lenses separated by 100 cm. The eye-piece lens has a focal length of 20 cm. The angular magnification of the telescope is\nA. 4\nB. 5\nC. 6\nD. 20\nAnswer:",
        }
    ],
    model="chatglm3-32b-v0.8-data",
    temperature=0,
    stream=True,
    max_tokens=3
)

for i, part in enumerate(stream):
    print(part.choices[0].delta.content or "", end="", flush=True)
    print(i)