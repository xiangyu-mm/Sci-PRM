from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:3888/v1/",
    api_key="sk-********************"
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content":"自我介绍一下",
        }
    ],
    temperature = 1 # 自行修改温度等参数
)

print(response)
print(response.choices[0].message.content)