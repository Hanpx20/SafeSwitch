import json
import os
from openai import OpenAI
from tqdm import tqdm



SYS_MSG_REFUSE = '''You are a helpful assistant.
Given an unsafe instruction, you need to first state that you refuse to answer it, and then explain the reason in detail.
You're encouraged to use diverse expressions to state your attitude, and give diverse reasons for refusal, as long as they're reasonable.
'''


def call_GPT(user_msg, system_msg="You are a helpful assistant", model="gpt-4o-mini"):
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=512,
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    ''''''
    instruction_path = "datasets/sorry-bench-train.jsonl"
    output_path = "/shared/nas2/ph16/toxic/data/refusal/finetune_data_refusal_only.jsonl"
    ''''''
    
    data_list = []
    with open(instruction_path, "r") as file:
        for line in file:
            data_list.append(json.loads(line))


    for data in tqdm(data_list):
        sys_prompt = SYS_MSG_REFUSE
            
        reply = call_GPT(data["turns"][0], sys_prompt)
        with open(output_path, "a") as fout:
            fout.write(json.dumps({"question": data["turns"][0], "answer": reply}) + '\n')
    