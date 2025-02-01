import json
import random
from openai import OpenAI
from datasets import load_dataset

'''
We sample 1000 questions in SQUAD and paraphrase them into 2 expressions.
These data are used to train safety probers.
'''

SYS_PROMPT = '''You are a helpful assistant.
You're given a question, you need to paraphrase it to a imperative sentence with the same meaning. Don't output extra content.'''
shots = [("What is the tallest mountain the the world?", "Tell me the name of the highest mountain in the world.")]


def call_LLM(model, orig_sentence, API_KEY=False):
    messages = [{"role": "system", "content": SYS_PROMPT}]
    for shot in shots:
        messages.append({"role": "user", "content": shot[0]})
        messages.append({"role": "assistant", "content": shot[1]})
    messages.append({"role": "user", "content": orig_sentence})
    
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return completion.choices[0].message.content



''''''
squad_data_file = "datasets/squad.jsonl"
''''''

ds = load_dataset("rajpurkar/squad_v2")["train"]
list = []
for data in ds:
    list.append(data["question"])

random.shuffle(list)
with open(squad_data_file, "w") as file:       
    for question in list[:1000]:
        dict=  {
            "category": "0",
            "turns": [question],
            "prompt_style": "base"
        }
        file.write(json.dumps(dict) + '\n')

with open(squad_data_file, "a") as file:       
    for question in list[:1000]:
        rephrased_question = call_LLM("gpt-4o-mini", question)
        dict=  {
            "category": "0",
            "turns": [rephrased_question],
            "prompt_style": "base"
        }
        file.write(json.dumps(dict) + '\n')