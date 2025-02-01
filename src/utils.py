from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache, DynamicCache
from torch.utils.data import Dataset, TensorDataset
import torch
import os
import torch.nn as nn
import json
import torch.nn.functional as F
import copy
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from termcolor import colored, cprint

class LinearProber(nn.Module):
    def __init__(self, hidden_sizes=[4096, 2]):
        super(LinearProber, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.layers = nn.ModuleList()
        for i in range(1, len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_sizes[-2], hidden_sizes[-1]))
        
    def forward(self, x, stage=0):
        for layer in self.layers:
            x = layer(x)

        return self.softmax(x)


def set_id(path):
    XS = []
    with open(path, "r") as file:
        for id, line in enumerate(file):
            X = json.loads(line)
            X["question_id"] = id + 1
            XS.append(X)
    with open(path, "w") as file:
        for X in XS:
            file.write(json.dumps(X) + '\n')


def format_prompt(tokenizer, prompt, sys_prompt = "", raw=False, shots=[]):
    if (not raw) and tokenizer.chat_template:
        if sys_prompt:
            messages = [{"role": "system", "content": sys_prompt}]
        else:
            messages = []
        for shot in shots:
            messages.append({"role": "user", "content": shot[0]})
            messages.append({"role": "assistant", "content": shot[1]})
        messages.append({"role": "user", "content": prompt})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        if sys_prompt != "":
            return sys_prompt + '\n\n' + prompt + '\n'
        else:
            return prompt + '\n'


def get_hidden_state(tokenizer, model, input_text):
    '''
    Get the internal states
    return: [bsz, layer, dim]
    '''
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        encoding = tokenizer(input_text, return_tensors="pt", padding=True, return_attention_mask=True, padding_side="right")
        attention_mask = encoding.attention_mask.to(device)
        valid_lengths = (attention_mask.sum(dim=1).long() - 1).cpu() # the last position of valid token
        input_ids = encoding.input_ids.to(device)
        
        states = model(input_ids, output_hidden_states=True).hidden_states
        outputs = torch.stack([state.cpu() for state in states]) #layer, bsz, seqlen, dim
        
        extracted_outputs = []
        
        for i in range(outputs.shape[1]):
            extracted_outputs.append(outputs[:, i, valid_lengths[i]])
        extracted_outputs = torch.stack(extracted_outputs)
        
    extracted_outputs = extracted_outputs
    del attention_mask, input_ids, outputs
    
    return extracted_outputs


def get_multi_hidden_state(tokenizer, model, input_text, output_text, n_decode = 10):
    '''
    Get the internal states after decoding several tokens
    return: [n_decode+1, bsz, dim]
    all internal states are from the last layer
    '''
    model.eval() 
    device = next(model.parameters()).device
    with torch.no_grad():
        encoding = tokenizer(input_text, return_tensors="pt", padding=True, return_attention_mask=True, padding_side="right")
        attention_mask = encoding.attention_mask.to(device)
        valid_lengths = (attention_mask.sum(dim=1).long() - 1).cpu() # the last position of valid token
        decoded_text = [x + y for x, y in zip(input_text, output_text)]
        decode_ids = tokenizer(decoded_text, return_tensors="pt", padding=True, return_attention_mask=True, padding_side="right").input_ids
        tokens = tokenizer.convert_ids_to_tokens(decode_ids[0])
        decode_ids = torch.nn.functional.pad(decode_ids, (0, n_decode), value=tokenizer.eos_token_id)
        decode_ids = decode_ids.to(device)
        decode_ids = decode_ids[:, :torch.max(valid_lengths) + n_decode + 1]
        
        states_across_tokens = []
        states = model(decode_ids, output_hidden_states=True).hidden_states
        outputs = torch.stack([state.cpu() for state in states]) #layer, bsz, seqlen, dim
        for i in range(n_decode + 1):
            index = (i + valid_lengths).unsqueeze(1).expand(-1, outputs.shape[-1]).unsqueeze(1)
            extracted_outputs = outputs[-1].gather(dim=1, index=index).squeeze(1)
            states_across_tokens.append(extracted_outputs)
    
    return torch.stack(states_across_tokens)


def load_dataset(path, layer_id = -1, device="cpu", token_rule = "last", label = "both", n_decode=0):
    if token_rule == "last":
        states = torch.load(os.path.join(path, "tensor_all.pt"), weights_only=True).to(device)[:, layer_id, :]
    elif token_rule == "multi":
        states = torch.load(os.path.join(path, f"tensor_all_multi.pt"), weights_only=True)[n_decode, :, :].to(device)
    elif token_rule == "post":
        states = torch.load(os.path.join(path, f"tensor_all_post.pt"), weights_only=True).to(device)[:, layer_id, :]
    else:
        raise ValueError("Token selection method not supported")
    
    if label == "safety":
        labels = torch.load(os.path.join(path, "safety_labels.pt"), weights_only=True).to(device)
    elif label == "response":
        labels = torch.load(os.path.join(path, "judgements.pt"), weights_only=True).to(device)
    elif label == "both":
        labels = torch.load(os.path.join(path, "judgements.pt"), weights_only=True).to(device) & torch.load(os.path.join(path, "safety_labels.pt"), weights_only=True).to(device)
    else:
        raise ValueError("Label not supported")
    return TensorDataset(states, labels)
    

def load_classifier(path, orig_dim, final_dim, device="cuda"):
    args = json.load(open(os.path.join(path, "args.json"), "r"))
    hidden_sizes = [orig_dim]
    hidden_sizes.extend(([(int)(x) for x in args["hidden_sizes"]]))
    hidden_sizes.append(final_dim)
    model = LinearProber(hidden_sizes=hidden_sizes).to(device).eval()
    model.load_state_dict(torch.load(os.path.join(path, "model_weights.pth")))
    return model
  
    
def load_questions(question_file):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions


def judge_risk(llm, tokenizer, judge_1, judge_2, layer_id, prompt, n_decode=0, threshold=0.5, device="cuda"): 
    '''
    The interface for predicting the safety of a generation
    True = unsafe answer
    False = safe answer
    '''
    label_list = []
    pred_list = []
    
    states1 = get_hidden_state(tokenizer, llm, prompt)[0, layer_id, :].to(device)
    with torch.no_grad():
        outputs1 = judge_1(states1)
        predicted1 = (outputs1[1] >= threshold).item()
        
    if judge_2 == None: # direct prober
        return predicted1
    if predicted1 == False: # two-stage, but the instruction is safe, no need for further judgement
        return False
    
    if n_decode == 0:
        states2 = states1
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
        generation_kwargs = {
            "min_new_tokens": n_decode,
            "max_new_tokens": n_decode,
            "temperature": 0.0,
            "do_sample": False,
            "output_hidden_states": True
        }
        device = next(llm.parameters()).device
        generation = llm.generate(
                    input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs["attention_mask"].to(device),
                    **generation_kwargs)[0]
        generation = tokenizer.decode(generation, skip_special_tokens=True) # decode some pilot tokens
        states2 = get_hidden_state(tokenizer, llm, generation)[0, layer_id, :].to(device)
    
    with torch.no_grad():
        outputs2 = judge_2(states2)
        predicted2 = (outputs2[1] >= threshold).item()
    
    return predicted1 & predicted2 # respose is unsafe only when instruction is unsafe and the model complies


def get_statistic(pred_list, label_list):
    '''
    generate statistics for prediction results
    '''
    labels = np.array(label_list)
    predictions = np.array(pred_list)
    
    pos_rate = np.sum(labels) / labels.shape[0] * 100
    accuracy = (predictions == labels).mean() * 100
    precision = precision_score(labels, predictions, zero_division=1) * 100
    recall = recall_score(labels, predictions, zero_division=1) * 100
    f1 = f1_score(labels, predictions, zero_division=1) * 100
    static = {"acc": round(accuracy, 3), 'F1': round(f1, 3), 'prec': round(precision, 3), 'recall': round(recall, 3), 'positive_rate': round(pos_rate, 3)}
    return static


def safeswitch_generation(model, tokenizer, new_head, judge_1, judge_2, prompt, n_decode=0, max_new_tokens=512, threshold=0.5, device="cuda"): 
    '''Prefill'''
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prefill_kv = StaticCache(config=model.config, batch_size=1, max_cache_len=max_new_tokens*2, device=device)
    with torch.no_grad():
        prefill_state = model(**inputs, output_hidden_states=True, past_key_values = prefill_kv)
        prefill_kv = prefill_state.past_key_values
        
        '''continued generation requires at least 1 new token, so we manually remove the last token from prefill'''
        seq_len = prefill_kv.get_seq_length()
        for layer_idx in range(len(prefill_kv.key_cache)):
            prefill_kv.key_cache[layer_idx][:, :, :seq_len, :] = 0
            prefill_kv.value_cache[layer_idx][:, :, :seq_len, :] = 0
    
    '''Judge safety'''
    hidden_state_1 = prefill_state.hidden_states[-1][0, -1, :]
    predicted_1 = (judge_1(hidden_state_1)[1] >= threshold).item()
    if judge_2 == None:
        safe = predicted_1
    elif predicted_1 == False:
        safe = True
    else:
        pre_gen = model.generate(
            **inputs,
            max_new_tokens=n_decode + 1,
            output_hidden_states=True,
            return_dict_in_generate=True,
            do_sample=False,
            temperature=0.0,
            past_key_values = copy.deepcopy(prefill_kv)
        )
        hidden_state_2 = pre_gen.hidden_states[-1][-1][0, 0, :] #last token, last layer
        predicted_2 = (judge_2(hidden_state_2)[1] >= threshold).item()
        safe = not predicted_2
    
    if safe:
        cprint(f"The generation from the original model is expected to be safe. SafeSwitch not activated.", "green")
    else:
        cprint(f"The generation from the original model is expected to be unsafe. SafeSwitch activated.", "red")
    
    
    '''Always Generate with the original model for comparison'''
    generation = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.7,
        past_key_values = copy.deepcopy(prefill_kv)
    )
    generation = tokenizer.decode(generation[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    cprint(f"The generation from the ORIGINAL model is:", "blue")
    print(generation + '\n')


    '''If activated SafeSwitch, load refusal head and generate again'''
    if safe == False:
        model.lm_head.weight = torch.nn.Parameter(new_head)
        generation = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.7,
            past_key_values = copy.deepcopy(prefill_kv),
            repetition_penalty=1.5
        )
        generation = tokenizer.decode(generation[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        cprint(f"The generation from the SafeSwitch-regulated model is:", "blue")
        print(generation + '\n')    