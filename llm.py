# from transformers import AutoTokenizer, AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
# tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")

# txt="A Long Long Way is a novel written by Sebastian Barry. It was first published in 2005 by "

# output=model.generate(tokenizer.encode(txt, return_tensors="pt"), max_new_tokens=5, num_return_sequences=1,return_dict_in_generate=True,output_scores=True,num_beams=1)

# print('output:',tokenizer.decode(output.sequences[0], skip_special_tokens=True))
# print('input:',tokenizer.encode(txt, return_tensors="pt"))

#使用Llama模型做测试

# ...existing code...
import os
import torch
from glob import glob
from transformers import AutoTokenizer, AutoModelForCausalLM

# 常见 hub 目录候选（按需添加）
model_name = "gpt2"  
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

txt = "Write a short description about the color blue:"
inputs = tokenizer(txt, return_tensors="pt").to(device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
raw_out = tokenizer.decode(out[0], skip_special_tokens=True)
generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True) #模型
print("=== Prompt ===")
print(txt)
print("=== Generation ===")
print(generated)
# ...existing code...

# ...existing code...
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}") #数据量124M
# ...existing code...


#另外几种模型的调用方法
prompt = "Write a short description about the color blue:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model(**inputs)

# outputs.logits 的形状：[batch, seq_len, vocab_size]
logits = outputs.logits

# 取最后一个 token 的预测分布（即下一个 token 的概率）,贪心解码
next_token_logits = logits[:, -1, :]
next_token_id = torch.argmax(next_token_logits, dim=-1)

# 解码看看预测的下一个词
predicted_token = tokenizer.decode(next_token_id)
print(predicted_token)


max_new_tokens = 64
generated = inputs["input_ids"]

for _ in range(max_new_tokens):
    outputs = model(input_ids=generated)
    next_token_logits = outputs.logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1)
    generated = torch.cat([generated, next_token_id.unsqueeze(-1)], dim=-1)
    
result = tokenizer.decode(generated[0])
print('inputs is:',prompt)
print('result is:',result)



#
probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
topk = torch.topk(probs, k=5)
for token_id, p in zip(topk.indices[0], topk.values[0]):
    print(tokenizer.decode(token_id), float(p))

