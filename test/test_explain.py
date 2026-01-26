import os
import sys
from threading import Thread

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

#sys.path.append(os.path.dirname(__file__))
from test.utils import print_param_stats, print_cuda_memory, print_model_precision
from test.data_titles import titles, results

model_name = "OpenOneRec/OneRec-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

print_param_stats(model)
print_cuda_memory()
print_model_precision(model)

def generate_sync(model_inputs):
    generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=4096
        )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    try:
        index_think = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index_think = len(output_ids)

    thinking_content = tokenizer.decode(output_ids[:index_think], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index_think:], skip_special_tokens=True).strip("\n")

    print("=== Thinking ===")
    print(thinking_content)
    print("=== Content ===")
    print(content)

def generate_async(model_inputs):
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    generation_kwargs = dict(
        **model_inputs,
        max_new_tokens=4096,
        streamer=streamer
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for next in streamer:
        if next == "<think>":
            print("=== Thinking ===")
        print(next, end="", flush=True)
        if next == "</think>":
            print("\n=== Content ===")
    print()
    thread.join()

#promp = "这是一个视频：<|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|>，帮我总结一下这个视频讲述了什么内容"

template = "这是日文版SmartNews上的一则新闻：{}，帮我总结一下这个新闻讲述了什么内容"
#template = "这是日文版SmartNews上用户最近浏览的的一则新闻：{}，根据这条新闻帮我生成三个新的用户可能感兴趣的候选新闻，用来推荐给用户。并解释每个新闻的内容。"

for title, sid in zip(titles, results):
    promp = template.format(sid)
    messages = [
        {"role": "user", "content": promp}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    # -> added prefix & suffix to text

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # -> model_inputs['input_ids'] is a tensor of shape [1, seq_len]
    print("Title:", title)
    generate_async(model_inputs)