import os
import sys
from threading import Thread

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

sys.path.append(os.path.dirname(__file__))
from utils import print_param_stats, print_cuda_memory, print_model_precision

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

#promp = "这是一个视频：<|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|>，帮我总结一下这个视频讲述了什么内容"
#promp = "这是一个视频：<|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|>，帮我用英语总结一下这个视频讲述了什么内容"
#promp = "这是一个视频：<|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|>，帮我用日语总结一下这个视频讲述了什么内容"

# "15元毛巾评测",
#promp = "这是一个视频：<|sid_begin|><s_a_6037><s_b_2395><s_c_4831><|sid_end|>，帮我总结一下这个视频讲述了什么内容"

# "快手爆款：10分钟学会糖醋里脊做法"
#promp = "这是一个视频：<|sid_begin|><s_a_5993><s_b_7362><s_c_6071><|sid_end|>，帮我总结一下这个视频讲述了什么内容" # mean + l2
#promp = "这是一个视频：<|sid_begin|><s_a_5993><s_b_1638><s_c_5241><|sid_end|>，帮我总结一下这个视频讲述了什么内容" # mean
promp = "这是一个视频：<|sid_begin|><s_a_5993><s_b_7362><s_c_1871><|sid_end|>，帮我总结一下这个视频讲述了什么内容" # last + l2
promp = "这是一个视频：<|sid_begin|><s_a_5993><s_b_1638><s_c_368><|sid_end|>，帮我总结一下这个视频讲述了什么内容" # last

#promp = "这是一个新闻：<|sid_begin|><s_a_8038><s_b_4887><s_c_857><|sid_end|>，帮我总结一下这个新闻讲述了什么内容"
#promp = "这是SmartNews上的一则新闻：<|sid_begin|><s_a_2610><s_b_1283><s_c_2477><|sid_end|>，帮我总结一下这个新闻讲述了什么内容"

# "半導体大手が来年度の設備投資を増額へ。AI需要の拡大を背景に生産能力を強化する。",
# promp = "这是SmartNews上的一则新闻：<|sid_begin|><s_a_2779><s_b_5195><s_c_6071><|sid_end|>，帮我总结一下这个新闻讲述了什么内容"
# "気象庁は週末にかけて広い範囲で大雨の可能性があるとして早めの備えを呼びかけた。"
#promp = "这是SmartNews上的一则新闻：<|sid_begin|><s_a_4885><s_b_4716><s_c_6071><|sid_end|>，帮我总结一下这个新闻讲述了什么内容"

messages = [
    {"role": "user", "content": promp}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)
# -> added prefix & suffix to text

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
# -> model_inputs['input_ids'] is a tensor of shape [1, seq_len]

def generate_sync():
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

def generate_async():
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

    print("=== Thinking ===")
    for next in streamer:
        print(next, end="", flush=True)
        if next == "</think>":
            print("\n=== Content ===")
    print()
    thread.join()

generate_async()