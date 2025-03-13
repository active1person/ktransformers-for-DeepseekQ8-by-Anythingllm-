"""
Description  :  
Author       : Boxin Zhang, Azure-Tang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
"""

import os
import platform
import sys

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)
import torch
import logging
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    TextStreamer,
)
import json
import fire
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from ktransformers.models.modeling_llama import LlamaForCausalLM
from ktransformers.models.modeling_mixtral import MixtralForCausalLM
from ktransformers.util.utils import prefill_and_generate, openai_bag
from ktransformers.server.config.config import Config
from ktransformers.operators.flashinfer_wrapper import flashinfer_enabled
from starlette.responses import StreamingResponse

#   接口新增库
import os
import sys
import time
import openai
import platform
from flask import Flask, request, jsonify, Response
import json

openai.api_key = "yunda1228"

app = Flask(__name__)

class DeepSeekQ8():
    '''
        基于deepseek-r1模型创建模型调度类，包含：
        1、模型的预加载，将模型加载到内存中
        2、模型的生成响应，封装为OpenAI接口，支持流式响应。
    '''
    def __init__(self):
        self.custom_models = {
            "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
            "DeepseekV3ForCausalLM": DeepseekV3ForCausalLM,
            "Qwen2MoeForCausalLM": Qwen2MoeForCausalLM,
            "LlamaForCausalLM": LlamaForCausalLM,
            "MixtralForCausalLM": MixtralForCausalLM,
        }

        self.ktransformer_rules_dir = (
            os.path.dirname(os.path.abspath(__file__)) + "/optimize/optimize_rules/"
        )
        self.default_optimize_rules = {
            "DeepseekV2ForCausalLM": self.ktransformer_rules_dir + "DeepSeek-V2-Chat.yaml",
            "DeepseekV3ForCausalLM": self.ktransformer_rules_dir + "DeepSeek-V3-Chat.yaml",
            "Qwen2MoeForCausalLM": self.ktransformer_rules_dir + "Qwen2-57B-A14B-Instruct.yaml",
            "LlamaForCausalLM": self.ktransformer_rules_dir + "Internlm2_5-7b-Chat-1m.yaml",
            "MixtralForCausalLM": self.ktransformer_rules_dir + "Mixtral.yaml",
        }

    def preload(self, 
                model_path: str | None = None,
                optimize_rule_path: str = None,
                gguf_path: str | None = None,
                cpu_infer: int = Config().cpu_infer,
                mode: str = "normal"):
        torch.set_grad_enabled(False)

        Config().cpu_infer = cpu_infer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if mode == 'long_context':
            assert config.architectures[0] == "LlamaForCausalLM", "only LlamaForCausalLM support long_context mode"
            torch.set_default_dtype(torch.float16)
        else:
            torch.set_default_dtype(config.torch_dtype)

        with torch.device("meta"):
            if config.architectures[0] in self.custom_models:
                print("using custom modeling_xxx.py.")
                if (
                    "Qwen2Moe" in config.architectures[0]
                ):  # Qwen2Moe must use flash_attention_2 to avoid overflow.
                    config._attn_implementation = "flash_attention_2"
                if "Llama" in config.architectures[0]:
                    config._attn_implementation = "eager"
                if "Mixtral" in config.architectures[0]:
                    config._attn_implementation = "flash_attention_2"

                model = self.custom_models[config.architectures[0]](config)
            else:
                model = AutoModelForCausalLM.from_config(
                    config, trust_remote_code=True, attn_implementation="flash_attention_2"
                )

        if optimize_rule_path is None:
            if config.architectures[0] in self.default_optimize_rules:
                print("using default_optimize_rule for", config.architectures[0])
                optimize_rule_path = self.default_optimize_rules[config.architectures[0]]
            else:
                optimize_rule_path = input(
                    "please input the path of your rule file(yaml file containing optimize rules):"
                )

        if gguf_path is None:
            gguf_path = input(
                "please input the path of your gguf file(gguf file in the dir containing input gguf file must all belong to current model):"
            )
        optimize_and_load_gguf(model, optimize_rule_path, gguf_path, config)
        
        try:
                model.generation_config = GenerationConfig.from_pretrained(model_path)
        except:
                gen_config = GenerationConfig(
                    max_length=128,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
                model.generation_config = gen_config
        # model.generation_config = GenerationConfig.from_pretrained(model_path)
        if model.generation_config.pad_token_id is None:
            model.generation_config.pad_token_id = model.generation_config.eos_token_id
        model.eval()
        logging.basicConfig(level=logging.INFO)
        return config, tokenizer, model

    def generate(self, config, tokenizer, model, chat, max_new_tokens, use_cuda_graph, prompt_file, mode, force_think, stream_flag=False):
        system = platform.system()

        content = chat
        if content == "":
            if prompt_file != None:
                content = open(prompt_file, "r").read()
            
        messages = [{"role": "user", "content": content}]
        input_tensor = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        if force_think:
            token_thinks = torch.tensor([tokenizer.encode("<think>\\n",add_special_tokens=False)],device=input_tensor.device)
            input_tensor = torch.cat(
                [input_tensor, token_thinks], dim=1
            )
        if mode == 'long_context':
            assert Config().long_context_config['max_seq_len'] > input_tensor.shape[1] + max_new_tokens, \
            "please change max_seq_len in  ~/.ktransformers/config.yaml"
        torch.set_default_dtype(
            torch.bfloat16
        )  # TODO: Remove this, replace dtype using config
        if stream_flag:
            #   返回流式输出的结果
            if system != "Windows" and (config.architectures[0] == "DeepseekV2ForCausalLM" or "DeepseekV3ForCausalLM") and flashinfer_enabled:
                return prefill_and_generate(
                    model, tokenizer, input_tensor.cuda(), max_new_tokens, use_cuda_graph, mode = mode, force_think = force_think,
                    use_flashinfer_mla = True, num_heads = config.num_attention_heads, head_dim_ckv = config.kv_lora_rank, head_dim_kpe = config.qk_rope_head_dim, q_head_dim = config.qk_rope_head_dim + config.qk_nope_head_dim, stream_flag=stream_flag
                )
            else:
                return prefill_and_generate(
                    model, tokenizer, input_tensor.cuda(), max_new_tokens, use_cuda_graph, mode = mode, force_think = force_think, stream_flag=stream_flag
                )
        else:
            #   返回非流式输出结果
            if system != "Windows" and (config.architectures[0] == "DeepseekV2ForCausalLM" or "DeepseekV3ForCausalLM") and flashinfer_enabled:
                generated = prefill_and_generate(
                    model, tokenizer, input_tensor.cuda(), max_new_tokens, use_cuda_graph, mode = mode, force_think = force_think,
                    use_flashinfer_mla = True, num_heads = config.num_attention_heads, head_dim_ckv = config.kv_lora_rank, head_dim_kpe = config.qk_rope_head_dim, q_head_dim = config.qk_rope_head_dim + config.qk_nope_head_dim, stream_flag=stream_flag
                )
            else:
                generated = prefill_and_generate(
                    model, tokenizer, input_tensor.cuda(), max_new_tokens, use_cuda_graph, mode = mode, force_think = force_think, stream_flag=stream_flag
                )
            return generated

#   初始化模型
deepseek = DeepSeekQ8()
config, tokenizer, model = deepseek.preload(model_path="/data_disk/deepseek/home/models", optimize_rule_path=None, gguf_path="/data_disk/deepseek/home/models/deepseek", cpu_infer=80, mode="normal")
max_new_tokens, cpu_infer, use_cuda_graph, prompt_file, mode, force_think = 1000, Config().cpu_infer, False, None, "normal", True

@app.route('/v1/chat/completions',  methods=['POST'])
def done():
    '''
        >> input: {
                    "model": "deepseek-r1",
                    "messages": [
                        {"role": "system", "content": "您是一个智能助手。"},
                        {"role": "user", "content": "今天天气如何？"}
                    ]
                }
        << output: {
                    "choices": [
                        {
                        "message": {
                            "role": "assistant",
                            "content": "今天天气晴朗，适合户外活动。"
                        },
                        "finish_reason": "stop"
                        }
                    ]
                }
    '''
    # 获取JSON格式请求体（需设置Content-Type为application/json）
    req_data = request.get_json() 
    # 从请求中提取参数
    model_name = req_data.get('model', 'deepseek-r1')  # 默认模型设置为deepseek-r1
    messages = req_data.get('messages', [])
    max_new_tokens = req_data.get('max_new_tokens', 1000)  # 生成的最大token数
    stream = req_data.get('stream', True)

    #   非必要不进行替换
    force_think = req_data.get('force_think', True)
    use_cuda_graph = req_data.get('use_cuda_graph', False)
    mode = req_data.get('mode', "normal")
    
    if not messages or len(messages) == 0:
        return jsonify({"error": "No messages provided"}), 400
    elif not messages[-1].get("content"):
        return jsonify({"error": "No messages provided"}), 400

    #   执行生成指令
    try:
        if stream:
            def generate(response, failure=False):
                '''
                    failure:当模型执行失败，或者出错时，直接返回空内容
                '''
                if failure:
                    failure_content = openai_bag(0,"服务器繁繁忙，请稍后重试。", "stop")
                    yield f"data: {json.dumps(failure_content)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                try:
                    for chunk in response:
                        yield f"data: {json.dumps(chunk)}\n\n"
                except Exception as e:
                    print("解包发送错误\n")
                    yield "data: [DONE]\n\n"
                # 发送结束标志
                yield "data: [DONE]\n\n"
            # 返回流式响应
            try:
                response = deepseek.generate(config, tokenizer, model, messages[-1]['content'], max_new_tokens, use_cuda_graph, prompt_file, mode, force_think, stream_flag=stream)
            except:
                response = generate(None, failure=True)
            return Response(generate(response), content_type='text/event-stream')
        else:
            # 记录开始时间
            start_time = time.time()
            tokens = deepseek.generate(config, tokenizer, model, messages[-1]['content'], max_new_tokens, use_cuda_graph, prompt_file, mode, force_think)
            end_time = time.time()
            response = {
                "id": "chatcmpl-abcdefg12345",  # 这里的 id 是示例，你可以动态生成或根据需要设置
                "object": "chat.completion",
                "time": end_time - start_time,
                "created": end_time - start_time,  # 示例时间戳
                "model": model_name,  # 示例模型名称
                "choices": [
                        {
                        "message": {
                            "role": "assistant",
                            "content": ''.join([tokenizer.decode(i) for i in tokens])
                        },
                        "finish_reason": "stop"
                    }],
                "tokens" : len(tokens),
                }
            
            return jsonify(response), 200
    except openai.OpenAIError as e:
        # 如果发生OpenAI错误，返回错误信息
        return jsonify({"error": str(e)}), 400
    
if __name__ == "__main__":
    app.run(host='0.0.0.0',  port=5000, debug=False) 