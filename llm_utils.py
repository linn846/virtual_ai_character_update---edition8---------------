# llm_utils.py
from typing import Optional, List, Dict # 根据实际使用的类型添加
import ollama
import re
import json
import config  # <--- 添加这一行
from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_SUMMARY_MODEL,
    OLLAMA_KG_EXTRACTION_MODEL, MAX_SUMMARY_INPUT_LEN_LLM,
    OLLAMA_EVALUATION_MODEL,
    OLLAMA_DEFAULT_TEMPERATURE, # 新增导入
    OLLAMA_DEFAULT_TOP_P        # 新增导入
)

_ollama_client = None

def get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        print(f"正在初始化 Ollama LLM 客户端以连接到: {OLLAMA_BASE_URL}")
        try:
            _ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
            list_response = _ollama_client.list()
            ollama_models_raw = []
            if isinstance(list_response, dict) and 'models' in list_response:
                ollama_models_raw = list_response['models']
            elif isinstance(list_response, list):
                 ollama_models_raw = list_response
            print(f"成功连接到 Ollama，地址为: {OLLAMA_BASE_URL}")

            available_model_names = []
            if ollama_models_raw:
                for model_item_raw in ollama_models_raw:
                    item_name = None
                    if isinstance(model_item_raw, dict):
                        item_name = model_item_raw.get('name')
                    elif hasattr(model_item_raw, 'name'):
                        item_name = model_item_raw.name
                    if item_name:
                        available_model_names.append(item_name)

            if not available_model_names and ollama_models_raw:
                print("警告：收到Ollama模型列表，但未能从中提取任何有效名称。")
            elif not ollama_models_raw:
                 print("警告：Ollama 返回的可用模型列表为空。")

            models_to_check_map = {
                "主 LLM (扮演/拓展)": OLLAMA_MODEL,
                "总结 LLM": OLLAMA_SUMMARY_MODEL,
                "知识提取 LLM": OLLAMA_KG_EXTRACTION_MODEL,
                "评估 LLM": OLLAMA_EVALUATION_MODEL,
                "上下文剪裁LLM": getattr(config, 'OLLAMA_CONTEXT_SNIPPING_MODEL', OLLAMA_MODEL),
                "角色特质提取LLM": getattr(config, 'OLLAMA_TRAIT_EXTRACTION_MODEL', OLLAMA_MODEL),
                "历史处理LLM": getattr(config, 'OLLAMA_HISTORY_PROCESSING_MODEL', OLLAMA_SUMMARY_MODEL),
                "KG改写LLM": getattr(config, 'OLLAMA_KG_REPHRASE_MODEL', OLLAMA_SUMMARY_MODEL),
            }
            model_usage_counts = {}
            for desc, model_name_val_or_default in models_to_check_map.items():
                # Resolve model name if it's a getattr result or direct config value
                model_name = model_name_val_or_default
                model_usage_counts[model_name] = model_usage_counts.get(model_name, []) + [desc]

            unique_model_names_to_check = set(m for m in model_usage_counts.keys() if m)
            all_specified_models_available_flag = True

            if not available_model_names and unique_model_names_to_check:
                print(f"警告：无法从Ollama获取可用模型列表。配置的 LLM ({', '.join(unique_model_names_to_check)}) 的可用性未知。")
                all_specified_models_available_flag = False
            else:
                for model_name_cfg in unique_model_names_to_check:
                    found = any(
                        name_from_ollama == model_name_cfg or \
                        name_from_ollama.startswith(model_name_cfg + ":")
                        for name_from_ollama in available_model_names
                    )
                    if not found:
                        missing_model_descs = model_usage_counts.get(model_name_cfg, ["未知用途"])
                        print(f"警告：配置用于 '{', '.join(missing_model_descs)}' 的 LLM '{model_name_cfg}' 在 Ollama 中未找到。")
                        all_specified_models_available_flag = False
                    else:
                        available_model_descs = model_usage_counts.get(model_name_cfg, [])
                        print(f"配置用于 '{', '.join(available_model_descs)}' 的 LLM '{model_name_cfg}' 在 Ollama 中可用或部分匹配。")

            if not all_specified_models_available_flag:
                print(f"  Ollama 中当前可用的模型 (可能不完整或包含tag): {available_model_names if available_model_names else '列表为空或获取失败'}")
                missing_configured_models = [
                    model_name_cfg for model_name_cfg in unique_model_names_to_check
                    if not any(name_from_ollama == model_name_cfg or name_from_ollama.startswith(model_name_cfg + ":") for name_from_ollama in available_model_names)
                ]
                if missing_configured_models:
                    print(f"  请确保已拉取缺失的配置 LLM (例如: ollama pull {missing_configured_models[0]}) 或更新 config.py。")
                elif not available_model_names and unique_model_names_to_check:
                     print(f"  请检查Ollama服务状态和模型列表获取是否正常。")

        except Exception as e:
            print(f"连接到 Ollama 或列出模型时发生错误: {e}")
            _ollama_client = None
            raise
    return _ollama_client


def remove_think_tags(text: str) -> str:
    if text is None:
        return ""
    cleaned_text = re.sub(r"<think>(.*?)</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned_text.strip()


def generate_text(
    prompt: str,
    system_message: str = None,
    model_name: str = None,
    use_json_format: bool = False,
    temperature: Optional[float] = None, # 新增
    top_p: Optional[float] = None,       # 新增
    # Можно добавить top_k, repeat_penalty и т.д. если нужно
) -> str:
    client = get_ollama_client()
    if not client:
        return "错误：Ollama 客户端不可用。"

    target_model = model_name if model_name else OLLAMA_MODEL
    if not target_model:
        return "错误：未在 config.py 中配置有效的 LLM 模型 (OLLAMA_MODEL 或指定的 model_name)。"

    messages = []
    if system_message:
        messages.append({'role': 'system', 'content': system_message})
    messages.append({'role': 'user', 'content': prompt})

    # --- 构建请求参数 ---
    request_params = {
        'model': target_model,
        'messages': messages
    }
    if use_json_format:
        request_params['format'] = 'json'

    options = {}
    final_temperature = temperature if temperature is not None else getattr(config, 'OLLAMA_DEFAULT_TEMPERATURE', None)
    final_top_p = top_p if top_p is not None else getattr(config, 'OLLAMA_DEFAULT_TOP_P', None)

    if final_temperature is not None:
        options['temperature'] = float(final_temperature)
    if final_top_p is not None:
        options['top_p'] = float(final_top_p)
    # Add other Ollama options here if needed, e.g., options['top_k'] = 40

    if options:
        request_params['options'] = options
        print(f"正在使用模型 '{target_model}' 生成文本 (JSON模式: {use_json_format}, 选项: {options})...")
    else:
        print(f"正在使用模型 '{target_model}' 生成文本 (JSON模式: {use_json_format}, 使用模型默认选项)...")


    try:
        response = client.chat(**request_params)
        generated_content = response['message']['content']
        print(f"模型 '{target_model}' 响应长度 (原始): {len(generated_content)}")

        if not use_json_format or "<think>" in generated_content or "</think>" in generated_content:
            cleaned_content = remove_think_tags(generated_content)
            # print(f"模型 '{target_model}' 响应长度 (清理后): {len(cleaned_content)}") # Redundant if no change
        else:
            cleaned_content = generated_content.strip()
            # print(f"模型 '{target_model}' (JSON模式) 响应长度 (strip后): {len(cleaned_content)}")

        return cleaned_content

    except ollama.ResponseError as e:
        error_message_detail = str(e)
        status_code_info = f" (Status Code: {e.status_code})" if hasattr(e, 'status_code') else ""
        if "model not found" in error_message_detail.lower() or \
           (hasattr(e, 'status_code') and e.status_code == 404) or \
           "pull model" in error_message_detail.lower():
             error_message = (f"错误：LLM 模型 '{target_model}' 在 Ollama 服务器上未找到{status_code_info}。"
                              f"请拉取该模型 (例如: `ollama pull {target_model}`) 或检查其名称。")
        elif use_json_format and ("json: non-object" in error_message_detail or "invalid json" in error_message_detail.lower()):
            error_message = (f"错误 (JSON模式)：LLM 模型 '{target_model}' 未能生成有效的JSON响应。请检查Prompt是否正确引导JSON输出，或模型是否支持。详情: {error_message_detail}{status_code_info}")
        else:
             error_message = f"Error: 调用 Ollama API 时发生响应错误 (模型: {target_model}): {error_message_detail}{status_code_info}"
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"Error: 调用 Ollama API 时发生未知错误 (模型: {target_model}): {e}"
        print(error_message)
        return error_message

def generate_summary(text_to_summarize: str, max_input_len: int = None, target_model: Optional[str] = None) -> str:
    # target_model can be passed for specific summary tasks like chunk summaries
    actual_max_input_len = max_input_len if max_input_len is not None else MAX_SUMMARY_INPUT_LEN_LLM

    if not text_to_summarize or not text_to_summarize.strip():
        return ""

    truncated_text = text_to_summarize
    if len(text_to_summarize) > actual_max_input_len:
        truncated_text = text_to_summarize[:actual_max_input_len] + "\n... [内容过长已截断]"
        print(f"用于总结的文本已从 {len(text_to_summarize)} 字符截断到 {actual_max_input_len} 字符。")

    prompt = f"""请为以下文本生成一个简洁但尽量保留关键信息的摘要。概要应该捕捉文本的核心思想和关键细节，并且自身是一段连贯的文字。不要添加任何额外的解释或评论，直接给出概要。

原始文本：
---
{truncated_text}
---

概要："""
    system_message = "你是一个专业的文本摘要助手，擅长提炼核心信息并生成高质量的摘要。"

    summary_model_to_use = target_model if target_model else OLLAMA_SUMMARY_MODEL
    if not summary_model_to_use:
        print("警告：未配置总结模型 (OLLAMA_SUMMARY_MODEL 或指定的 target_model)，将尝试使用 OLLAMA_MODEL 进行总结。")
        summary_model_to_use = OLLAMA_MODEL
    if not summary_model_to_use:
        error_msg = "错误：总结功能需要配置有效的 LLM 模型。"
        print(error_msg)
        return error_msg

    # Summaries usually benefit from lower temperature for factuality
    summary_temperature = getattr(config, 'OLLAMA_SUMMARY_TEMPERATURE', 0.3) # Example, could be in config

    summary = generate_text(
        prompt,
        system_message=system_message,
        model_name=summary_model_to_use,
        temperature=summary_temperature # Pass lower temperature for summaries
    )

    if summary.startswith("错误:") or summary.startswith("Error:"):
        print(f"使用 LLM 生成概要失败。将返回原始文本的前N个字符作为备用。错误: {summary}")
        fallback_len = min(len(text_to_summarize), 500)
        return text_to_summarize[:fallback_len] + ("..." if len(text_to_summarize) > fallback_len else "")

    return summary.strip()