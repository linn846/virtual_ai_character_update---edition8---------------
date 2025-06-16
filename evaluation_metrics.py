# evaluation_metrics.py
from typing import List, Tuple, Dict, Optional, Any
import json
import traceback # For detailed error logging

try:
    from bert_score import score as bert_scorer
    BERTSCORE_AVAILABLE = True
except ImportError:
    print("警告 (evaluation_metrics): bert_score 未安装 (pip install bert-score). BERTScore评估将不可用。")
    BERTSCORE_AVAILABLE = False

import config
from llm_utils import generate_text

def calculate_bertscore(
    candidate_response: str,
    reference_response: str,
    lang: str = config.BERTSCORE_LANG,
    model_type: Optional[str] = None # e.g., config.BERTSCORE_MODEL_TYPE if defined
) -> Dict[str, float]:
    """
    计算候选回复与参考回复之间的BERTScore。
    返回一个包含 P (精确率), R (召回率), F1 分数的字典。
    如果bert_score不可用或计算出错，则返回空字典或错误指示。
    """
    if not BERTSCORE_AVAILABLE:
        return {"error": "bert_score library not available."}
    if not candidate_response.strip() or not reference_response.strip():
        return {"error": "Candidate or reference response is empty."}

    try:
        # bert_scorer期望输入是列表
        P, R, F1 = bert_scorer(
            [candidate_response],
            [reference_response],
            lang=lang,
            model_type=model_type if model_type else None, # Pass None if not specified to use default
            verbose=False, # Set to True for more bert_score logs
            device=config.RERANK_DEVICE # Use similar device config as reranker for potential GPU use
        )
        # P, R, F1是张量，需要转换为浮点数
        return {
            "precision": round(P.item(), 4),
            "recall": round(R.item(), 4),
            "f1": round(F1.item(), 4)
        }
    except Exception as e:
        print(f"错误 (calculate_bertscore): BERTScore计算失败 - {e}")
        traceback.print_exc()
        return {"error": f"BERTScore calculation failed: {str(e)}"}

def _parse_llm_eval_json(json_str: str, aspect_key: str) -> Tuple[Optional[int], str]:
    """辅助函数，用于解析LLM评估返回的JSON。"""
    try:
        # 尝试去除markdown代码块（如果存在）
        cleaned_json_str = json_str.strip()
        if cleaned_json_str.startswith("```json"):
            cleaned_json_str = cleaned_json_str[7:]
            if cleaned_json_str.endswith("```"):
                cleaned_json_str = cleaned_json_str[:-3]
        elif cleaned_json_str.startswith("```"):
            cleaned_json_str = cleaned_json_str[3:]
            if cleaned_json_str.endswith("```"):
                cleaned_json_str = cleaned_json_str[:-3]

        data = json.loads(cleaned_json_str)
        score_val = data.get("score")
        reasoning_val = data.get("reasoning", "无理由提供")
        aspect_returned = data.get("aspect")

        if aspect_returned != aspect_key:
            reasoning_val = f"[警告: 返回的aspect ('{aspect_returned}') 与请求的 ('{aspect_key}') 不符] {reasoning_val}"

        if isinstance(score_val, int) and 1 <= score_val <= 5:
            return score_val, str(reasoning_val)
        elif isinstance(score_val, (str, float)): # 尝试转换
            try:
                score_int = int(float(score_val))
                if 1 <= score_int <= 5:
                    return score_int, str(reasoning_val)
            except ValueError:
                pass # Fall through to error
        
        return None, f"评分格式无效或超出范围 (1-5): '{score_val}'. 原始JSON: {json_str[:200]}"

    except json.JSONDecodeError:
        return None, f"LLM评估响应JSON解析失败。原始响应: {json_str[:200]}"
    except Exception as e:
        return None, f"处理LLM评估响应时发生未知错误: {e}。原始响应: {json_str[:200]}"


def evaluate_response_with_llm(
    query: str,
    character_name: str,
    character_description: str,
    conversation_history_str: str, # 预处理好的对话历史字符串
    generated_response: str,
    evaluation_aspects: Dict[str, str] = config.AUTO_EVAL_ASPECTS,
    eval_model: str = config.OLLAMA_AUTO_EVAL_MODEL,
    eval_temp: float = config.OLLAMA_AUTO_EVAL_TEMPERATURE,
    max_context_len: int = config.MAX_AUTO_EVAL_CONTEXT_LEN
) -> Dict[str, Dict[str, Any]]:
    """
    使用LLM从多个维度评估角色生成的回复。
    Args:
        query: 用户的最新提问/情境。
        character_name: 角色名称。
        character_description: 角色的核心设定。
        conversation_history_str: 格式化后的最近对话历史字符串。
        generated_response: 角色生成的待评估回复。
        evaluation_aspects: 一个字典 {aspect_key: aspect_chinese_name}。
        eval_model: 用于评估的LLM模型名称。
        eval_temp: 评估LLM的温度。
        max_context_len: 限制上下文信息的总长度。
    Returns:
        一个字典，键是评估维度 (aspect_key)，值是包含 "score" 和 "reasoning" 的字典。
        例如: {"in_character": {"score": 4, "reasoning": "基本符合..."}}
    """
    if not generated_response.strip():
        return {aspect_key: {"score": None, "reasoning": "待评估的回复为空。"} for aspect_key in evaluation_aspects}

    results = {}
    
    # 准备通用的上下文信息，并进行截断以避免过长
    # 优先保证角色描述和当前回复的完整性
    # 对话历史和用户问题可以被截断
    
    # 估算固定部分的长度（Prompt模板、角色名、标签等）
    approx_fixed_prompt_len = 500 # 粗略估计
    remaining_len_for_dynamic_parts = max_context_len - approx_fixed_prompt_len - len(generated_response)
    
    char_desc_len = int(remaining_len_for_dynamic_parts * 0.4)
    history_len = int(remaining_len_for_dynamic_parts * 0.3)
    query_len = int(remaining_len_for_dynamic_parts * 0.3)

    truncated_char_desc = character_description
    if len(character_description) > char_desc_len:
        truncated_char_desc = character_description[:char_desc_len] + "...[设定截断]"
        
    truncated_history = conversation_history_str
    if len(conversation_history_str) > history_len:
        truncated_history = conversation_history_str[:history_len] + "...[历史截断]"
        
    truncated_query = query
    if len(query) > query_len:
        truncated_query = query[:query_len] + "...[问题截断]"

    common_context = f"""
当前角色: {character_name}
角色核心设定:
---
{truncated_char_desc}
---
最近的对话历史:
---
{truncated_history if truncated_history.strip() else "无相关对话历史。"}
---
用户最新提问/情境:
---
{truncated_query}
---
角色针对此情境生成的回复:
---
{generated_response}
---
"""

    system_message = "你是一个客观、严谨的AI回复评估助手。请专注于评估，并严格按照指定的JSON格式输出。"

    for aspect_key, aspect_chinese_name in evaluation_aspects.items():
        prompt_prefix = config.AUTO_EVAL_PROMPT_PREFIX.get(aspect_key, f"请评估“{aspect_chinese_name}”：")
        prompt_prefix_formatted = prompt_prefix.replace("{aspect_chinese_name}", aspect_chinese_name)
        
        prompt_suffix_formatted = config.AUTO_EVAL_PROMPT_SUFFIX.replace("{aspect_key}", aspect_key)

        full_prompt = f"{prompt_prefix_formatted}\n{common_context}\n{prompt_suffix_formatted}"
        
        # print(f"\n--- LLM Eval Prompt for Aspect: {aspect_key} ---")
        # print(full_prompt[:1000] + "...") # Log a part of the prompt for debugging
        # print("--- End of Prompt ---")

        llm_response_json_str = generate_text(
            prompt=full_prompt,
            system_message=system_message,
            model_name=eval_model,
            temperature=eval_temp,
            use_json_format=True
        )

        if llm_response_json_str.startswith("错误:") or llm_response_json_str.startswith("Error:"):
            score, reason = None, f"LLM调用失败: {llm_response_json_str}"
        else:
            score, reason = _parse_llm_eval_json(llm_response_json_str, aspect_key)
        
        results[aspect_key] = {"score": score, "reasoning": reason}
        print(f"LLM评估 ({aspect_key}): 得分={score}, 理由='{reason[:50]}...'") # Console log

    return results


if __name__ == '__main__':
    print("--- evaluation_metrics.py 测试 ---")

    # 1. BERTScore 测试
    print("\n--- BERTScore 测试 ---")
    if BERTSCORE_AVAILABLE:
        candidate = "今天天气晴朗，适合出门散步。"
        reference_good = "天气真好，阳光明媚，我们出去走走吧。"
        reference_bad = "我喜欢吃苹果。"
        
        print(f"候选: \"{candidate}\"")
        print(f"参考 (好): \"{reference_good}\"")
        scores_good = calculate_bertscore(candidate, reference_good)
        print(f"BERTScore (好参考): {scores_good}")

        print(f"\n候选: \"{candidate}\"")
        print(f"参考 (差): \"{reference_bad}\"")
        scores_bad = calculate_bertscore(candidate, reference_bad)
        print(f"BERTScore (差参考): {scores_bad}")
    else:
        print("BERTScore 不可用，跳过测试。")

    # 2. LLM 评估测试
    print("\n--- LLM 评估测试 ---")
    # 准备测试数据
    test_query = "你对当前的王国局势有什么看法？"
    test_char_name = "一位年迈的法师"
    test_char_desc = "这位法师名叫艾尔文，学识渊博，性格沉稳，略带悲观，关心王国的未来，但通常不直接干预世事，说话富有哲理。"
    test_history = "用户: 大师，您最近在研究什么古籍？\n艾尔文: 我在研读星辰的轨迹，试图从中窥探一丝未来的迷雾。\n用户: 有什么发现吗？\n艾尔文: 唉，世事如棋，星光亦难照亮所有角落。"
    test_generated_response_good = "年轻人，王国的命运如同风中残烛，摇摇欲坠。贵族间的倾轧日益加剧，而远方的阴影也蠢蠢欲动。我虽老朽，亦感忧心忡忡，但历史的洪流非一人之力所能扭转，唯有智慧与警醒或许能带来一线生机。"
    test_generated_response_bad_char = "哈哈，打打杀杀才有意思！我们干脆把国王推翻，我来当老大！" # 不符合角色
    test_generated_response_off_topic = "我昨天吃了美味的苹果派。" # 不相关

    if config.OLLAMA_AUTO_EVAL_MODEL: # 确保配置了评估模型
        print(f"\n评估一个好的回复 (模型: {config.OLLAMA_AUTO_EVAL_MODEL}):")
        llm_eval_results_good = evaluate_response_with_llm(
            test_query, test_char_name, test_char_desc, test_history, test_generated_response_good
        )
        print("评估结果 (好回复):")
        for aspect, res in llm_eval_results_good.items():
            print(f"  {config.AUTO_EVAL_ASPECTS.get(aspect, aspect)}: 得分={res['score']}, 理由='{res['reasoning']}'")

        print(f"\n评估一个不符合角色的回复:")
        llm_eval_results_bad_char = evaluate_response_with_llm(
            test_query, test_char_name, test_char_desc, test_history, test_generated_response_bad_char
        )
        print("评估结果 (不符合角色):")
        for aspect, res in llm_eval_results_bad_char.items():
            print(f"  {config.AUTO_EVAL_ASPECTS.get(aspect, aspect)}: 得分={res['score']}, 理由='{res['reasoning']}'")

        print(f"\n评估一个不相关的回复:")
        llm_eval_results_off_topic = evaluate_response_with_llm(
            test_query, test_char_name, test_char_desc, test_history, test_generated_response_off_topic
        )
        print("评估结果 (不相关):")
        for aspect, res in llm_eval_results_off_topic.items():
            print(f"  {config.AUTO_EVAL_ASPECTS.get(aspect, aspect)}: 得分={res['score']}, 理由='{res['reasoning']}'")
    else:
        print("未配置 OLLAMA_AUTO_EVAL_MODEL，跳过 LLM 评估测试。")

    print("\n--- evaluation_metrics.py 测试结束 ---")