# evaluation_utils.py

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import json
import time # 用于限速，避免过于频繁调用LLM

try:
    # 确保 llm_utils 和 config 被正确导入
    from llm_utils import generate_text, get_ollama_client
    from config import OLLAMA_EVALUATION_MODEL, MAX_EVAL_TEXT_INPUT_LEN_LLM, OLLAMA_EVALUATION_TEMPERATURE
except ImportError:
    print("错误 (evaluation_utils): 无法导入 llm_utils 或 config。请确保它们在正确的路径。")
    def generate_text(prompt: str, system_message: str = None, model_name: str = None, use_json_format: bool = False, temperature: Optional[float] = None) -> str:
        print(f"警告 (evaluation_utils): llm_utils.generate_text 未加载。评估功能将返回错误。")
        return '{"relevance_score": 0, "reasoning": "LLM功能未加载"}' if use_json_format else "错误: LLM功能未加载"
    OLLAMA_EVALUATION_MODEL = "mock_model"
    MAX_EVAL_TEXT_INPUT_LEN_LLM = 1000
    OLLAMA_EVALUATION_TEMPERATURE = 0.1


def parse_llm_evaluation_response(llm_response_str: str) -> Tuple[Optional[int], str]:
    """
    解析LLM返回的JSON格式评估响应。
    """
    if llm_response_str.startswith("错误:") or llm_response_str.startswith("Error:"):
        return None, f"LLM调用失败: {llm_response_str}"
    try:
        # 尝试去除markdown代码块（如果存在）
        if llm_response_str.strip().startswith("```json"):
            llm_response_str = llm_response_str.strip()[7:]
            if llm_response_str.strip().endswith("```"):
                llm_response_str = llm_response_str.strip()[:-3]
        elif llm_response_str.strip().startswith("```"): # 针对 ```\n{\n...}\n``` 格式
            llm_response_str = llm_response_str.strip()[3:]
            if llm_response_str.strip().endswith("```"):
                llm_response_str = llm_response_str.strip()[:-3]

        data = json.loads(llm_response_str.strip())
        score = data.get("relevance_score")
        reasoning = data.get("reasoning", "无理由提供")

        if isinstance(score, int) and 1 <= score <= 5:
            return score, str(reasoning)
        elif isinstance(score, (str, float)): # 尝试转换
            try:
                score_int = int(float(score))
                if 1 <= score_int <= 5:
                    return score_int, str(reasoning)
            except ValueError:
                pass
        
        return None, f"评分格式无效或超出范围 (1-5): {score}. 原始响应: {llm_response_str[:200]}"

    except json.JSONDecodeError:
        return None, f"LLM评估响应JSON解析失败。原始响应: {llm_response_str[:200]}"
    except Exception as e:
        return None, f"处理LLM评估响应时发生未知错误: {e}。原始响应: {llm_response_str[:200]}"


def evaluate_single_text_relevance_llm(
    query_text: str,
    retrieved_text_snippet: str,
    evaluation_model: str,
    query_main_character: Optional[str] = None,     # 新增参数: 用户问题中提及的核心人物
    character_name_for_context: Optional[str] = None # 新增参数: 当前交互的角色名 (例如 胡桃)
) -> Tuple[Optional[int], str]:
    """
    使用LLM评估单个检索到的文本片段与查询的相关性。
    """
    if not query_text.strip() or not retrieved_text_snippet.strip():
        return 0, "查询或文本片段为空"

    # 截断被评估的文本片段，如果太长
    truncated_snippet = retrieved_text_snippet
    if len(retrieved_text_snippet) > MAX_EVAL_TEXT_INPUT_LEN_LLM:
        truncated_snippet = retrieved_text_snippet[:MAX_EVAL_TEXT_INPUT_LEN_LLM] + " ...[内容已截断]"
        # print(f"评估 (LLM): 文本片段 '{retrieved_text_snippet[:30]}...' 已截断至 {MAX_EVAL_TEXT_INPUT_LEN_LLM} 字符进行评估。") # 可以取消注释用于调试

    # 动态构建Prompt中的人物信息部分
    query_char_mention_text = f"用户问题中的核心人物是：{query_main_character}。" if query_main_character else "请注意识别并核对用户问题中讨论的核心人物。"
    current_char_context_text = f"当前交互的角色是“{character_name_for_context}”。" if character_name_for_context else ""

    # 用户问题主题的提取可以是一个可选的、更高级的步骤，这里暂时留空或让LLM自行判断
    # query_topic_extraction_llm = "另一个LLM提取的主题关键词" # 这是一个可以后续添加的特性
    query_topic_text = "[请评估LLM自行判断用户问题的主题]" # 或者如果你有提取主题的逻辑，可以在这里填入

    prompt = f"""
请仔细评估下面提供的“背景知识片段”与“用户问题”的相关性。
{query_char_mention_text}
{current_char_context_text}
用户问题的主题是：{query_topic_text}

用户问题:
---
{query_text}
---

背景知识片段:
---
{truncated_snippet}
---

评估任务：
1.  **核心人物匹配：** “背景知识片段”中描述的主要事件或行为是否确实与“{query_main_character if query_main_character else "用户问题中的核心人物"}”直接相关？如果不是，请在理由中明确指出。
2.  **内容相关性与价值：**
    - 在确认核心人物匹配（或不匹配）的前提下，片段的内容是否有助于理解用户问题、提供解决问题的直接线索、解释背景原因、或帮助角色（例如，“{character_name_for_context if character_name_for_context else "当前交互角色"}”）构建有深度、有依据的回应？
    - 该片段提供的信息是否具有独特性或补充性，而非简单重复已知信息？（可选考虑，主要关注前一点）

请严格按照以下JSON格式输出你的评估，不要包含任何额外的解释或Markdown标记：
{{
  "relevance_score": <一个1到5之间的整数。5表示最高度相关且人物匹配。如果核心人物不匹配但内容仍有极少量间接参考价值，分数不应高于2。如果完全不相关或核心人物严重不匹配，则为1。>,
  "reasoning": "<简明扼要地解释你给出该分数的原因，务必包含对核心人物匹配情况的判断，并说明内容为何相关/不相关，以及其价值高低。>"
}}
"""
    system_message = "你是一个专业的文本相关性评估助手。请专注、客观地评估，并严格按照指定的JSON格式输出。"

    # 调用LLM，请求JSON格式输出
    llm_response = generate_text(
        prompt=prompt,
        system_message=system_message,
        model_name=evaluation_model,
        use_json_format=True, # 请求JSON输出
        temperature=OLLAMA_EVALUATION_TEMPERATURE # 使用配置的评估温度
    )

    return parse_llm_evaluation_response(llm_response)


def evaluate_retrieved_texts_relevance_llm(
    query_text: str,
    retrieved_texts: List[str],
    query_main_character: Optional[str] = None,     # 新增透传参数
    character_name_for_context: Optional[str] = None, # 新增透传参数
    max_texts_to_eval: int = 5,
    progress_callback: Optional[callable] = None
) -> List[Tuple[str, Optional[int], str]]:
    """
    使用LLM评估一组检索到的文本与给定查询的相关性。
    Args:
        query_text: 用户的原始查询。
        retrieved_texts: 从知识库中检索到的文本片段列表。
        query_main_character: 用户问题中讨论的核心人物。
        character_name_for_context: 当前进行交互的角色名。
        max_texts_to_eval: 最多评估多少条文本。
        progress_callback: 可选的回调函数，用于报告进度。
    Returns:
        一个元组列表，每个元组包含 (文本片段, 相关性得分 (1-5 或 None), 评估理由)。
    """
    if not query_text.strip() or not retrieved_texts:
        return []

    if not OLLAMA_EVALUATION_MODEL:
        print("警告 (LLM评估): OLLAMA_EVALUATION_MODEL 未配置。跳过LLM评估。")
        return [(text, None, "评估模型未配置") for text in retrieved_texts[:max_texts_to_eval]]

    results_with_scores: List[Tuple[str, Optional[int], str]] = []
    
    texts_to_process = retrieved_texts[:max_texts_to_eval]
    total_to_process = len(texts_to_process)

    for i, text_content in enumerate(texts_to_process):
        if progress_callback:
            progress_callback(i, total_to_process, desc=f"LLM评估进度: {i+1}/{total_to_process}")

        if not text_content or not text_content.strip():
            results_with_scores.append(("", 0, "空文本片段"))
            continue
        
        # print(f"LLM 正在评估片段 {i+1}/{total_to_process}: \"{text_content[:50]}...\"") # 可以取消注释用于调试
        score, reason = evaluate_single_text_relevance_llm(
            query_text=query_text,
            retrieved_text_snippet=text_content,
            evaluation_model=OLLAMA_EVALUATION_MODEL,
            query_main_character=query_main_character,         # 传递参数
            character_name_for_context=character_name_for_context # 传递参数
        )
        results_with_scores.append((text_content, score, reason))
        
        if i < total_to_process - 1:
            time.sleep(0.5) 

    if progress_callback:
         progress_callback(total_to_process, total_to_process, desc="LLM评估完成")

    return results_with_scores


if __name__ == '__main__':
    print("运行 evaluation_utils.py LLM评估测试...")
    try:
        get_ollama_client() 
        if not OLLAMA_EVALUATION_MODEL:
            raise ValueError("OLLAMA_EVALUATION_MODEL 未在config.py中配置，无法进行测试。")
        print(f"测试将使用评估模型: {OLLAMA_EVALUATION_MODEL}")
    except Exception as e:
        print(f"错误：测试 evaluation_utils 时初始化依赖失败: {e}")
        print("请确保Ollama服务正在运行，并且config.py中配置的评估模型可用。测试中止。")
        exit(1)

    sample_query = "胡桃，如果有一天璃月七星中的玉衡星刻晴认为往生堂占用了璃月重要的繁华位置，实在太不吉利，理应拆迁，你会怎么回答"
    sample_query_main_char = "刻晴" # 从问题中识别出的核心对话对象
    sample_current_char_context = "胡桃" # 当前交互的角色

    sample_retrieved_docs = [
        "在奥赛尔危机中，她（凝光）为了守护璃月，毅然牺牲了自己毕生心血建成的群玉阁...", # 应该被识别为人物不匹配
        "刻晴作为玉衡星，一直致力于璃月港的效率提升和革新，对不合理的传统持审慎态度。", # 高度相关且人物匹配
        "往生堂是璃月传承数百年的重要机构，负责逝者的仪轨，维系阴阳平衡，深受部分民众敬畏。", # 相关背景，但不直接涉及刻晴
        "钟离先生是往生堂的客卿，学识渊博，对璃月的历史和契约了如指掌。" # 间接相关
    ]

    print(f"\n查询: \"{sample_query}\"")
    print(f"查询中的核心人物: {sample_query_main_char}")
    print(f"当前交互角色: {sample_current_char_context}")
    print("待评估的检索文档:")
    for i, doc in enumerate(sample_retrieved_docs):
        print(f"  {i+1}. \"{doc[:60]}...\"")
    
    def dummy_progress_callback(current, total, desc=""):
        print(f"\r{desc} [{'*'*current}{'-'*(total-current)}] {current}/{total}", end="")
        if current == total:
            print()

    eval_results = evaluate_retrieved_texts_relevance_llm(
        sample_query, 
        sample_retrieved_docs,
        query_main_character=sample_query_main_char,
        character_name_for_context=sample_current_char_context,
        max_texts_to_eval=len(sample_retrieved_docs),
        progress_callback=dummy_progress_callback
    )
    
    print("\nLLM 相关性评估结果:")
    if eval_results:
        for i, (text, score, reason) in enumerate(eval_results):
            print(f"  文档 {i+1}: \"{text[:60]}...\"")
            print(f"    相关性得分: {score if score is not None else 'N/A'}/5")
            print(f"    评估理由: {reason}")
            print("-" * 20)
    else:
        print("评估未能产生结果。")
    
    print("\nevaluation_utils.py LLM评估测试完毕。")