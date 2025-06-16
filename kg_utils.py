# kg_utils.py
import json
import os
from typing import List, Dict, Optional, Tuple, Any # Ensure Tuple, Any are imported if used

import config # Use updated config
from llm_utils import generate_text, get_ollama_client # generate_text for rephrase and extract

# Import database_utils carefully to avoid circular dependencies if kg_utils is also imported by it.
# It seems database_utils._active_world_id is used, so a direct import is likely fine as per existing code.
import database_utils
# from database_utils import get_world_path, get_world_display_name, add_triples_to_kg, get_kg_triples_count_for_active_world, get_all_worldview_texts_for_active_world

def extract_triples_from_text_llm(text_content: str) -> List[List[str]]:
    """
    Uses LLM to extract knowledge triples (Subject, Predicate, Object) from text.
    """
    if not text_content.strip():
        return [] # Return empty list for empty input

    truncated_text = text_content
    if len(text_content) > config.MAX_KG_TEXT_INPUT_LEN_LLM:
        truncated_text = text_content[:config.MAX_KG_TEXT_INPUT_LEN_LLM] + "\n...[内容已截断]"
        print(f"KG提取 (LLM): 输入文本已从 {len(text_content)} 字符截断到 {len(truncated_text)} (最大允许 {config.MAX_KG_TEXT_INPUT_LEN_LLM})。")
    
    prompt = f"""
请从以下提供的文本中提取核心的知识三元组 (Subject, Predicate, Object)。
Subject 和 Object 应该是名词或名词短语，代表实体、概念或角色。
Predicate 应该是动词或动词短语，描述它们之间的关系或属性。
尽量使三元组简洁且信息丰富。忽略不重要的细节或不确定的信息。
输出格式要求严格为JSON数组，其中每个元素是一个包含三个字符串的数组 `["Subject", "Predicate", "Object"]`。
例如：`[["角色A", "是朋友", "角色B"], ["城市X", "位于", "区域Y"], ["技能Z", "属于", "角色A"]]`
如果文本中没有可提取的有效三元组，请返回一个空数组 `[]`。

文本内容如下：
---
{truncated_text}
---
提取的三元组 (JSON格式)：
"""
    system_message = "你是一个专门从文本中提取结构化知识三元组的AI助手。请严格按照指定的JSON格式输出。"
    
    llm_response_str = generate_text(
        prompt=prompt,
        system_message=system_message,
        model_name=config.OLLAMA_KG_EXTRACTION_MODEL,
        use_json_format=True # Request JSON output
    )

    if llm_response_str.startswith("错误:") or llm_response_str.startswith("Error:"): # Check for LLM error
        print(f"LLM提取三元组失败: {llm_response_str}")
        return []

    extracted_triples: List[List[str]] = []
    try:
        # Attempt to find a JSON block, possibly within markdown code fences
        json_block_match = None
        json_start_tag_strict = "```json"
        json_end_tag = "```"
        
        # Try ```json ... ```
        start_index_strict = llm_response_str.find(json_start_tag_strict)
        if start_index_strict != -1:
            end_index_strict = llm_response_str.rfind(json_end_tag, start_index_strict + len(json_start_tag_strict))
            if end_index_strict != -1:
                json_block_match = llm_response_str[start_index_strict + len(json_start_tag_strict) : end_index_strict]
        
        # If not found, try ``` ... ``` (generic code block that might contain JSON)
        if not json_block_match:
            json_start_tag_generic = "```"
            start_index_generic = llm_response_str.find(json_start_tag_generic)
            if start_index_generic != -1:
                # Ensure the end tag is after the start tag
                end_index_generic = llm_response_str.rfind(json_end_tag, start_index_generic + len(json_start_tag_generic))
                if end_index_generic != -1:
                     json_block_match = llm_response_str[start_index_generic + len(json_start_tag_generic) : end_index_generic]
        
        if json_block_match:
            # print(f"KG提取 (LLM): Found JSON block: {json_block_match[:100]}...") # Debug
            llm_response_str_to_parse = json_block_match.strip()
        else: 
            # Fallback: try to find first '[' and last ']' if no code block found
            first_bracket = llm_response_str.find('[')
            last_bracket = llm_response_str.rfind(']')
            if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
                llm_response_str_to_parse = llm_response_str[first_bracket : last_bracket + 1]
            else: # If no clear JSON structure, use the whole string (stripped)
                llm_response_str_to_parse = llm_response_str.strip()


        parsed_json = json.loads(llm_response_str_to_parse)
        if isinstance(parsed_json, list):
            for item in parsed_json:
                if isinstance(item, list) and len(item) == 3 and all(isinstance(s, str) for s in item):
                    cleaned_item = [s.strip() for s in item]
                    if any(s for s in cleaned_item): # Ensure not all are empty after stripping
                         extracted_triples.append(cleaned_item)
                    # else:
                    #     print(f"KG提取 (LLM)：跳过完全由空白组成的三元组: {item}") # Verbose
                # else:
                #     print(f"KG提取 (LLM)：跳过格式不正确的三元组: {item}") # Verbose
        # else:
        #     print(f"KG提取 (LLM)：LLM返回的JSON不是列表: {parsed_json}") # Verbose

    except json.JSONDecodeError:
        # print(f"KG提取 (LLM)：解析LLM返回的JSON失败。原始响应 (部分): {llm_response_str[:300]}") # Verbose
        # Try to find a list within a more complex JSON if the root is not a list
        try:
            outer_json = json.loads(llm_response_str.strip()) # Use original stripped response
            if isinstance(outer_json, dict) and "triples" in outer_json and isinstance(outer_json["triples"], list):
                # print("KG提取 (LLM): Found 'triples' key in a dict, parsing that list.") # Debug
                for item in outer_json["triples"]:
                    if isinstance(item, list) and len(item) == 3 and all(isinstance(s, str) for s in item):
                        cleaned_item = [s.strip() for s in item]
                        if any(s for s in cleaned_item):
                             extracted_triples.append(cleaned_item)
            # else:
            #     print(f"KG提取 (LLM): JSON解析失败, 且未找到 'triples' 键。原始响应 (部分): {llm_response_str[:300]}") # Verbose
        except json.JSONDecodeError: # Final fallback print if nested parse also fails
            print(f"KG提取 (LLM): 彻底解析LLM JSON失败。原始响应 (部分): {llm_response_str[:300]}")
    except Exception as e:
        print(f"KG提取 (LLM)：处理LLM响应时发生未知错误: {e}。原始响应 (部分): {llm_response_str[:300]}")
    
    return extracted_triples


def build_kg_for_active_world_from_json(progress_callback=None) -> str:
    """
    为当前活动世界从预定义的JSON文件构建或更新知识图谱。
    (Renamed from build_kg_for_active_world to be more specific about source)
    """
    current_active_world_id = database_utils._active_world_id 
    if not current_active_world_id:
        return "错误：没有活动的存储世界来构建知识图谱。"

    world_name = database_utils.get_world_display_name(current_active_world_id)
    world_path = database_utils.get_world_path(current_active_world_id)
    
    if not world_path:
        return f"错误：无法获取世界 '{world_name}' (ID: {current_active_world_id}) 的路径。"

    source_json_path = os.path.join(world_path, config.KNOWLEDGE_SOURCE_JSON_FILENAME)

    if progress_callback:
        progress_callback(0.1, desc=f"检查知识源文件: {config.KNOWLEDGE_SOURCE_JSON_FILENAME}")

    if not os.path.exists(source_json_path):
        if progress_callback: progress_callback(1.0, desc="错误：源文件未找到")
        return (f"错误：在世界 '{world_name}' 目录中未找到知识源文件 '{config.KNOWLEDGE_SOURCE_JSON_FILENAME}'。\n"
                f"请在该路径下创建此文件 ({source_json_path})，并按以下格式填入三元组数据:\n"
                f'{{\n  "triples": [\n    ["主体1", "谓词1", "客体1"],\n    ["主体2", "谓词2", "客体2"]\n  ]\n}}')

    print(f"开始从 '{source_json_path}' 为世界 '{world_name}' 构建知识图谱...")
    
    loaded_triples: List[List[str]] = []
    try:
        with open(source_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict) or "triples" not in data:
            if progress_callback: progress_callback(1.0, desc="格式错误")
            return f"错误：知识源文件 '{source_json_path}' 格式不正确。期望顶层是一个对象，且包含一个名为 'triples' 的键，其值为三元组列表。"
        
        raw_triples_list = data["triples"]
        if not isinstance(raw_triples_list, list):
            if progress_callback: progress_callback(1.0, desc="格式错误")
            return f"错误：知识源文件 '{source_json_path}' 中的 'triples' 字段必须是一个列表。"

        for i, item in enumerate(raw_triples_list):
            if isinstance(item, list) and len(item) == 3 and all(isinstance(s, str) and s.strip() for s in item): 
                loaded_triples.append([s.strip() for s in item]) 
            else:
                print(f"警告：跳过源文件 '{source_json_path}' 中格式不正确或包含空元素的三元组 (条目 {i+1}): {item}")
        
        if progress_callback:
            progress_callback(0.5, desc=f"已从JSON文件加载 {len(loaded_triples)} 条有效三元组")

    except json.JSONDecodeError as e:
        if progress_callback: progress_callback(1.0, desc="JSON解析失败")
        return f"错误：解析知识源文件 '{source_json_path}' 失败: {e}。请检查JSON格式。"
    except Exception as e:
        if progress_callback: progress_callback(1.0, desc="读取错误")
        return f"错误：读取或处理知识源文件 '{source_json_path}' 时发生未知错误: {e}"

    message = database_utils.add_triples_to_kg(loaded_triples, overwrite=True)
    final_count = database_utils.get_kg_triples_count_for_active_world()
    
    if progress_callback:
        progress_callback(1.0, desc="知识图谱构建完成！")

    if not loaded_triples and os.path.exists(source_json_path): 
         return f"警告：从知识源文件 '{source_json_path}' 未加载到有效的三元组。世界 '{world_name}' 的知识图谱已被清空。当前总数: {final_count}。"
    
    return f"世界 '{world_name}' 知识图谱构建完成。从 '{config.KNOWLEDGE_SOURCE_JSON_FILENAME}' 共加载并存储 {final_count} 条三元组。"


def auto_update_kg_from_worldview(progress_callback=None) -> str:
    """
    从当前活动世界的所有世界观文本中提取三元组并更新知识图谱。
    """
    current_active_world_id = database_utils._active_world_id
    if not current_active_world_id:
        return "错误：没有活动的存储世界来从世界观更新知识图谱。"

    world_name = database_utils.get_world_display_name(current_active_world_id)
    print(f"开始从世界 '{world_name}' 的世界观文本自动提取三元组并更新知识图谱...")

    if progress_callback:
        progress_callback(0.05, desc=f"正在加载世界 '{world_name}' 的所有世界观文本...")

    worldview_texts = database_utils.get_all_worldview_texts_for_active_world()
    if not worldview_texts:
        if progress_callback: progress_callback(1.0, desc="无世界观文本")
        return f"世界 '{world_name}' 中没有世界观文本可供提取三元组。"

    if progress_callback:
        progress_callback(0.1, desc=f"共找到 {len(worldview_texts)} 条世界观文本。开始提取...")

    all_extracted_triples: List[List[str]] = []
    total_texts = len(worldview_texts)
    for i, text_content in enumerate(worldview_texts):
        if not text_content or not text_content.strip():
            continue
        
        print(f"  处理世界观文本 {i+1}/{total_texts} (长度: {len(text_content)} chars)...")
        extracted = extract_triples_from_text_llm(text_content)
        if extracted:
            print(f"    提取到 {len(extracted)} 条三元组。")
            all_extracted_triples.extend(extracted)
        
        if progress_callback:
            # Calculate progress: 0.1 for loading, 0.8 for extraction, 0.1 for saving
            current_progress = 0.1 + (0.8 * (i + 1) / total_texts)
            progress_callback(current_progress, desc=f"提取进度: {i+1}/{total_texts}")
    
    if not all_extracted_triples:
        if progress_callback: progress_callback(1.0, desc="未提取到新三元组")
        return f"未能从世界 '{world_name}' 的世界观文本中提取到任何新的三元组。"

    if progress_callback:
        progress_callback(0.9, desc=f"共提取 {len(all_extracted_triples)} 条三元组，正在添加到知识图谱...")

    # Add triples without overwriting existing ones from other sources (like JSON)
    # The add_triples_to_kg function should handle deduplication internally.
    message = database_utils.add_triples_to_kg(all_extracted_triples, overwrite=False)
    
    final_count = database_utils.get_kg_triples_count_for_active_world()
    if progress_callback:
        progress_callback(1.0, desc="更新完成!")
    
    return f"从世界 '{world_name}' 的世界观文本中自动提取并更新知识图谱完成。\n{message} 当前总数: {final_count}。"


def rephrase_triples_for_prompt_llm(
    triples: List[List[str]], 
    character_name: Optional[str] = None, 
    query_context: Optional[str] = None
) -> str:
    """
    使用LLM将S-P-O三元组列表改写成更自然的描述性语句。
    """
    if not triples:
        return "" # Return empty if no triples

    # Limit the number of triples to avoid overly long prompts for rephrasing
    triples_to_rephrase = triples[:config.MAX_KG_REPHRASE_INPUT_TRIPLES]
    
    # Format triples into a string list
    triples_str_list = []
    for s, p, o in triples_to_rephrase:
        triples_str_list.append(f"- \"{s}\" {p} \"{o}\".")
    
    triples_for_prompt = "\n".join(triples_str_list)

    context_info = ""
    if character_name:
        context_info += f"这些信息与角色“{character_name}”有关。"
    if query_context:
        context_info += f" 当前的对话或情境是：“{query_context}”。"
    if context_info:
        context_info = f"背景参考信息：{context_info}\n"

    prompt = f"""
{context_info}
以下是一些以“主语-谓语-宾语”形式存在的结构化知识点，它们可能与角色“{character_name}”当前面临的情境“{query_context if query_context else '某个一般情境'}”有关：
---
{triples_for_prompt}
---
请你将上述这些独立的知识点巧妙地融合，改写成一段或几段连贯、自然的描述性文字。
这段文字应该读起来像是角色“{character_name}”在思考或阐述背景设定的一部分，**能够自然地衔接或支撑角色对当前情境的理解或回应。**
-如果信息量大，可以适当分段。如果信息较少，一段即可。
+请专注于信息的准确传达和流畅表达，如果可能，可以尝试用更符合“{character_name}”已知风格的间接方式来组织语言（例如，如果角色喜欢用比喻，可以尝试融入；如果角色说话简洁，则保持简洁）。
请直接输出改写后的自然语言描述，不要添加任何额外的解释、标题或引言。
改写后的描述：
"""
    system_message = "你是一位优秀的文本编辑和知识整合助手，擅长将结构化的信息转化为流畅自然的叙述。"
    
    rephrased_text = generate_text(
        prompt,
        system_message=system_message,
        model_name=config.OLLAMA_KG_REPHRASE_MODEL,
        temperature=config.OLLAMA_KG_REPHRASE_TEMPERATURE
    )

    if rephrased_text.startswith("错误:") or rephrased_text.startswith("Error:") or not rephrased_text.strip():
        print(f"LLM改写KG三元组失败: {rephrased_text}. 将返回原始三元组的简单罗列。")
        # Fallback: simple listing if LLM fails
        fallback_list = [f"{s} {p} {o}。" for s, p, o in triples_to_rephrase]
        return " ".join(fallback_list)
        
    return rephrased_text.strip()


# __main__ part of kg_utils.py (if it exists for testing)
if __name__ == '__main__':
    print("运行知识图谱工具 (kg_utils.py) 示例...")
    
    # Ensure config is loaded and client can be initialized for tests
    try:
        get_ollama_client() 
        print("Ollama LLM 客户端连接正常。")
    except Exception as e:
        print(f"LLM客户端初始化失败: {e}. 部分测试功能可能受影响。")

    # --- Test KG Rephrasing ---
    print("\n--- 测试 KG 三元组改写功能 (rephrase_triples_for_prompt_llm) ---")
    sample_triples_for_rephrase = [
        ["艾莉亚·史塔克", "是", "史塔克家族成员"],
        ["艾莉亚·史塔克", "拥有武器", "缝衣针"],
        ["艾莉亚·史塔克", "擅长", "剑术和潜行"],
        ["君临城", "是", "七大王国的首都"],
    ]
    if config.OLLAMA_KG_REPHRASE_MODEL:
        print(f"测试三元组: {sample_triples_for_rephrase}")
        rephrased_output = rephrase_triples_for_prompt_llm(
            sample_triples_for_rephrase, 
            character_name="艾莉亚·史塔克",
            query_context="艾莉亚在君临城的行动"
            )
        print(f"LLM改写后的文本:\n---\n{rephrased_output}\n---")
    else:
        print("未配置 OLLAMA_KG_REPHRASE_MODEL，跳过 KG 改写测试。")

    # --- Test Auto Update from Worldview (needs a test world setup) ---
    # This part is more involved as it requires setting up a test world with worldview texts.
    # For now, the main test path for this will be through the Gradio UI.
    print("\n--- 自动从世界观更新KG功能 (auto_update_kg_from_worldview) ---")
    print("此功能主要通过 Gradio UI 进行测试，因为它依赖于活动的测试世界和其中的世界观数据。")
    print("你可以在Gradio的'知识图谱构建'标签页找到手动触发此功能的按钮。")
    
    # (Original __main__ for JSON loading can remain if needed, but should be clearly separated)
    # ... (original __main__ for JSON loading can be here or in a separate test script) ...
    print("\n知识图谱工具 (kg_utils.py) 示例运行完毕。")