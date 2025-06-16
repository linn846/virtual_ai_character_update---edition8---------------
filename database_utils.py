# database_utils.py
from utils import load_kg_entities_to_jieba_dict_for_world
import os
import json
import shutil
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import faiss
import pickle
import jieba
from rank_bm25 import BM25Okapi
import logging

from config import (
    DATA_DIR, WORLDS_METADATA_FILE,
    WORLD_CHARACTERS_DB_FILENAME, WORLD_WORLDVIEW_TEXTS_FILENAME,
    WORLD_WORLDVIEW_FAISS_INDEX_FILENAME, WORLD_KNOWLEDGE_GRAPH_FILENAME,
    CHUNK_SIZE, CHUNK_OVERLAP,
    MAX_CHAR_FULL_DESC_LEN, OLLAMA_EMBEDDING_MODEL_NAME,
    WORLD_WORLDVIEW_BM25_FILENAME,
    BM25_SEARCH_TOP_K_HYBRID, KEYWORD_SEARCH_TOP_K_HYBRID,
    OLLAMA_TRAIT_EXTRACTION_MODEL # 新增导入，用于角色特质提取模型
)
from embedding_utils import get_embedding, get_embeddings, get_model_embedding_dimension
from llm_utils import generate_summary, generate_text # 确保 generate_text 也被导入
from text_splitting_utils import semantic_text_splitter

jieba.setLogLevel(logging.INFO)

_active_world_id: Optional[str] = None
_worlds_metadata: Dict[str, str] = {}


def _load_worlds_metadata():
    global _worlds_metadata
    if os.path.exists(WORLDS_METADATA_FILE):
        try:
            with open(WORLDS_METADATA_FILE, 'r', encoding='utf-8') as f:
                _worlds_metadata = json.load(f)
        except json.JSONDecodeError:
            print(f"警告：世界元数据文件 '{WORLDS_METADATA_FILE}' 解析失败，将使用空元数据。")
            _worlds_metadata = {}
        except Exception as e:
            print(f"警告：加载世界元数据文件 '{WORLDS_METADATA_FILE}' 时发生未知错误: {e}。将使用空元数据。")
            _worlds_metadata = {}
    else:
        _worlds_metadata = {}

def _save_worlds_metadata():
    try:
        with open(WORLDS_METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(_worlds_metadata, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"错误：保存世界元数据到 '{WORLDS_METADATA_FILE}' 失败: {e}")

def get_world_path(world_id: str) -> Optional[str]:
    if not world_id or not isinstance(world_id, str):
        return None
    return os.path.join(DATA_DIR, world_id)

def _initialize_world_data(world_id: str):
    world_path = get_world_path(world_id)
    if not world_path: return

    os.makedirs(world_path, exist_ok=True)

    char_file = os.path.join(world_path, WORLD_CHARACTERS_DB_FILENAME)
    if not os.path.exists(char_file):
        with open(char_file, 'w', encoding='utf-8') as f:
            json.dump([], f)

    wv_texts_file = os.path.join(world_path, WORLD_WORLDVIEW_TEXTS_FILENAME)
    if not os.path.exists(wv_texts_file):
        with open(wv_texts_file, 'w', encoding='utf-8') as f:
            json.dump({}, f)

    faiss_index_file = os.path.join(world_path, WORLD_WORLDVIEW_FAISS_INDEX_FILENAME)
    if not os.path.exists(faiss_index_file):
        try:
            dimension = get_model_embedding_dimension()
            index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
            faiss.write_index(index, faiss_index_file)
            print(f"为世界 '{world_id}' 初始化了维度为 {dimension} 的 FAISS 索引。")
        except Exception as e:
            print(f"错误：为世界 '{world_id}' 初始化 FAISS 索引失败: {e}")

    kg_file = os.path.join(world_path, WORLD_KNOWLEDGE_GRAPH_FILENAME)
    if not os.path.exists(kg_file):
        with open(kg_file, 'w', encoding='utf-8') as f:
            json.dump({"triples": []}, f)

_load_worlds_metadata()

def add_world(world_id: str, display_name: str) -> str:
    global _worlds_metadata
    if not world_id or not display_name:
        return "错误：世界ID和显示名称不能为空。"
    if world_id in _worlds_metadata:
        return f"错误：世界ID '{world_id}' 已存在。"
    if not all(c.isalnum() or c in ['_', '-'] for c in world_id):
         return "错误：世界ID只能包含字母、数字、下划线和连字符。"

    _worlds_metadata[world_id] = display_name
    _save_worlds_metadata()
    _initialize_world_data(world_id)
    return f"世界 '{display_name}' (ID: {world_id}) 已添加。"

def get_available_worlds() -> Dict[str, str]:
    return _worlds_metadata.copy()

def switch_active_world(world_id: Optional[str]) -> bool:
    global _active_world_id
    # 确保导入 os
    import os
    # 确保导入或定义了 load_kg_entities_to_jieba_dict_for_world
    # from .utils_module import load_kg_entities_to_jieba_dict_for_world # 示例

    if world_id is None:
        # 如果有必要，在这里可以清除或重置Jieba的自定义词典
        # jieba.initialize() # 强制重新初始化（如果需要完全清除）
        # 或者维护一个已加载词典的列表，并逐个卸载（jieba没有直接的卸载API，通常是重新加载默认词典）
        # 简单起见，我们先不处理取消激活时的词典卸载，因为Jieba词典是全局的。
        # 如果严格要求每个世界词典独立且不冲突，可能需要更复杂的Jieba管理。
        _active_world_id = None
        print("已取消活动世界。")
        return True

    if world_id in _worlds_metadata:
        world_path_check = get_world_path(world_id) # 先检查路径

        if world_path_check and os.path.isdir(world_path_check):
            _active_world_id = world_id # 确认路径有效后再正式设置活动ID
            print(f"已激活世界: '{get_world_display_name(world_id)}' (ID: {world_id})")
            _initialize_world_data(world_id) # 确保世界数据文件和目录存在

            # 在初始化数据之后，加载该世界的自定义实体词典
            # 确保 load_kg_entities_to_jieba_dict_for_world 在此作用域可见
            # 如果它在同一个文件，直接调用即可
            # 如果在其他文件，例如 utils.py，需要 from utils import load_kg_entities_to_jieba_dict_for_world
            try:
                # 将 load_kg_entities_to_jieba_dict_for_world 的定义放在本文件或确保正确导入
                # 全局 _loaded_world_dictionaries 应该在 load_kg_entities_to_jieba_dict_for_world 函数内部或其模块级别管理
                load_kg_entities_to_jieba_dict_for_world(world_id, world_path_check)
            except NameError:
                print(f"警告：函数 'load_kg_entities_to_jieba_dict_for_world' 未定义，无法为世界 '{world_id}' 加载自定义Jieba词典。")
            except Exception as e_jieba_load:
                print(f"警告：为世界 '{world_id}' 加载自定义Jieba词典时发生错误: {e_jieba_load}")

            return True
        else:
            print(f"错误：世界 '{get_world_display_name(world_id)}' 的数据目录 '{world_path_check}' 不存在或无效。激活失败。")
            # _active_world_id = None # 不需要在这里设置，因为前面没有成功设置新的 _active_world_id
            return False
    else:
        print(f"错误：尝试激活不存在的世界ID '{world_id}'。")
        return False

def get_world_display_name(world_id: str) -> str:
    return _worlds_metadata.get(world_id, f"未知世界 (ID: {world_id})")

def delete_world(world_id: str) -> str:
    global _worlds_metadata, _active_world_id
    if world_id not in _worlds_metadata:
        return f"错误：世界ID '{world_id}' 不存在，无法删除。"

    world_name = _worlds_metadata.pop(world_id)
    _save_worlds_metadata()

    world_path_to_delete = get_world_path(world_id)
    if world_path_to_delete and os.path.exists(world_path_to_delete):
        try:
            bm25_file_path = os.path.join(world_path_to_delete, WORLD_WORLDVIEW_BM25_FILENAME)
            if os.path.exists(bm25_file_path):
                os.remove(bm25_file_path)
                print(f"已删除世界 '{world_name}' 的BM25模型文件。")
            shutil.rmtree(world_path_to_delete)
        except Exception as e:
            print(f"警告：从元数据中移除了世界 '{world_name}'，但删除其数据目录 '{world_path_to_delete}' 或BM25模型失败: {e}")
            return f"世界 '{world_name}' 已从元数据中移除，但其数据目录可能未能完全删除。请手动检查。"

    if _active_world_id == world_id:
        _active_world_id = None
        print(f"当前活动世界 '{world_name}' 已被删除，自动取消激活。")

    return f"世界 '{world_name}' (ID: {world_id}) 及其所有数据已成功删除。"

# --- 角色管理 (修改) ---
def _load_character_data() -> List[Dict]:
    if not _active_world_id: return []
    world_path = get_world_path(_active_world_id)
    if not world_path: return []
    char_file = os.path.join(world_path, WORLD_CHARACTERS_DB_FILENAME)
    if os.path.exists(char_file):
        try:
            with open(char_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, TypeError):
            print(f"警告：角色数据文件 '{char_file}' 解析失败或为空。返回空列表。")
            return []
    return []

def _save_character_data(characters: List[Dict]):
    if not _active_world_id: return
    world_path = get_world_path(_active_world_id)
    if not world_path: return

    char_file = os.path.join(world_path, WORLD_CHARACTERS_DB_FILENAME)
    try:
        with open(char_file, 'w', encoding='utf-8') as f:
            json.dump(characters, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"错误：保存角色数据到 '{char_file}' 失败: {e}")

def _extract_structured_traits_llm(full_description: str, character_name: str) -> Dict[str, Any]:
    """
    使用LLM从角色完整描述中提取结构化特质。
    这是一个实验性功能。
    """
    trait_extraction_model = getattr(config, 'OLLAMA_TRAIT_EXTRACTION_MODEL', config.OLLAMA_MODEL) # 使用特定模型或回退
    if not trait_extraction_model:
        print("警告 (角色特质提取): 未配置 OLLAMA_TRAIT_EXTRACTION_MODEL。跳过此步骤。")
        return {}

    # 截断输入以避免超长
    max_input_len_trait_extraction = getattr(config, 'MAX_TRAIT_EXTRACTION_INPUT_LEN_LLM', 3000)
    truncated_description = full_description
    if len(full_description) > max_input_len_trait_extraction:
        truncated_description = full_description[:max_input_len_trait_extraction] + "...[描述过长已截断]"
        print(f"角色特质提取: 角色 '{character_name}' 的描述已从 {len(full_description)} 截断至 {len(truncated_description)} 字符。")

    prompt = f"""
    请仔细阅读以下关于角色“{character_name}”的描述，并提取出结构化的角色特质。
    你需要识别并输出以下方面的信息（如果存在）：
    1.  "core_values": 角色的核心价值观或信仰 (字符串列表，例如：["正义", "家庭至上"])
    2.  "main_goals": 角色的主要短期或长期目标 (字符串列表，例如：["复仇", "寻找失落的宝藏"])
    3.  "key_skills": 角色的关键技能或能力 (字符串列表，例如：["剑术高超", "擅长潜行", "博学"])
    4.  "significant_relationships": 角色的重要人际关系及其性质 (对象列表，每个对象包含 "name" 和 "relationship_type"，例如：[{{"name": "艾丽娅", "relationship_type": "妹妹/保护对象"}}, {{"name": "魔王", "relationship_type": "宿敌"}}])
    5.  "personality_tags": 描述角色性格的关键词 (字符串列表，例如：["勇敢", "冲动", "善良", "多疑"])

    如果某些方面的信息在描述中不存在，请在对应的键中返回一个空列表 `[]` 或 `null`。
    请严格按照以下JSON格式输出，不要包含任何额外的解释或Markdown标记：
    {{
      "core_values": [],
      "main_goals": [],
      "key_skills": [],
      "significant_relationships": [],
      "personality_tags": []
    }}

    角色描述:
    ---
    {truncated_description}
    ---
    提取的结构化特质 (JSON格式):
    """
    system_message = "你是一个专业的角色信息分析和结构化提取助手。请专注于准确提取信息，并严格按照指定的JSON格式输出。"

    llm_response = generate_text(
        prompt=prompt,
        system_message=system_message,
        model_name=trait_extraction_model,
        use_json_format=True # 请求JSON输出
    )

    if llm_response.startswith("错误:") or llm_response.startswith("Error:"):
        print(f"角色特质提取: LLM调用失败: {llm_response}")
        return {}

    try:
        # 尝试去除markdown代码块（如果存在）
        if llm_response.strip().startswith("```json"):
            llm_response = llm_response.strip()[7:]
            if llm_response.strip().endswith("```"):
                llm_response = llm_response.strip()[:-3]
        elif llm_response.strip().startswith("```"):
            llm_response = llm_response.strip()[3:]
            if llm_response.strip().endswith("```"):
                llm_response = llm_response.strip()[:-3]

        data = json.loads(llm_response.strip())
        # 基本验证
        expected_keys = ["core_values", "main_goals", "key_skills", "significant_relationships", "personality_tags"]
        extracted_data = {}
        for key in expected_keys:
            value = data.get(key)
            if isinstance(value, list): # 确保是列表
                if key == "significant_relationships": # 对关系列表做进一步检查
                    validated_rels = []
                    for item in value:
                        if isinstance(item, dict) and "name" in item and "relationship_type" in item:
                            validated_rels.append(item)
                    extracted_data[key] = validated_rels
                else: # 其他是字符串列表
                    extracted_data[key] = [str(v) for v in value if isinstance(v, (str, int, float))] # 转换为字符串
            else:
                extracted_data[key] = [] # 如果不是列表或不存在，则为空列表
        return extracted_data
    except json.JSONDecodeError:
        print(f"角色特质提取: LLM响应JSON解析失败。原始响应 (部分): {llm_response[:200]}")
        return {}
    except Exception as e:
        print(f"角色特质提取: 处理LLM响应时发生未知错误: {e}。原始响应 (部分): {llm_response[:200]}")
        return {}


def add_character(name: str, full_description: str) -> str:
    if not _active_world_id: return "错误：没有活动的存储世界。"
    if not name.strip() or not full_description.strip():
        return "错误：角色名称和完整描述不能为空。"

    if len(full_description) > MAX_CHAR_FULL_DESC_LEN:
        return f"错误：角色完整描述过长 (超过 {MAX_CHAR_FULL_DESC_LEN} 字符)。请缩减。"

    characters = _load_character_data()
    existing_char_index = next((i for i, char in enumerate(characters) if char['name'] == name), None)

    print(f"正在为角色 '{name}' 生成概要描述 (使用模型: {getattr(config, 'OLLAMA_SUMMARY_MODEL', config.OLLAMA_MODEL)})...")
    summary_desc = generate_summary(full_description) # generate_summary 内部会使用 OLLAMA_SUMMARY_MODEL
    if not summary_desc or summary_desc.startswith("错误:") or summary_desc.startswith("Error:"):
        print(f"为角色 '{name}' 生成概要描述失败。LLM响应: {summary_desc}。概要将为空。")
        summary_desc = ""

    # 新增：尝试提取结构化特质
    print(f"正在为角色 '{name}' 提取结构化特质 (使用模型: {getattr(config, 'OLLAMA_TRAIT_EXTRACTION_MODEL', config.OLLAMA_MODEL)})...")
    structured_traits = _extract_structured_traits_llm(full_description, name)
    if structured_traits:
        print(f"为角色 '{name}' 提取到的结构化特质: {json.dumps(structured_traits, ensure_ascii=False, indent=2)}")
    else:
        print(f"未能为角色 '{name}' 提取有效的结构化特质。")


    new_char_data = {
        "name": name,
        "full_description": full_description,
        "summary_description": summary_desc,
        "structured_traits": structured_traits # 新增字段
    }

    if existing_char_index is not None:
        characters[existing_char_index] = new_char_data
        msg = f"角色 '{name}' 的信息已更新。"
    else:
        characters.append(new_char_data)
        msg = f"角色 '{name}' 已添加到世界 '{get_world_display_name(_active_world_id)}'。"

    _save_character_data(characters)
    return msg

def get_character(name: str) -> Optional[Dict]:
    if not _active_world_id: return None
    characters = _load_character_data()
    char_data = next((char for char in characters if char['name'] == name), None)
    # 确保旧数据也能兼容，如果缺少 structured_traits 字段
    if char_data and "structured_traits" not in char_data:
        char_data["structured_traits"] = {}
    return char_data

def get_character_names() -> List[str]:
    if not _active_world_id: return []
    characters = _load_character_data()
    return [char['name'] for char in characters]

def get_all_characters() -> List[Dict]:
    if not _active_world_id: return []
    all_chars = _load_character_data()
    # 确保旧数据也能兼容
    for char_data in all_chars:
        if "structured_traits" not in char_data:
            char_data["structured_traits"] = {}
    return all_chars

def delete_character(name: str) -> str:
    if not _active_world_id: return "错误：没有活动的存储世界。"
    characters = _load_character_data()
    original_len = len(characters)
    characters = [char for char in characters if char['name'] != name]
    if len(characters) < original_len:
        _save_character_data(characters)
        return f"角色 '{name}' 已从世界 '{get_world_display_name(_active_world_id)}' 中删除。"
    else:
        return f"错误：在世界 '{get_world_display_name(_active_world_id)}' 中未找到角色 '{name}'。"

# --- 世界观文本和FAISS索引管理 (无修改) ---
def _load_worldview_texts_map() -> Dict[int, Dict]:
    if not _active_world_id: return {}
    world_path = get_world_path(_active_world_id)
    if not world_path: return {}

    wv_texts_file = os.path.join(world_path, WORLD_WORLDVIEW_TEXTS_FILENAME)
    if os.path.exists(wv_texts_file):
        try:
            with open(wv_texts_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                return {int(k): v for k, v in loaded_data.items()}
        except (json.JSONDecodeError, TypeError, ValueError):
            print(f"警告：世界观文本文件 '{wv_texts_file}' 解析失败或格式不正确。返回空字典。")
            return {}
    return {}

def _save_worldview_texts_map(texts_map: Dict[int, Dict]):
    if not _active_world_id: return
    world_path = get_world_path(_active_world_id)
    if not world_path: return

    wv_texts_file = os.path.join(world_path, WORLD_WORLDVIEW_TEXTS_FILENAME)
    try:
        with open(wv_texts_file, 'w', encoding='utf-8') as f:
            json.dump({str(k): v for k, v in texts_map.items()}, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"错误：保存世界观文本映射到 '{wv_texts_file}' 失败: {e}")

def _load_faiss_index() -> Optional[faiss.IndexIDMap]:
    if not _active_world_id: return None
    world_path = get_world_path(_active_world_id)
    if not world_path: return None

    faiss_index_file = os.path.join(world_path, WORLD_WORLDVIEW_FAISS_INDEX_FILENAME)
    if os.path.exists(faiss_index_file):
        try:
            index = faiss.read_index(faiss_index_file)
            current_model_dim = get_model_embedding_dimension()
            if index.d != current_model_dim:
                print(f"警告：加载的FAISS索引维度 ({index.d}) 与当前嵌入模型 "
                      f"('{OLLAMA_EMBEDDING_MODEL_NAME}' 的维度 {current_model_dim}) 不匹配。"
                      f"这可能导致错误或不准确的搜索结果。建议删除索引文件 '{faiss_index_file}' 并重新生成数据。")
            return index
        except Exception as e:
            print(f"错误：加载FAISS索引文件 '{faiss_index_file}' 失败: {e}。可能需要重建索引。")
            return None
    else:
        print(f"FAISS索引文件 '{faiss_index_file}' 未找到，将尝试创建一个新的空索引。")
        try:
            dimension = get_model_embedding_dimension()
            index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
            faiss.write_index(index, faiss_index_file)
            print(f"已创建新的空FAISS索引，维度为 {dimension}。")
            return index
        except Exception as e:
            print(f"错误：初始化新的空FAISS索引失败: {e}")
            return None

# --- BM25 模型管理 (无修改) ---
def _get_bm25_model_path() -> Optional[str]:
    if not _active_world_id: return None
    world_path = get_world_path(_active_world_id)
    if not world_path: return None
    return os.path.join(world_path, WORLD_WORLDVIEW_BM25_FILENAME)

def _load_bm25_model_and_doc_ids() -> Tuple[Optional[BM25Okapi], Optional[List[int]]]:
    bm25_file_path = _get_bm25_model_path()
    if not bm25_file_path or not os.path.exists(bm25_file_path):
        return None, None
    try:
        with open(bm25_file_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict) and 'model' in data and 'doc_ids' in data:
                return data['model'], data['doc_ids']
            else:
                print(f"警告：BM25模型文件 '{bm25_file_path}' 格式不符合预期（缺少doc_ids或非字典）。尝试仅加载模型。")
                if isinstance(data, BM25Okapi):
                     print(f"警告：BM25模型文件 '{bm25_file_path}' 为旧格式。建议重建世界观数据以更新BM25模型。")
                     return data, None
                return None, None
    except Exception as e:
        print(f"错误：加载BM25模型文件 '{bm25_file_path}' 失败: {e}")
        return None, None

def _build_and_save_bm25_model_for_active_world(texts_map: Dict[int, Dict]) -> bool:
    if not _active_world_id:
        print("错误 (BM25构建): 没有活动的存储世界。")
        return False
    if not texts_map:
        print(f"警告 (BM25构建): 世界 '{get_world_display_name(_active_world_id)}' 的文本映射为空，将清除BM25模型。")
        return _clear_bm25_model_for_active_world()

    bm25_file_path = _get_bm25_model_path()
    if not bm25_file_path:
        print("错误 (BM25构建): 无法获取BM25模型文件路径。")
        return False

    print(f"正在为世界 '{get_world_display_name(_active_world_id)}' 构建BM25模型...")
    try:
        doc_ids_for_bm25 = sorted(list(texts_map.keys()))
        corpus = [texts_map[doc_id]["full_text"] for doc_id in doc_ids_for_bm25]
        tokenized_corpus = [jieba.lcut(doc.lower()) for doc in corpus]
        bm25_model = BM25Okapi(tokenized_corpus)
        data_to_save = {'model': bm25_model, 'doc_ids': doc_ids_for_bm25}
        with open(bm25_file_path, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"BM25模型已为世界 '{get_world_display_name(_active_world_id)}' 构建并保存。包含 {len(doc_ids_for_bm25)} 个文档。")
        return True
    except Exception as e:
        print(f"错误 (BM25构建): 构建或保存BM25模型失败: {e}")
        return False

def _clear_bm25_model_for_active_world() -> bool:
    if not _active_world_id: return False
    bm25_file_path = _get_bm25_model_path()
    if bm25_file_path and os.path.exists(bm25_file_path):
        try:
            os.remove(bm25_file_path)
            print(f"已清除世界 '{get_world_display_name(_active_world_id)}' 的BM25模型文件。")
            return True
        except Exception as e:
            print(f"错误：清除BM25模型文件 '{bm25_file_path}' 失败: {e}")
            return False
    return True

# --- 文本分块和添加世界观 (无修改) ---
def _text_splitter(text: str, chunk_size_config: int, chunk_overlap_config: int) -> List[str]:
    print(f"Using semantic_text_splitter with chunk_size={chunk_size_config}, overlap={chunk_overlap_config}")
    return semantic_text_splitter(text, chunk_size_config, chunk_overlap_config)

def add_worldview_text(full_text_block: str) -> str:
    if not _active_world_id: return "错误：没有活动的存储世界。"
    if not full_text_block.strip(): return "错误：世界观文本不能为空。"

    faiss_index = _load_faiss_index()
    worldview_texts_map = _load_worldview_texts_map()

    if faiss_index is None:
        return "错误：FAISS索引不可用，无法添加世界观文本。请检查日志。"

    chunks = _text_splitter(full_text_block, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks: return "错误：文本分块失败或产生空块。"

    print(f"正在为世界 '{get_world_display_name(_active_world_id)}' 的 {len(chunks)} 个文本块使用模型 '{OLLAMA_EMBEDDING_MODEL_NAME}' 生成嵌入...")
    chunk_embeddings_np = get_embeddings(chunks)

    if chunk_embeddings_np.shape[0] == 0 :
        return "错误：未能为文本块生成有效的嵌入向量 (可能是所有块的文本都为空或处理失败)。"
    if chunk_embeddings_np.shape[1] != faiss_index.d:
        error_msg = (f"错误：生成的嵌入维度 ({chunk_embeddings_np.shape[1]}) 与FAISS索引维度 ({faiss_index.d}) 不匹配。"
                     f"这通常发生在更改了嵌入模型后。请删除当前世界目录下的 '{WORLD_WORLDVIEW_FAISS_INDEX_FILENAME}' 文件，"
                     f"然后重新添加文本，或重建整个世界的数据。")
        print(error_msg)
        return error_msg

    num_zero_vectors = 0
    for i in range(chunk_embeddings_np.shape[0]):
        if np.all(chunk_embeddings_np[i] == 0):
            num_zero_vectors += 1
            print(f"警告 (add_worldview_text): 第 {i+1}/{len(chunks)} 块 (文本片段: '{chunks[i][:50]}...') 的嵌入是零向量。")

    zero_vector_warning_msg = ""
    if num_zero_vectors > 0:
        zero_vector_warning_msg = (f" (注意：其中 {num_zero_vectors}/{len(chunks)} 个块的嵌入为零向量。 "
                                   f"这可能严重影响检索质量。请检查嵌入模型 '{OLLAMA_EMBEDDING_MODEL_NAME}' 的状态和控制台日志。)")
        print(zero_vector_warning_msg.strip())
        if num_zero_vectors == len(chunks):
             return f"错误：所有 {len(chunks)} 个文本块的嵌入都为零向量。添加中止。请检查嵌入模型 '{OLLAMA_EMBEDDING_MODEL_NAME}' 是否正常工作。"

    next_id = 0
    if worldview_texts_map:
        current_max_id = max(worldview_texts_map.keys()) if worldview_texts_map else -1
        next_id = current_max_id + 1

    ids_to_add = np.array([next_id + i for i in range(len(chunks))]).astype('int64')

    try:
        faiss_index.add_with_ids(chunk_embeddings_np, ids_to_add)
    except Exception as e:
        return f"错误：向FAISS索引添加向量失败: {e}"

    for i, chunk_text in enumerate(chunks):
        current_chunk_id = ids_to_add[i]
        worldview_texts_map[int(current_chunk_id)] = {"full_text": chunk_text, "summary_text": None}

    world_path = get_world_path(_active_world_id)
    if not world_path: return "错误：无法获取活动世界路径以保存数据。"

    faiss_index_file = os.path.join(world_path, WORLD_WORLDVIEW_FAISS_INDEX_FILENAME)
    try:
        faiss.write_index(faiss_index, faiss_index_file)
    except Exception as e:
        return f"错误：保存更新后的FAISS索引到 '{faiss_index_file}' 失败: {e}"

    _save_worldview_texts_map(worldview_texts_map)

    if not _build_and_save_bm25_model_for_active_world(worldview_texts_map):
        print(f"警告：向世界 '{get_world_display_name(_active_world_id)}' 添加文本后，更新BM25模型失败。")

    success_msg = f"成功将文本分割为 {len(chunks)} 个块，并添加到世界 '{get_world_display_name(_active_world_id)}' 的世界观中。"
    return success_msg + zero_vector_warning_msg

# --- 检索函数 (无修改) ---
def search_worldview_semantic(query_text: str, k: int) -> List[Tuple[int, float, str]]:
    if not _active_world_id: return []
    faiss_index = _load_faiss_index()
    worldview_texts_map = _load_worldview_texts_map()
    if faiss_index is None or faiss_index.ntotal == 0 or not worldview_texts_map:
        return []
    query_embedding = get_embedding(query_text)
    if np.all(query_embedding == 0):
        print(f"警告 (search_worldview_semantic): 查询文本 '{query_text[:50]}...' 的嵌入是零向量。")
        return []
    if query_embedding.shape[0] != faiss_index.d:
         print(f"错误 (search_worldview_semantic): 查询嵌入维度与索引维度不符。")
         return []
    query_embedding_np = query_embedding.reshape(1, -1).astype('float32')
    actual_k = min(k, faiss_index.ntotal)
    if actual_k == 0: return []
    try:
        distances, indices = faiss_index.search(query_embedding_np, actual_k)
    except Exception as e:
        print(f"错误 (search_worldview_semantic): FAISS搜索失败: {e}")
        return []
    results = []
    for i in range(len(indices[0])):
        doc_id = indices[0][i]
        dist = distances[0][i]
        if doc_id == -1: continue
        text_data = worldview_texts_map.get(int(doc_id))
        if text_data:
            results.append((int(doc_id), float(dist), text_data["full_text"]))
        else:
            print(f"警告 (search_worldview_semantic): FAISS找到ID {doc_id}，但在文本映射中未找到。")
    return results

def search_worldview_bm25(query_text: str, k: int) -> List[Tuple[int, float, str]]:
    if not _active_world_id: return []
    bm25_model, doc_ids_for_bm25 = _load_bm25_model_and_doc_ids()
    worldview_texts_map = _load_worldview_texts_map()
    if not bm25_model or not doc_ids_for_bm25 or not worldview_texts_map:
        if not bm25_model: print("提示 (BM25搜索): BM25模型未加载或不可用。")
        return []
    tokenized_query = jieba.lcut(query_text.lower())
    if not tokenized_query: return []
    try:
        doc_scores = bm25_model.get_scores(tokenized_query)
        scored_doc_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)
        results = []
        for i in range(min(k, len(scored_doc_indices))):
            bm25_internal_idx = scored_doc_indices[i]
            doc_id = doc_ids_for_bm25[bm25_internal_idx]
            score = doc_scores[bm25_internal_idx]
            text_data = worldview_texts_map.get(doc_id)
            if text_data:
                results.append((doc_id, score, text_data["full_text"]))
            else:
                 print(f"警告 (BM25搜索): 内部索引 {bm25_internal_idx} -> ID {doc_id} 在文本映射中未找到。")
        return results
    except Exception as e:
        print(f"错误 (BM25搜索): BM25检索时发生错误: {e}")
        return []

def search_worldview_keyword(query_text: str, k: int) -> List[Tuple[int, int, str]]:
    if not _active_world_id: return []
    worldview_texts_map = _load_worldview_texts_map()
    if not worldview_texts_map: return []
    query_tokens = set(jieba.lcut(query_text.lower()))
    if not query_tokens: return []
    results_with_scores = []
    for doc_id, text_data in worldview_texts_map.items():
        full_text = text_data.get("full_text", "")
        if not full_text: continue
        doc_tokens = set(jieba.lcut(full_text.lower()))
        matched_keywords = query_tokens.intersection(doc_tokens)
        match_count = len(matched_keywords)
        if match_count > 0:
            results_with_scores.append((doc_id, match_count, full_text))
    results_with_scores.sort(key=lambda x: x[1], reverse=True)
    return results_with_scores[:k]

# --- 其他辅助函数 (无修改) ---
def get_worldview_size() -> int:
    if not _active_world_id: return 0
    faiss_index = _load_faiss_index()
    return faiss_index.ntotal if faiss_index else 0

def get_all_worldview_data_for_active_world() -> Tuple[Optional[faiss.IndexIDMap], Dict[int, Dict]]:
    if not _active_world_id: return None, {}
    return _load_faiss_index(), _load_worldview_texts_map()

def get_all_worldview_texts_for_active_world() -> List[str]:
    if not _active_world_id: return []
    texts_map = _load_worldview_texts_map()
    return [data["full_text"] for data in texts_map.values() if "full_text" in data and data["full_text"].strip()]

def rebuild_worldview_from_data(texts_with_ids: Dict[int, Dict], embeddings_np: np.ndarray) -> bool:
    if not _active_world_id:
        print("错误 (rebuild_worldview_from_data): 没有活动的存储世界。")
        return False
    world_path = get_world_path(_active_world_id)
    if not world_path:
        print(f"错误 (rebuild_worldview_from_data): 无法获取世界 '{_active_world_id}' 的路径。")
        return False
    dimension = get_model_embedding_dimension()
    new_texts_map: Dict[int, Dict]
    new_faiss_index: faiss.IndexIDMap
    if not texts_with_ids or embeddings_np.shape[0] == 0:
        print(f"警告 (rebuild_worldview_from_data): 提供的文本或嵌入为空。将清空世界 '{_active_world_id}' 的世界观。")
        new_texts_map = {}
        new_faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
    elif embeddings_np.shape[0] != len(texts_with_ids):
        print(f"错误 (rebuild_worldview_from_data): 文本数量 ({len(texts_with_ids)}) 与嵌入数量 ({embeddings_np.shape[0]}) 不匹配。")
        return False
    elif embeddings_np.shape[1] != dimension:
        print(f"错误 (rebuild_worldview_from_data): 提供的嵌入维度 ({embeddings_np.shape[1]}) "
              f"与当前模型配置的维度 ({dimension}) 不匹配。重建中止。")
        return False
    else:
        num_zero_vectors_rebuild = 0
        for i in range(embeddings_np.shape[0]):
            if np.all(embeddings_np[i] == 0):
                num_zero_vectors_rebuild += 1
        if num_zero_vectors_rebuild > 0:
            print(f"警告 (rebuild_worldview_from_data): 用于重建的数据包含 {num_zero_vectors_rebuild}/{embeddings_np.shape[0]} 个零向量。")
            if num_zero_vectors_rebuild == embeddings_np.shape[0] and embeddings_np.shape[0] > 0:
                 print(f"错误 (rebuild_worldview_from_data): 所有提供的嵌入均为零向量。重建中止以防数据损坏。")
                 return False
        new_texts_map = texts_with_ids
        new_faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(embeddings_np.shape[1]))
        all_ids = np.array(list(new_texts_map.keys()), dtype=np.int64)
        if embeddings_np.dtype != np.float32:
            embeddings_np = embeddings_np.astype(np.float32)
        try:
            new_faiss_index.add_with_ids(embeddings_np, all_ids)
        except Exception as e:
            print(f"错误 (rebuild_worldview_from_data): 向新FAISS索引添加向量时失败: {e}")
            return False
    _save_worldview_texts_map(new_texts_map)
    faiss_index_file = os.path.join(world_path, WORLD_WORLDVIEW_FAISS_INDEX_FILENAME)
    try:
        faiss.write_index(new_faiss_index, faiss_index_file)
        print(f"世界 '{get_world_display_name(_active_world_id)}' 的世界观（FAISS和文本映射）已成功重建。包含 {new_faiss_index.ntotal} 条目。")
        if new_faiss_index.ntotal > 0:
            success_bm25 = _build_and_save_bm25_model_for_active_world(new_texts_map)
            if not success_bm25:
                print(f"警告 (rebuild_worldview_from_data): 为世界 '{get_world_display_name(_active_world_id)}' 重建BM25模型失败。")
        else:
            _clear_bm25_model_for_active_world()
        return True
    except Exception as e:
        print(f"错误 (rebuild_worldview_from_data): 保存重建的FAISS索引到 '{faiss_index_file}' 失败: {e}")
        return False

# --- 知识图谱管理 (无修改) ---
def _load_kg_data() -> Dict[str, List[List[str]]]:
    if not _active_world_id: return {"triples": []}
    world_path = get_world_path(_active_world_id)
    if not world_path: return {"triples": []}
    kg_file = os.path.join(world_path, WORLD_KNOWLEDGE_GRAPH_FILENAME)
    if os.path.exists(kg_file):
        try:
            with open(kg_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and "triples" in data and isinstance(data["triples"], list):
                    validated_triples = []
                    for t in data["triples"]:
                        if isinstance(t, list) and len(t) == 3 and all(isinstance(s, str) for s in t):
                            validated_triples.append(t)
                        else:
                            print(f"警告：在 '{kg_file}' 中发现格式不正确的三元组并已跳过: {t}")
                    return {"triples": validated_triples}
                else:
                    print(f"警告：知识图谱文件 '{kg_file}' 格式不正确。将返回空知识图谱。")
                    return {"triples": []}
        except (json.JSONDecodeError, TypeError):
            print(f"警告：知识图谱文件 '{kg_file}' 解析失败或为空。返回空知识图谱。")
            return {"triples": []}
    return {"triples": []}

def _save_kg_data(kg_data: Dict[str, List[List[str]]]):
    if not _active_world_id: return
    world_path = get_world_path(_active_world_id)
    if not world_path: return
    kg_file = os.path.join(world_path, WORLD_KNOWLEDGE_GRAPH_FILENAME)
    try:
        with open(kg_file, 'w', encoding='utf-8') as f:
            json.dump(kg_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"错误：保存知识图谱数据到 '{kg_file}' 失败: {e}")

def add_triples_to_kg(triples: List[List[str]], overwrite: bool = False) -> str:
    if not _active_world_id: return "错误：没有活动的存储世界。"
    valid_triples_to_add = []
    for t in triples:
        if isinstance(t, list) and len(t) == 3 and all(isinstance(s, str) for s in t):
            valid_triples_to_add.append(t)
        else:
            print(f"警告 (add_triples_to_kg): 跳过格式不正确的三元组: {t}")
    if not valid_triples_to_add and not overwrite :
        return "提示：没有有效的输入三元组可供添加。"
    if not valid_triples_to_add and overwrite and not triples:
         pass
    current_kg_data = _load_kg_data()
    if overwrite:
        current_kg_data["triples"] = valid_triples_to_add
        msg_action = "覆盖"
    else:
        existing_triples_set = {tuple(t) for t in current_kg_data["triples"]}
        added_count = 0
        for triple in valid_triples_to_add:
            if tuple(triple) not in existing_triples_set:
                current_kg_data["triples"].append(triple)
                existing_triples_set.add(tuple(triple))
                added_count += 1
        if added_count == 0 and valid_triples_to_add:
            return "提示：提供的三元组均已存在于知识图谱中，未添加新内容。"
        elif added_count == 0 and not valid_triples_to_add:
             return "提示：没有有效的输入三元组可供添加。"
        msg_action = f"追加了 {added_count} 条新"
    _save_kg_data(current_kg_data)
    total_triples = len(current_kg_data["triples"])
    return f"成功{msg_action}三元组到世界 '{get_world_display_name(_active_world_id)}' 的知识图谱。当前总数: {total_triples}。"

def get_kg_triples_for_active_world() -> List[List[str]]:
    if not _active_world_id: return []
    kg_data = _load_kg_data()
    return kg_data.get("triples", [])

def get_kg_triples_count_for_active_world() -> int:
    if not _active_world_id: return 0
    return len(get_kg_triples_for_active_world())