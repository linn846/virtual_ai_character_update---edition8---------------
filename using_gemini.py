中文回答
下面是我的代码，目前已经实现了很多东西，但是还缺少 评测功能，我认为 除了人工评测外，似乎评测一下知识库检索的内容是否符合问题要求是很有用的，请你添加这一功能，给出修改后的完整代码，如果代码有修改，那么给出该文件的完整代码，如果未修改则不需要给出。
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple 

from config import COMPRESSION_TARGET_CLUSTERS, COMPRESSION_FALLBACK_SUMMARY_LENGTH 
from database_utils import (
    get_all_worldview_data_for_active_world,
    rebuild_worldview_from_data,
    get_worldview_size,
    _active_world_id, get_world_display_name
)
from embedding_utils import get_embedding, get_embeddings 
from llm_utils import generate_summary 

def compress_worldview_db_for_active_world(force_compress: bool = False) -> str:
    """
    使用K-Means聚类和LLM总结来压缩当前活动世界的世界观数据库。
    """
    if not _active_world_id:
        return "错误：没有活动的存储世界来进行压缩。"

    world_name = get_world_display_name(_active_world_id)
    current_size = get_worldview_size() 
    
    if not isinstance(COMPRESSION_TARGET_CLUSTERS, (int, float)) or COMPRESSION_TARGET_CLUSTERS <= 0:
        return f"错误：配置的压缩目标簇数 (COMPRESSION_TARGET_CLUSTERS={COMPRESSION_TARGET_CLUSTERS}) 无效。压缩中止。"

    if current_size <= COMPRESSION_TARGET_CLUSTERS * 1.5 and not force_compress: 
        return f"世界 '{world_name}' 的世界观大小 ({current_size}) 未达到压缩标准或已接近目标 ({COMPRESSION_TARGET_CLUSTERS})。除非强制，否则不执行压缩。"

    faiss_index, worldview_texts_map = get_all_worldview_data_for_active_world()

    if not faiss_index or faiss_index.ntotal == 0:
        return f"世界 '{world_name}' 的世界观为空。无需压缩。"
    if faiss_index.ntotal <= 1: 
        return f"世界 '{world_name}' 的世界观只有1条记录。无法压缩。"

    print(f"开始压缩世界 '{world_name}' 的世界观。当前大小: {faiss_index.ntotal}")
    
    if not hasattr(faiss_index, 'reconstruct_n'):
        return f"错误：当前FAISS索引类型不支持 reconstruct_n 方法，无法获取所有向量进行压缩。"
    
    try:
        original_ids_list = []
        texts_for_summary_list = []
        vectors_for_clustering_list = []

        for doc_id_int, text_data_dict in worldview_texts_map.items():
            try:
                vector = faiss_index.reconstruct(int(doc_id_int)) 
                vectors_for_clustering_list.append(vector)
                original_ids_list.append(doc_id_int) 
                texts_for_summary_list.append(text_data_dict["full_text"])
            except RuntimeError as e: 
                print(f"警告 (压缩): 重建世界 '{world_name}' 的向量时，ID {doc_id_int} 在FAISS索引中未找到或发生错误: {e}。跳过此条目。")
                continue 
            except KeyError:
                 print(f"警告 (压缩): 重建世界 '{world_name}' 的文本时，ID {doc_id_int} 在 worldview_texts_map 中缺失 'full_text'。跳过。")
                 continue


        if not vectors_for_clustering_list:
            return f"世界 '{world_name}' 未能从FAISS索引和文本映射中提取到有效的向量和文本对。压缩中止。"

        all_vectors_np = np.array(vectors_for_clustering_list, dtype=np.float32)
        
    except Exception as e_recon:
        return f"错误 (压缩): 从世界 '{world_name}' 的FAISS索引提取向量时发生严重错误: {e_recon}。压缩中止。"


    if all_vectors_np.shape[0] == 0: 
        return f"世界 '{world_name}' 中没有找到用于压缩的向量。"

    num_samples = all_vectors_np.shape[0]
    n_clusters_target = COMPRESSION_TARGET_CLUSTERS
    n_clusters_actual = min(num_samples, n_clusters_target)
    
    if num_samples > 0 and n_clusters_actual <= 0: n_clusters_actual = 1
    
    if n_clusters_actual == 0 : 
        return f"世界 '{world_name}' 无法确定有效的聚类数量 (num_samples: {num_samples})。"
    if n_clusters_actual == num_samples and not force_compress: 
        return f"世界 '{world_name}' 的样本数 ({num_samples}) 等于目标聚类数 ({n_clusters_actual})，除非强制，否则不执行压缩。"

    
    print(f"正在将世界 '{world_name}' 的 {num_samples} 个向量聚类为 {n_clusters_actual} 个簇。")
    
    try: 
        from sklearn import __version__ as sklearn_version
        sklearn_version_tuple = tuple(map(int, sklearn_version.split('.')[:2]))

        if sklearn_version_tuple < (1, 2):
            kmeans = KMeans(n_clusters=n_clusters_actual, random_state=42, n_init=10)
        elif sklearn_version_tuple < (1, 4): # n_init='auto' was default in 1.2, explicit in 1.3
            kmeans = KMeans(n_clusters=n_clusters_actual, random_state=42, n_init='auto')
        else: 
             kmeans = KMeans(n_clusters=n_clusters_actual, random_state=42, n_init='auto') 
    except ImportError: 
        kmeans = KMeans(n_clusters=n_clusters_actual, random_state=42, n_init='auto') 
        
    cluster_labels = kmeans.fit_predict(all_vectors_np)

    new_worldview_texts_with_ids: Dict[int, Dict] = {} 
    new_worldview_embeddings_list: List[np.ndarray] = []
    
    print(f"正在为世界 '{world_name}' 的 {n_clusters_actual} 个簇生成LLM摘要...")
    next_new_id_counter = 0 
    for i_cluster in range(n_clusters_actual):
        indices_in_this_cluster = np.where(cluster_labels == i_cluster)[0]
        
        if len(indices_in_this_cluster) == 0: continue 

        texts_in_cluster = [texts_for_summary_list[idx] for idx in indices_in_this_cluster]
        
        if not texts_in_cluster: 
            print(f"警告 (世界 '{world_name}'): 簇 {i_cluster} 没有关联的文本（异常情况）。跳过。")
            continue

        summary_text_content: str
        if len(texts_in_cluster) == 1:
            summary_text_content = texts_in_cluster[0] 
        else:
            combined_text_for_summary = "\n\n<CHUNK_SEPARATOR>\n\n".join(texts_in_cluster)
            summary_text_content = generate_summary(combined_text_for_summary) 

            if not summary_text_content or summary_text_content.startswith("错误:") or summary_text_content.startswith("Error:"):
                print(f"LLM对簇 {i_cluster} (世界 '{world_name}') 的总结失败或返回空。LLM消息: {summary_text_content if summary_text_content else '空响应'}")
                print("将使用合并文本的截断版本作为备用摘要。")
                fallback_summary = (" ".join(texts_in_cluster))[:COMPRESSION_FALLBACK_SUMMARY_LENGTH] 
                summary_text_content = fallback_summary if fallback_summary else "摘要生成失败且无备用内容"
        
        summary_embedding = get_embedding(summary_text_content)
        if summary_embedding.size == 0: 
            print(f"警告 (世界 '{world_name}'): 簇 {i_cluster} 的摘要 '{summary_text_content[:50]}...' 嵌入生成失败。跳过此簇。")
            continue
            
        new_worldview_texts_with_ids[next_new_id_counter] = {
            "full_text": summary_text_content,
            "summary_text": None 
        }
        new_worldview_embeddings_list.append(summary_embedding)
        print(f"世界 '{world_name}' 的簇 {i_cluster} 已总结。原始文本数: {len(texts_in_cluster)}, 摘要长度: {len(summary_text_content)}")
        next_new_id_counter += 1

    if not new_worldview_texts_with_ids or not new_worldview_embeddings_list:
        return f"世界 '{world_name}' 的压缩未能产生新的世界观条目。原始数据可能过于稀疏，或LLM总结/嵌入全部失败。"

    new_worldview_embeddings_np = np.array(new_worldview_embeddings_list, dtype=np.float32)
    
    if new_worldview_embeddings_np.ndim == 1 and new_worldview_embeddings_np.size > 0:
        new_worldview_embeddings_np = new_worldview_embeddings_np.reshape(1, -1)
    elif new_worldview_embeddings_np.size == 0 and next_new_id_counter > 0 : 
         return f"世界 '{world_name}' 的压缩后摘要均未能成功生成嵌入。压缩中止。"


    print(f"正在为世界 '{world_name}' 重建FAISS索引和文本映射，共 {len(new_worldview_texts_with_ids)} 条新条目...")
    rebuild_success = rebuild_worldview_from_data(new_worldview_texts_with_ids, new_worldview_embeddings_np)
    
    if not rebuild_success:
        return f"错误：为世界 '{world_name}' 重建世界观数据库失败。压缩可能未完成。"
        
    final_size = get_worldview_size() 
    return f"世界 '{world_name}' 的世界观压缩完成。条目数从 {current_size} 减少到 {final_size}。"


if __name__ == '__main__':
    print("运行压缩工具测试 (需要先通过 database_utils.py 创建并激活一个世界)...")
    
    import os 
    import config as app_config 
    from database_utils import (
        switch_active_world, get_available_worlds, add_world, 
        add_worldview_text, _active_world_id as current_active_id_for_test, 
        get_world_display_name as get_display_name_for_test,
        get_worldview_size as get_wv_size_for_test
    )
    import shutil 

    try:
        from embedding_utils import get_model_embedding_dimension
        from llm_utils import get_ollama_client
        print("正在初始化嵌入和LLM客户端以供测试...")
        get_model_embedding_dimension() 
        get_ollama_client() 
        print("模型客户端初始化成功。")
    except Exception as e_init:
        print(f"测试压缩前初始化模型客户端失败: {e_init}")
        print("请确保Ollama服务正在运行，并且config.py中配置的模型可用。")
        exit(1)

    test_compress_world_id = "compression_tool_test_world"
    test_compress_world_name = "压缩工具测试世界"

    world_dir_to_clean = os.path.join(app_config.DATA_DIR, test_compress_world_id)
    if os.path.exists(world_dir_to_clean):
        print(f"正在清理旧的测试世界目录: {world_dir_to_clean}")
        shutil.rmtree(world_dir_to_clean)
        if hasattr(database_utils, '_worlds_metadata') and test_compress_world_id in database_utils._worlds_metadata: 
            del database_utils._worlds_metadata[test_compress_world_id]
            if hasattr(database_utils, '_save_worlds_metadata'):
                 database_utils._save_worlds_metadata()


    current_worlds = get_available_worlds()
    if test_compress_world_id not in current_worlds:
        add_world_msg = add_world(test_compress_world_id, test_compress_world_name)
        print(add_world_msg)
        if "错误" in add_world_msg:
            print("创建压缩测试世界失败，测试中止。")
            exit(1)
    
    if switch_active_world(test_compress_world_id):
        # Access _active_world_id via database_utils module's global
        active_world_name = get_display_name_for_test(database_utils._active_world_id) # Use the updated ID from database_utils
        print(f"已激活世界 '{active_world_name}' 进行压缩测试。")
        
        initial_size_test = get_wv_size_for_test()
        print(f"压缩前世界观大小: {initial_size_test}")

        num_items_to_add = 0
        if initial_size_test < 5: 
            num_items_to_add = 10 - initial_size_test 
            print(f"为压缩测试添加 {num_items_to_add} 条数据...")
            test_sentences = [
                "宇宙的奥秘深不可测，星球在星云中诞生，经历亿万年的演化。这是一个非常长的句子，旨在测试分块逻辑是否能正确处理并可能将其分割，或者由于其自然段落的结构而保持为一个块，具体取决于CHUNK_SIZE的设置。如果CHUNK_SIZE设置得当，这一整段话如果作为一个知识点输入，应该被我们的新语义分块器处理。",
                "黑洞是时空的极端扭曲，其巨大的引力甚至能吞噬光线。关于黑洞的理论仍在不断发展，天文学家们通过观测遥远星系的中心来寻找它们的踪迹。",
                "人工智能技术正以前所未有的速度进步，深刻地改变着人类社会。从自然语言处理到计算机视觉，AI的应用无处不在。机器学习是其核心。",
                "机器学习作为人工智能的核心分支，通过数据驱动模型学习规律。例如，监督学习使用标记数据，而无监督学习则探索未标记数据的结构。",
                "全球气候变化已成为全人类共同面临的严峻挑战，生态系统受到威胁。极端天气事件频发，海平面上升，生物多样性减少，这些都是警示。",
                "发展可再生能源，如太阳能、风能，是减缓气候变化的关键途径。同时，提高能源效率和推广碳捕捉技术也至关重要。",
                "量子计算利用量子叠加和纠缠等特性，有望解决经典计算机难以处理的问题。例如药物研发、材料科学和复杂系统建模等领域。",
                "基因编辑技术如CRISPR-Cas9为治疗遗传病带来了希望，也引发伦理担忧。我们需要在推动科学进步的同时，审慎考虑其社会影响。",
                "虚拟现实(VR)和增强现实(AR)正在改变娱乐、教育和工作方式。它们提供了沉浸式的体验，模糊了物理世界与数字世界的界限。",
                "区块链技术以其去中心化和不可篡改性，在金融等领域展现潜力。智能合约和去中心化应用（DApps）是其重要发展方向。"
            ]
            for i in range(min(num_items_to_add, len(test_sentences))): 
                print(f"\nAdding text for compression test (item {i+1}): '{test_sentences[i % len(test_sentences)][:50]}...'")
                add_msg = add_worldview_text(test_sentences[i % len(test_sentences)]) 
                print(f"Add message: {add_msg}")

            initial_size_test = get_wv_size_for_test() 
            print(f"添加数据后的大小 (总chunk数): {initial_size_test}")
        
        if initial_size_test > 1: 
            original_target_clusters_cfg = app_config.COMPRESSION_TARGET_CLUSTERS
            new_target_clusters = max(2, min(original_target_clusters_cfg, initial_size_test // 2 if initial_size_test > 3 else initial_size_test -1 if initial_size_test > 1 else 1 ))
            if new_target_clusters == 0 and initial_size_test > 0: new_target_clusters = 1 
            
            app_config.COMPRESSION_TARGET_CLUSTERS = new_target_clusters

            print(f"测试时临时调整 COMPRESSION_TARGET_CLUSTERS 为: {app_config.COMPRESSION_TARGET_CLUSTERS}")
            
            print("\n开始压缩测试 (强制执行)...")
            compression_message = compress_worldview_db_for_active_world(force_compress=True) 
            print(f"\n压缩操作消息: {compression_message}")
            print(f"压缩后世界观大小: {get_wv_size_for_test()}")

            app_config.COMPRESSION_TARGET_CLUSTERS = original_target_clusters_cfg
        else:
            print("数据不足 (少于2条chunks)，无法进行有意义的压缩测试。")
    else:
        print(f"无法激活世界 '{test_compress_world_name}' 进行压缩测试。")

    print("\n压缩工具 (compression_utils.py) 测试运行完毕。")  import os

# --- Ollama 配置 ---
OLLAMA_BASE_URL = "http://localhost:11434"

OLLAMA_MODEL = "deepseek-r1:14b" 
OLLAMA_SUMMARY_MODEL = OLLAMA_MODEL 
OLLAMA_KG_EXTRACTION_MODEL = OLLAMA_MODEL 

OLLAMA_EMBEDDING_MODEL_NAME = "quentinz/bge-large-zh-v1.5" 

# --- 数据存储配置 ---
DATA_DIR = "data_worlds"
WORLDS_METADATA_FILE = os.path.join(DATA_DIR, "worlds_metadata.json")

# --- 世界观数据库配置 ---
WORLD_CHARACTERS_DB_FILENAME = "characters.json"
WORLD_WORLDVIEW_TEXTS_FILENAME = "worldview_texts.json"
WORLD_WORLDVIEW_FAISS_INDEX_FILENAME = "worldview_index.faiss"
WORLD_WORLDVIEW_BM25_FILENAME = "worldview_bm25.pkl" 
WORLD_KNOWLEDGE_GRAPH_FILENAME = "knowledge_graph.json"
KNOWLEDGE_SOURCE_JSON_FILENAME = "knowledge_source.json" 

# --- 文本分块配置 ---
CHUNK_SIZE = 300  
CHUNK_OVERLAP = 50 

# --- 检索配置 ---
# SIMILARITY_TOP_K 已被混合检索中的配置取代

# --- 混合检索配置 ---
HYBRID_SEARCH_ENABLED = True 
SEMANTIC_SEARCH_TOP_K_HYBRID = 5  # 增加召回数，给Reranker更多选择
BM25_SEARCH_TOP_K_HYBRID = 7      # 增加召回数
KEYWORD_SEARCH_TOP_K_HYBRID = 7   # 增加召回数
HYBRID_RRF_K = 60                 
HYBRID_FINAL_TOP_K = 10           # RRF融合后保留的结果数，也作为Reranker的输入数量

# --- 新增：Rerank 配置 ---
RERANK_ENABLED = True # 是否启用Rerank
RERANK_MODEL_NAME = "BAAI/bge-reranker-base" # HuggingFace上的模型名，或本地路径
# 如果您的Ollama服务或其他本地服务部署了rerank模型，可以修改此配置
# 例如: RERANK_MODEL_NAME = "http://localhost:8000/rerank" (自定义API端点)
# 如果是 sentence-transformers 模型，则直接写模型名或路径
RERANK_DEVICE = "cpu" # "cuda" if GPU is available, "mps" for M1/M2 Mac, else "cpu"
RERANK_TOP_K_FINAL = 3 # Rerank后最终保留的结果数，用于构建Prompt

# --- 世界观压缩配置 ---
COMPRESSION_THRESHOLD = 100 
COMPRESSION_TARGET_CLUSTERS = 20 
COMPRESSION_FALLBACK_SUMMARY_LENGTH = 1000 

# --- 文本长度限制 ---
MAX_CHAR_FULL_DESC_LEN = 10000 
MAX_SUMMARY_INPUT_LEN_LLM = 4000 
MAX_KG_TEXT_INPUT_LEN_LLM = 3000 

# 确保数据目录存在
os.makedirs(DATA_DIR, exist_ok=True)import os
import json
import shutil
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import faiss
import pickle
import jieba # 新增
from rank_bm25 import BM25Okapi # 新增
import logging # 新增，用于jieba日志控制

from config import (
    DATA_DIR, WORLDS_METADATA_FILE,
    WORLD_CHARACTERS_DB_FILENAME, WORLD_WORLDVIEW_TEXTS_FILENAME,
    WORLD_WORLDVIEW_FAISS_INDEX_FILENAME, WORLD_KNOWLEDGE_GRAPH_FILENAME,
    CHUNK_SIZE, CHUNK_OVERLAP, 
    MAX_CHAR_FULL_DESC_LEN, OLLAMA_EMBEDDING_MODEL_NAME, 
    WORLD_KNOWLEDGE_GRAPH_FILENAME,
    WORLD_WORLDVIEW_BM25_FILENAME, # 新增
    BM25_SEARCH_TOP_K_HYBRID, KEYWORD_SEARCH_TOP_K_HYBRID # 新增，用于独立检索方法
)
from embedding_utils import get_embedding, get_embeddings, get_model_embedding_dimension
from llm_utils import generate_summary 
from text_splitting_utils import semantic_text_splitter 

# 配置jieba日志级别，避免过多打印信息
jieba.setLogLevel(logging.INFO)

_active_world_id: Optional[str] = None
_worlds_metadata: Dict[str, str] = {} 

# --- BM25 模型相关的辅助数据结构 ---
# 这些可以在加载世界时填充，或者按需加载
# 为了简单，我们将在需要时加载BM25模型及其ID映射
# _current_bm25_model: Optional[BM25Okapi] = None
# _current_bm25_doc_ids: Optional[List[int]] = None


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
    
    # BM25模型文件在首次构建时创建，此处无需初始化空文件
    # bm25_model_file = os.path.join(world_path, WORLD_WORLDVIEW_BM25_FILENAME)
    # if not os.path.exists(bm25_model_file):
    #     pass # Initialized on first build

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
    if world_id is None: 
        _active_world_id = None
        print("已取消活动世界。")
        return True
    if world_id in _worlds_metadata:
        _active_world_id = world_id
        world_path = get_world_path(world_id)
        if world_path and os.path.isdir(world_path):
            print(f"已激活世界: '{get_world_display_name(world_id)}' (ID: {world_id})")
            _initialize_world_data(world_id) # 确保目录和基本文件存在
            return True
        else:
            print(f"错误：世界 '{get_world_display_name(world_id)}' 的数据目录不存在或无效。激活失败。")
            _active_world_id = None 
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
            # 删除BM25模型文件（如果存在）
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

# --- 角色管理 (无修改) ---
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

def add_character(name: str, full_description: str) -> str:
    if not _active_world_id: return "错误：没有活动的存储世界。"
    if not name.strip() or not full_description.strip():
        return "错误：角色名称和完整描述不能为空。"
    
    if len(full_description) > MAX_CHAR_FULL_DESC_LEN:
        return f"错误：角色完整描述过长 (超过 {MAX_CHAR_FULL_DESC_LEN} 字符)。请缩减。"

    characters = _load_character_data()
    existing_char_index = next((i for i, char in enumerate(characters) if char['name'] == name), None)

    print(f"正在为角色 '{name}' 生成概要描述...")
    summary_desc = generate_summary(full_description) 
    if not summary_desc or summary_desc.startswith("错误:") or summary_desc.startswith("Error:"):
        print(f"为角色 '{name}' 生成概要描述失败。LLM响应: {summary_desc}。概要将为空。")
        summary_desc = "" 

    new_char_data = {
        "name": name,
        "full_description": full_description,
        "summary_description": summary_desc 
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
    return next((char for char in characters if char['name'] == name), None)

def get_character_names() -> List[str]:
    if not _active_world_id: return []
    characters = _load_character_data()
    return [char['name'] for char in characters]

def get_all_characters() -> List[Dict]:
    if not _active_world_id: return []
    return _load_character_data()

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

# --- 世界观文本和FAISS索引管理 ---
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

# --- BM25 模型管理 ---
def _get_bm25_model_path() -> Optional[str]:
    if not _active_world_id: return None
    world_path = get_world_path(_active_world_id)
    if not world_path: return None
    return os.path.join(world_path, WORLD_WORLDVIEW_BM25_FILENAME)

def _load_bm25_model_and_doc_ids() -> Tuple[Optional[BM25Okapi], Optional[List[int]]]:
    """加载BM25模型及其对应的文档ID列表。"""
    bm25_file_path = _get_bm25_model_path()
    if not bm25_file_path or not os.path.exists(bm25_file_path):
        return None, None
    try:
        with open(bm25_file_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict) and 'model' in data and 'doc_ids' in data:
                return data['model'], data['doc_ids']
            else: # 旧版可能只存了模型
                print(f"警告：BM25模型文件 '{bm25_file_path}' 格式不符合预期（缺少doc_ids或非字典）。尝试仅加载模型。")
                if isinstance(data, BM25Okapi): # 兼容非常旧的格式（不推荐）
                    # 这种情况下，doc_ids需要从当前的worldview_texts_map重建，顺序可能不保准
                    # 最好是强制重建
                     print(f"警告：BM25模型文件 '{bm25_file_path}' 为旧格式。建议重建世界观数据以更新BM25模型。")
                     return data, None # 返回None for doc_ids to signal potential issue
                return None, None
    except Exception as e:
        print(f"错误：加载BM25模型文件 '{bm25_file_path}' 失败: {e}")
        return None, None

def _build_and_save_bm25_model_for_active_world(texts_map: Dict[int, Dict]) -> bool:
    """根据提供的文本映射构建并保存BM25模型。"""
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
        # 确保doc_ids的顺序与corpus的顺序一致
        doc_ids_for_bm25 = sorted(list(texts_map.keys())) # 按ID排序以保证一致性
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
    """清除当前活动世界的BM25模型文件。"""
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
    return True # 文件不存在也视为成功清除

# --- 文本分块和添加世界观 ---
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
    
    # 更新BM25模型
    if not _build_and_save_bm25_model_for_active_world(worldview_texts_map):
        print(f"警告：向世界 '{get_world_display_name(_active_world_id)}' 添加文本后，更新BM25模型失败。")
    
    success_msg = f"成功将文本分割为 {len(chunks)} 个块，并添加到世界 '{get_world_display_name(_active_world_id)}' 的世界观中。"
    return success_msg + zero_vector_warning_msg

# --- 检索函数 ---
def search_worldview_semantic(query_text: str, k: int) -> List[Tuple[int, float, str]]:
    """语义搜索，返回 (doc_id, distance, text_content)列表。"""
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
    """BM25搜索，返回 (doc_id, score, text_content)列表。"""
    if not _active_world_id: return []

    bm25_model, doc_ids_for_bm25 = _load_bm25_model_and_doc_ids()
    worldview_texts_map = _load_worldview_texts_map()

    if not bm25_model or not doc_ids_for_bm25 or not worldview_texts_map:
        if not bm25_model: print("提示 (BM25搜索): BM25模型未加载或不可用。")
        return []

    tokenized_query = jieba.lcut(query_text.lower())
    if not tokenized_query: return []

    try:
        # BM25Okapi.get_scores 计算的是所有文档的分数
        # BM25Okapi.get_top_n 返回的是文档本身（分词后的），我们需要ID
        # 我们需要先获取分数，然后排序，再映射回ID
        
        # 如果BM25库的get_top_n直接返回原始文档的索引，那么：
        # top_n_indices_bm25 = bm25_model.get_top_n_indices(tokenized_query, n=k) # 假设有这个方法
        # top_scores = bm25_model.get_scores(tokenized_query)
        # results = []
        # for i in top_n_indices_bm25:
        #     doc_id = doc_ids_for_bm25[i]
        #     score = top_scores[i] #需要确认这个score是否准确对应
        #     text_data = worldview_texts_map.get(doc_id)
        #     if text_data:
        #         results.append((doc_id, score, text_data["full_text"]))

        # 更通用的方式，适用于 rank_bm25
        doc_scores = bm25_model.get_scores(tokenized_query)
        
        # 将分数与原始文档ID关联起来
        scored_doc_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)
        
        results = []
        for i in range(min(k, len(scored_doc_indices))):
            bm25_internal_idx = scored_doc_indices[i]
            doc_id = doc_ids_for_bm25[bm25_internal_idx] # 映射回全局ID
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
    """关键词搜索，返回 (doc_id, match_count, text_content)列表。"""
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
            
    # 按匹配关键词数量降序排序
    results_with_scores.sort(key=lambda x: x[1], reverse=True)
    return results_with_scores[:k]


def get_worldview_size() -> int:
    if not _active_world_id: return 0
    # FAISS索引大小作为主要衡量标准
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
        
        # 重建BM25模型
        if new_faiss_index.ntotal > 0:
            success_bm25 = _build_and_save_bm25_model_for_active_world(new_texts_map)
            if not success_bm25:
                print(f"警告 (rebuild_worldview_from_data): 为世界 '{get_world_display_name(_active_world_id)}' 重建BM25模型失败。")
        else:
            _clear_bm25_model_for_active_world() # 清空BM25如果世界观为空
        
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
    return len(get_kg_triples_for_active_world())import ollama
import numpy as np
from config import OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL_NAME

_ollama_client_for_embed = None
_embedding_dimension = None
_checked_model_name_for_dimension = None 

def get_ollama_client_for_embeddings():
    global _ollama_client_for_embed
    if _ollama_client_for_embed is None:
        try:
            print(f"正在初始化 Ollama 嵌入客户端以连接到: {OLLAMA_BASE_URL}")
            _ollama_client_for_embed = ollama.Client(host=OLLAMA_BASE_URL)
            _ollama_client_for_embed.list() 
            print(f"Ollama 嵌入客户端已连接。将使用模型 '{OLLAMA_EMBEDDING_MODEL_NAME}'。")
        except Exception as e:
            print(f"连接到 Ollama (用于嵌入) 或初始化客户端时发生错误: {e}")
            _ollama_client_for_embed = None
            raise
    return _ollama_client_for_embed

def get_model_embedding_dimension() -> int:
    global _embedding_dimension, _checked_model_name_for_dimension
    
    if _checked_model_name_for_dimension != OLLAMA_EMBEDDING_MODEL_NAME:
        _embedding_dimension = None 

    if _embedding_dimension is None:
        client = get_ollama_client_for_embeddings()
        try:
            print(f"首次调用或模型更改，正在为嵌入模型 '{OLLAMA_EMBEDDING_MODEL_NAME}' 获取嵌入维度...")
            test_prompt = "dimension_test_string" 
            if not OLLAMA_EMBEDDING_MODEL_NAME:
                 raise ValueError("OLLAMA_EMBEDDING_MODEL_NAME 未在 config.py 中配置。")

            response = client.embeddings(
                model=OLLAMA_EMBEDDING_MODEL_NAME,
                prompt=test_prompt
            )
            if 'embedding' not in response or not isinstance(response['embedding'], list) or not response['embedding']:
                raise ValueError(f"从Ollama模型 '{OLLAMA_EMBEDDING_MODEL_NAME}' 收到的嵌入响应格式不正确或为空: {response}")

            _embedding_dimension = len(response['embedding'])
            if _embedding_dimension == 0:
                 raise ValueError(f"模型 '{OLLAMA_EMBEDDING_MODEL_NAME}' 返回了零维度嵌入。")
            
            _checked_model_name_for_dimension = OLLAMA_EMBEDDING_MODEL_NAME 
            print(f"获取到嵌入模型 '{OLLAMA_EMBEDDING_MODEL_NAME}' 的维度: {_embedding_dimension}")

        except ollama.ResponseError as e:
            error_msg_detail = str(e)
            status_code_info = f" (Status Code: {e.status_code})" if hasattr(e, 'status_code') else ""
            if "model not found" in error_msg_detail.lower() or \
               (hasattr(e, 'status_code') and e.status_code == 404) or \
               "pull model" in error_msg_detail.lower(): 
                 error_msg = (f"错误：嵌入模型 '{OLLAMA_EMBEDDING_MODEL_NAME}' 在 Ollama 服务器上未找到{status_code_info}。"
                              f"请确保已拉取该模型 (例如: `ollama pull {OLLAMA_EMBEDDING_MODEL_NAME}`) 或检查config.py中的名称。")
            else:
                 error_msg = f"通过 Ollama API 获取嵌入维度时发生响应错误 (模型: {OLLAMA_EMBEDDING_MODEL_NAME}): {error_msg_detail}{status_code_info}"
            print(error_msg)
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"获取模型 '{OLLAMA_EMBEDDING_MODEL_NAME}' 嵌入维度时发生未知错误: {e}"
            print(error_msg)
            raise ValueError(f"无法确定模型 '{OLLAMA_EMBEDDING_MODEL_NAME}' 的嵌入维度。") from e
    return _embedding_dimension

def get_embedding(text: str) -> np.ndarray:
    client = get_ollama_client_for_embeddings()
    dimension = get_model_embedding_dimension() 

    if not text or not isinstance(text, str) or not text.strip():
        print(f"警告：接收到用于嵌入的空或无效文本: '{text}'。返回维度为 {dimension} 的零向量。")
        return np.zeros(dimension, dtype=np.float32)

    try:
        response = client.embeddings(
            model=OLLAMA_EMBEDDING_MODEL_NAME,
            prompt=text
        )
        embedding = np.array(response['embedding'], dtype=np.float32)
        
        if embedding.shape[0] != dimension:
            error_msg = (f"严重警告: 模型 '{OLLAMA_EMBEDDING_MODEL_NAME}' 返回的嵌入维度 ({embedding.shape[0]}) "
                         f"与程序启动时或模型更改后检测到的预期维度 ({dimension}) 不符。文本: '{text[:30]}...'。"
                         f"这可能表示Ollama服务器上的模型配置已更改，或者存在多个同名但版本/配置不同的模型实例。"
                         f"将返回零向量以避免数据损坏。请调查Ollama服务和模型状态。")
            print(error_msg)
            return np.zeros(dimension, dtype=np.float32)

        # --- 新增：检查是否为零向量 ---
        if np.all(embedding == 0):
            warning_msg = (f"警告 (get_embedding)：模型 '{OLLAMA_EMBEDDING_MODEL_NAME}' 为文本 (片段: '{text[:50]}...') "
                           f"返回了零向量。这通常表示模型未能正确处理输入或存在内部问题。")
            print(warning_msg)
            # 明确返回零向量，调用方需要意识到这一点
            return np.zeros(dimension, dtype=np.float32) 

        return embedding
        
    except ollama.ResponseError as e:
        # 明确指出错误来源和影响
        error_type_msg = f"Ollama响应错误 (模型: {OLLAMA_EMBEDDING_MODEL_NAME}, 文本片段: '{text[:50]}...'): {e}"
        if "model not found" in str(e).lower() or (hasattr(e, 'status_code') and e.status_code == 404):
            error_type_msg = (f"Ollama致命错误：嵌入模型 '{OLLAMA_EMBEDDING_MODEL_NAME}' 未找到。 "
                              f"文本片段: '{text[:50]}...'. 请确保模型已拉取并可用。")
        print(f"{error_type_msg}. 返回零向量。")
        return np.zeros(dimension, dtype=np.float32)
    except Exception as e:
        print(f"使用 Ollama 生成嵌入时发生未知错误 (文本片段: '{text[:50]}...'): {e}。返回零向量。")
        return np.zeros(dimension, dtype=np.float32)

def get_embeddings(texts: list[str]) -> np.ndarray:
    # 此函数不需要修改，因为它依赖于 get_embedding 的行为
    client = get_ollama_client_for_embeddings()
    dimension = get_model_embedding_dimension()

    if not texts:
        return np.empty((0, dimension), dtype=np.float32)

    all_embeddings = np.zeros((len(texts), dimension), dtype=np.float32) # Initialize with zeros

    for i, text_content in enumerate(texts):
        # get_embedding 内部会处理空文本、错误和零向量情况
        # 如果 get_embedding 返回零向量，它会被赋给 all_embeddings[i]
        all_embeddings[i] = get_embedding(text_content) 
            
    return all_embeddingsimport json
import os
from typing import List, Dict

from config import (
    OLLAMA_KG_EXTRACTION_MODEL, MAX_KG_TEXT_INPUT_LEN_LLM,
    KNOWLEDGE_SOURCE_JSON_FILENAME
)
from llm_utils import generate_text, get_ollama_client

# 只导入 database_utils 模块本身，或需要的其他函数
import database_utils # 或者 from database_utils import get_world_path, ...

def extract_triples_from_text_llm(text_content: str) -> List[List[str]]:
    if not text_content.strip():
        return []
    truncated_text = text_content
    if len(text_content) > MAX_KG_TEXT_INPUT_LEN_LLM:
        truncated_text = text_content[:MAX_KG_TEXT_INPUT_LEN_LLM] + "\n...[内容已截断]"
        print(f"KG提取 (LLM): 输入文本已从 {len(text_content)} 截断到 {len(truncated_text)} 字符。")
    
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
        model_name=OLLAMA_KG_EXTRACTION_MODEL
    )

    if llm_response_str.startswith("错误:") or llm_response_str.startswith("Error:"):
        print(f"LLM提取三元组失败: {llm_response_str}")
        return []

    extracted_triples = []
    try:
        # Attempt to find a JSON block, possibly within markdown code fences
        json_block_match = None
        # Try ```json ... ```
        json_start_tag = "```json"
        json_end_tag = "```"
        start_index_tag = llm_response_str.find(json_start_tag)
        if start_index_tag != -1:
            end_index_tag = llm_response_str.rfind(json_end_tag, start_index_tag + len(json_start_tag))
            if end_index_tag != -1:
                json_block_match = llm_response_str[start_index_tag + len(json_start_tag) : end_index_tag]
        
        # If not found, try ``` ... ``` (generic code block)
        if not json_block_match:
            json_start_tag_generic = "```"
            start_index_tag_generic = llm_response_str.find(json_start_tag_generic)
            if start_index_tag_generic != -1:
                end_index_tag_generic = llm_response_str.rfind(json_end_tag, start_index_tag_generic + len(json_start_tag_generic))
                if end_index_tag_generic != -1:
                     json_block_match = llm_response_str[start_index_tag_generic + len(json_start_tag_generic) : end_index_tag_generic]
        
        if json_block_match:
            llm_response_str = json_block_match.strip()
        else: 
            # Fallback: try to find first '[' and last ']'
            first_bracket = llm_response_str.find('[')
            last_bracket = llm_response_str.rfind(']')
            if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
                llm_response_str = llm_response_str[first_bracket : last_bracket + 1]
            # else: the string might be direct JSON or malformed

        parsed_json = json.loads(llm_response_str)
        if isinstance(parsed_json, list):
            for item in parsed_json:
                if isinstance(item, list) and len(item) == 3 and all(isinstance(s, str) for s in item):
                    # Further clean subject, predicate, object from leading/trailing whitespace
                    cleaned_item = [s.strip() for s in item]
                    # Ensure not all are empty after stripping
                    if any(s for s in cleaned_item):
                         extracted_triples.append(cleaned_item)
                    else:
                        print(f"KG提取 (LLM)：跳过完全由空白组成的三元组: {item}")
                else:
                    print(f"KG提取 (LLM)：跳过格式不正确的三元组: {item}")
        else:
            print(f"KG提取 (LLM)：LLM返回的JSON不是列表: {parsed_json}")

    except json.JSONDecodeError:
        print(f"KG提取 (LLM)：解析LLM返回的JSON失败。原始响应 (部分): {llm_response_str[:300]}")
    except Exception as e:
        print(f"KG提取 (LLM)：处理LLM响应时发生未知错误: {e}。原始响应 (部分): {llm_response_str[:300]}")
    
    return extracted_triples


def build_kg_for_active_world(progress_callback=None) -> str:
    """
    为当前活动世界从预定义的JSON文件构建或更新知识图谱。
    """
    # Access _active_world_id via the database_utils module's global variable
    # This is safe as long as database_utils._active_world_id is managed correctly
    current_active_world_id = database_utils._active_world_id 
    # print(f"[DEBUG] Inside build_kg_for_active_world: current_active_world_id (accessed via database_utils) = {current_active_world_id}")

    if not current_active_world_id:
        # print("[ERROR] build_kg_for_active_world: No active world ID detected.")
        return "错误：没有活动的存储世界来构建知识图谱。"

    # 使用从 database_utils 导入的其他函数
    world_name = database_utils.get_world_display_name(current_active_world_id)
    world_path = database_utils.get_world_path(current_active_world_id)
    
    if not world_path:
        # print(f"[ERROR] build_kg_for_active_world: Could not get world path for ID '{current_active_world_id}'.")
        return f"错误：无法获取世界 '{world_name}' (ID: {current_active_world_id}) 的路径。"

    source_json_path = os.path.join(world_path, KNOWLEDGE_SOURCE_JSON_FILENAME)

    if progress_callback:
        progress_callback(0.1, desc=f"检查知识源文件: {KNOWLEDGE_SOURCE_JSON_FILENAME}")

    if not os.path.exists(source_json_path):
        if progress_callback: progress_callback(1.0, desc="错误：源文件未找到")
        return (f"错误：在世界 '{world_name}' 目录中未找到知识源文件 '{KNOWLEDGE_SOURCE_JSON_FILENAME}'。\n"
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

    # 使用从 database_utils 导入的其他函数
    message = database_utils.add_triples_to_kg(loaded_triples, overwrite=True)
    final_count = database_utils.get_kg_triples_count_for_active_world()
    
    if progress_callback:
        progress_callback(1.0, desc="知识图谱构建完成！")

    if not loaded_triples and os.path.exists(source_json_path): # File exists but no valid triples
         return f"警告：从知识源文件 '{source_json_path}' 未加载到有效的三元组。世界 '{world_name}' 的知识图谱已被清空。当前总数: {final_count}。"

    return f"世界 '{world_name}' 知识图谱构建完成。从 '{KNOWLEDGE_SOURCE_JSON_FILENAME}' 共加载并存储 {final_count} 条三元组。"


# __main__ 部分的导入也需要调整，如果它也依赖 _active_world_id
if __name__ == '__main__':
    print("运行知识图谱构建工具 (kg_utils.py) 示例 - 从JSON文件加载...")
    
    from config import DATA_DIR, WORLD_KNOWLEDGE_GRAPH_FILENAME # WORLD_KNOWLEDGE_GRAPH_FILENAME for check
    # import database_utils # 导入整个模块
    # 或者只导入需要的函数
    from database_utils import (
        add_world, switch_active_world, get_available_worlds, 
        get_kg_triples_for_active_world, get_kg_triples_count_for_active_world, 
        get_world_display_name as get_world_display_name_db, 
        get_world_path as get_world_path_db
        # _active_world_id 不再在这里导入到 __main__ 的局部作用域
    )
    import shutil

    try:
        get_ollama_client() 
        print("Ollama LLM 客户端连接正常。")
    except Exception as e:
        print(f"LLM客户端初始化失败: {e} (这对于基于JSON的KG构建不是致命的，但其他功能可能受影响)")

    test_world_id = "kg_json_test_world_main"
    test_world_name = "知识图谱JSON测试世界 (主测试)"
    
    test_world_full_path = os.path.join(DATA_DIR, test_world_id)
    if os.path.exists(test_world_full_path):
        print(f"清理旧的测试世界 '{test_world_name}' 目录: {test_world_full_path}")
        shutil.rmtree(test_world_full_path)
    
    # 清理元数据，如果之前存在 (假设 database_utils._worlds_metadata 和 _save_worlds_metadata 可访问)
    # 更好的做法是 database_utils 提供一个清理元数据的函数
    if hasattr(database_utils, '_worlds_metadata') and test_world_id in database_utils._worlds_metadata:
        del database_utils._worlds_metadata[test_world_id]
        if hasattr(database_utils, '_save_worlds_metadata'):
            database_utils._save_worlds_metadata()


    if test_world_id not in get_available_worlds():
        add_msg = add_world(test_world_id, test_world_name)
        print(add_msg)
        if "错误" in add_msg:
            print("创建测试世界失败，中止示例。")
            exit()
    
    if not switch_active_world(test_world_id): # switch_active_world 会更新 database_utils._active_world_id
        print(f"激活测试世界 '{test_world_name}' 失败。示例中止。")
        exit()
    
    # 在 __main__ 中访问时，也应该通过 database_utils.来获取最新值
    print(f"已激活测试世界: '{get_world_display_name_db(database_utils._active_world_id)}' (ID: {database_utils._active_world_id})")
    
    active_world_actual_path_check = get_world_path_db(database_utils._active_world_id)
    if not active_world_actual_path_check or not os.path.isdir(active_world_actual_path_check):
        print(f"测试世界 '{test_world_name}' 的路径 '{active_world_actual_path_check}' 无效。示例中止。")
        exit()

    sample_kg_content = {
        "triples": [
            ["艾莉亚·史塔克", "是", "史塔克家族成员"],
            ["艾莉亚·史塔克", "拥有武器", "缝衣针"],
            ["君临城", "是", "七大王国的首都"],
            ["琼恩·雪诺", "曾是", "守夜人总司令"],
            ["龙", "可以", "喷火"],
            [" 冰与火之歌 ", " 作者是 ", " 乔治·R·R·马丁 "] # Test with leading/trailing spaces
        ]
    }
    source_json_full_path_check = os.path.join(active_world_actual_path_check, KNOWLEDGE_SOURCE_JSON_FILENAME)
    try:
        with open(source_json_full_path_check, 'w', encoding='utf-8') as f_json:
            json.dump(sample_kg_content, f_json, ensure_ascii=False, indent=2)
        print(f"已在 '{source_json_full_path_check}' 创建示例知识源文件。")
    except Exception as e_json:
        print(f"创建示例知识源文件失败: {e_json}")
        exit()

    print("\n开始执行知识图谱构建流程 (从JSON文件)...")
    
    def cli_progress_sim(progress_val, desc_str=""):
        bar_len = 30
        filled_len = int(round(bar_len * progress_val))
        bar_str = '█' * filled_len + '-' * (bar_len - filled_len)
        print(f'\r{desc_str} [{bar_str}] {progress_val*100:.1f}%', end='')
        if progress_val == 1.0:
            print() 

    build_status_msg = build_kg_for_active_world(progress_callback=cli_progress_sim) # 这个函数内部会用最新的 active_id
    print(f"\n构建状态: {build_status_msg}")

    final_triples_count_check = get_kg_triples_count_for_active_world() # 这个函数内部也会用最新的 active_id
    print(f"数据库中最终三元组数量: {final_triples_count_check}")

    final_triples_list_check = get_kg_triples_for_active_world() # 这个函数内部也会用最新的 active_id
    if final_triples_list_check:
        print(f"\n从 '{test_world_name}' 的知识图谱中加载到的三元组 (数据库内):")
        for i_triple, triple_item in enumerate(final_triples_list_check):
            print(f"  {i_triple+1}. {triple_item}")
        expected_count = len([t for t in sample_kg_content["triples"] if all(s.strip() for s in t)])
        if final_triples_count_check == expected_count:
            print(f"数量验证成功：期望 {expected_count}，实际 {final_triples_count_check}")
        else:
            print(f"数量验证注意：期望 {expected_count} (基于源文件有效条目)，实际 {final_triples_count_check}")
    else:
        print(f"\n未能从 '{test_world_name}' 的知识图谱中加载到任何三元组。")

    print("\n知识图谱构建 (kg_utils.py, 从JSON文件) 示例运行完毕。")
    print(f"你可以检查 '{source_json_full_path_check}' (源文件) 和 "
          f"'{os.path.join(active_world_actual_path_check, WORLD_KNOWLEDGE_GRAPH_FILENAME)}' (数据库存储文件)。")

    print("\n--- 测试LLM三元组提取功能 (extract_triples_from_text_llm) ---")
    sample_text_for_llm = "瓦里斯外号“八爪蜘蛛”，是君临城的情报总管。他服务于多位国王。"
    print(f"测试文本: \"{sample_text_for_llm}\"")
    if OLLAMA_KG_EXTRACTION_MODEL: 
        try:
            get_ollama_client() 
            llm_triples = extract_triples_from_text_llm(sample_text_for_llm)
            if llm_triples:
                print("LLM提取到的三元组:")
                for t in llm_triples: print(f"  {t}")
            else:
                print("LLM未能从测试文本中提取到三元组，或提取失败。")
        except Exception as e_llm_test:
            print(f"测试LLM提取功能时发生错误: {e_llm_test} (可能是Ollama服务或模型 '{OLLAMA_KG_EXTRACTION_MODEL}' 问题)")
    else:
        print("未配置 OLLAMA_KG_EXTRACTION_MODEL，跳过LLM提取测试。")import ollama
import re 
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_SUMMARY_MODEL, OLLAMA_KG_EXTRACTION_MODEL, MAX_SUMMARY_INPUT_LEN_LLM

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

            # 使用 config.py 中定义的模型名称
            models_to_check_map = {
                "主 LLM (扮演/拓展)": OLLAMA_MODEL,
                "总结 LLM": OLLAMA_SUMMARY_MODEL,
                "知识提取 LLM": OLLAMA_KG_EXTRACTION_MODEL
            }
            model_usage_counts = {} 
            for desc, model_name in models_to_check_map.items():
                model_usage_counts[model_name] = model_usage_counts.get(model_name, []) + [desc]
            
            for model_name, descs in model_usage_counts.items():
                if len(descs) > 1:
                    print(f"提示: 模型 '{model_name}' 被用于以下角色: {', '.join(descs)}。")

            unique_model_names_to_check = set(models_to_check_map.values()) 
            all_specified_models_available_flag = True

            if not available_model_names and unique_model_names_to_check: 
                print(f"警告：无法从Ollama获取可用模型列表。配置的 LLM ({', '.join(unique_model_names_to_check)}) 的可用性未知。")
                all_specified_models_available_flag = False
            else:
                for model_name_cfg in unique_model_names_to_check:
                    if not model_name_cfg: # Skip if a model role in config is empty
                        print(f"提示：一个或多个LLM角色 (主/总结/知识提取) 在 config.py 中未指定模型名称。")
                        continue
                    found = any(
                        name_from_ollama == model_name_cfg or \
                        name_from_ollama.startswith(model_name_cfg + ":") 
                        for name_from_ollama in available_model_names
                    )
                    if not found:
                        missing_model_descs = [desc for desc, m_name in models_to_check_map.items() if m_name == model_name_cfg]
                        print(f"警告：配置用于 '{', '.join(missing_model_descs)}' 的 LLM '{model_name_cfg}' 在 Ollama 中未找到。")
                        all_specified_models_available_flag = False
                    else:
                        available_model_descs = [desc for desc, m_name in models_to_check_map.items() if m_name == model_name_cfg]
                        print(f"配置用于 '{', '.join(available_model_descs)}' 的 LLM '{model_name_cfg}' 在 Ollama 中可用或部分匹配。")
            
            if not all_specified_models_available_flag:
                print(f"  Ollama 中当前可用的模型 (可能不完整或包含tag): {available_model_names if available_model_names else '列表为空或获取失败'}")
                missing_configured_models = [
                    model_name_cfg for model_name_cfg in unique_model_names_to_check 
                    if model_name_cfg and not any(name_from_ollama == model_name_cfg or name_from_ollama.startswith(model_name_cfg + ":") for name_from_ollama in available_model_names)
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


def generate_text(prompt: str, system_message: str = None, model_name: str = None) -> str:
    client = get_ollama_client()
    if not client:
        return "错误：Ollama 客户端不可用。"

    target_model = model_name if model_name else OLLAMA_MODEL
    if not target_model: # Fallback if OLLAMA_MODEL itself is not set
        return "错误：未在 config.py 中配置有效的 LLM 模型 (OLLAMA_MODEL)。"
        
    messages = []
    if system_message:
        messages.append({'role': 'system', 'content': system_message})
    messages.append({'role': 'user', 'content': prompt})

    try:
        print(f"正在使用模型 '{target_model}' 生成文本...")
        response = client.chat(
            model=target_model,
            messages=messages
        )
        generated_content = response['message']['content']
        print(f"模型 '{target_model}' 响应长度 (原始): {len(generated_content)}")

        cleaned_content = remove_think_tags(generated_content)
        print(f"模型 '{target_model}' 响应长度 (清理后): {len(cleaned_content)}")
        
        return cleaned_content 
        
    except ollama.ResponseError as e: 
        error_message_detail = str(e)
        status_code_info = f" (Status Code: {e.status_code})" if hasattr(e, 'status_code') else ""
        if "model not found" in error_message_detail.lower() or \
           (hasattr(e, 'status_code') and e.status_code == 404) or \
           "pull model" in error_message_detail.lower(): # Added check for "pull model"
             error_message = (f"错误：LLM 模型 '{target_model}' 在 Ollama 服务器上未找到{status_code_info}。"
                              f"请拉取该模型 (例如: `ollama pull {target_model}`) 或检查其名称。")
        else:
             error_message = f"Error: 调用 Ollama API 时发生响应错误 (模型: {target_model}): {error_message_detail}{status_code_info}"
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"Error: 调用 Ollama API 时发生未知错误 (模型: {target_model}): {e}"
        print(error_message)
        return error_message

def generate_summary(text_to_summarize: str, max_input_len: int = None) -> str:
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
    
    summary_model_to_use = OLLAMA_SUMMARY_MODEL
    if not summary_model_to_use:
        print("警告：未配置 OLLAMA_SUMMARY_MODEL，将尝试使用 OLLAMA_MODEL 进行总结。")
        summary_model_to_use = OLLAMA_MODEL
    if not summary_model_to_use: # If both are None/empty
        error_msg = "错误：总结功能需要配置 OLLAMA_SUMMARY_MODEL 或 OLLAMA_MODEL。"
        print(error_msg)
        return error_msg


    summary = generate_text(prompt, system_message=system_message, model_name=summary_model_to_use)
    
    if summary.startswith("错误:") or summary.startswith("Error:"):
        print(f"使用 LLM 生成概要失败。将返回原始文本的前N个字符作为备用。错误: {summary}")
        fallback_len = min(len(text_to_summarize), 500) 
        return text_to_summarize[:fallback_len] + ("..." if len(text_to_summarize) > fallback_len else "")
    
    return summary.strip()   import gradio as gr
import time
import os
from typing import Optional, List, Dict, Tuple, Any
import numpy as np 
import jieba 
import logging 

import config 

import database_utils 
from llm_utils import generate_text, get_ollama_client 
from embedding_utils import get_model_embedding_dimension, get_embedding 
from compression_utils import compress_worldview_db_for_active_world 
from kg_utils import build_kg_for_active_world 
import json 
from rerank_utils import rerank_documents, get_rerank_model 

from config import ( 
    COMPRESSION_THRESHOLD, OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL_NAME,
    OLLAMA_KG_EXTRACTION_MODEL,
    HYBRID_SEARCH_ENABLED, SEMANTIC_SEARCH_TOP_K_HYBRID, 
    BM25_SEARCH_TOP_K_HYBRID, KEYWORD_SEARCH_TOP_K_HYBRID,
    HYBRID_RRF_K, HYBRID_FINAL_TOP_K,
    RERANK_ENABLED, RERANK_TOP_K_FINAL 
)
from database_utils import (
    add_world, get_available_worlds, switch_active_world, get_world_display_name, delete_world,
    add_character, get_character, get_character_names, get_all_characters, delete_character,
    add_worldview_text, 
    get_worldview_size,
    search_worldview_semantic, search_worldview_bm25, search_worldview_keyword, 
    _load_worldview_texts_map 
)

jieba.setLogLevel(logging.INFO)

_initial_world_activated_on_startup = False
_initial_active_world_id_on_startup = None

# --- 初始化对话历史状态 ---
# 这个状态将与Gradio UI的生命周期绑定
# 我们为每个 (world_id, character_name) 组合维护一个独立的对话历史
# 为了简化，这里先用一个全局的，更完善的设计可能需要基于session或更复杂的key
# 或者，让对话历史成为UI组件的一部分，由Gradio的State管理
# conversation_history_state = gr.State({}) # key: (world_id, char_name), value: List[Tuple[str, str]]
# 改为在UI中定义 gr.State


try:
    print("正在执行 main_app.py 的初始化...")
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        print(f"主数据目录 '{config.DATA_DIR}' 已创建。")

    get_model_embedding_dimension() 
    get_ollama_client() 
    if RERANK_ENABLED: 
        get_rerank_model()

    available_worlds_init = get_available_worlds()
    if available_worlds_init:
        first_world_id = list(available_worlds_init.keys())[0]
        if switch_active_world(first_world_id):
            _initial_world_activated_on_startup = True
            _initial_active_world_id_on_startup = database_utils._active_world_id
            print(f"已在启动时自动激活默认世界: '{get_world_display_name(first_world_id)}'")
        else:
            print(f"警告：启动时未能自动激活默认世界 '{first_world_id}'。")
    else:
        print("提示：当前没有已创建的世界。请在'世界管理'标签页中添加新世界。")
    print("main_app.py 初始化完成。")
except Exception as e:
    print(f"致命错误：main_app.py 初始设置期间出错: {e}")
    import traceback
    traceback.print_exc()
    _initial_active_world_id_on_startup = f"初始化错误: {e}"


def refresh_world_dropdown_choices_for_gradio():
    available_worlds = get_available_worlds()
    return [(name, id_val) for id_val, name in available_worlds.items()]

def refresh_character_dropdown_choices():
    if not database_utils._active_world_id:
        return []
    return get_character_names()

def refresh_worldview_status_display_text():
    if not database_utils._active_world_id:
        return "无活动世界。世界观信息不可用。"
    size = get_worldview_size()
    world_name = get_world_display_name(database_utils._active_world_id)
    status_text = f"世界 '{world_name}' | 世界观条目数: {size}. "
    ct = COMPRESSION_THRESHOLD
    if isinstance(ct, (int, float)) and ct > 0:
        if size > ct: status_text += f"建议压缩 (阈值: {ct})."
        else: status_text += f"压缩阈值: {ct}."
    else: status_text += "压缩阈值未有效配置或为零。"
    return status_text

def refresh_kg_status_display_text():
    if not database_utils._active_world_id:
        return "无活动世界。知识图谱信息不可用。"
    count = database_utils.get_kg_triples_count_for_active_world()
    world_name = get_world_display_name(database_utils._active_world_id)
    return f"世界 '{world_name}' | 知识图谱三元组数量: {count}."


def get_active_world_markdown_text_for_global_display():
    active_id = database_utils._active_world_id
    if isinstance(active_id, str) and "初始化错误" in active_id: 
        return f"<p style='color:red; font-weight:bold;'>应用初始化失败: {active_id}. 请检查控制台日志和Ollama服务。</p>"
    if active_id:
        return f"当前活动世界: **'{get_world_display_name(active_id)}'** (ID: `{active_id}`)"
    else:
        return "<p style='color:orange; font-weight:bold;'>当前无活动世界。请从上方选择或在“世界管理”中创建一个新世界。</p>"

def clear_textboxes_and_checkboxes(*args):
    updates = []
    for arg_comp in args:
        comp_type_str = str(type(arg_comp))
        if 'Checkbox' in comp_type_str:
            updates.append(gr.Checkbox(value=False))
        elif 'Textbox' in comp_type_str:
            updates.append(gr.Textbox(value=""))
        elif 'Dropdown' in comp_type_str:
            updates.append(gr.Dropdown(value=None)) 
        else: 
            updates.append(gr.update(value=None)) 
    return tuple(updates) if updates else gr.update()


def update_all_ui_elements_after_world_change(feedback_message: str = "", specific_feedback_elem_id: Optional[str] = None):
    world_choices_dd = refresh_world_dropdown_choices_for_gradio()
    current_active_id = database_utils._active_world_id
    is_world_active = current_active_id is not None and not (isinstance(current_active_id, str) and "初始化错误" in current_active_id)

    char_choices_dd = refresh_character_dropdown_choices() 
    wv_status_text_val = refresh_worldview_status_display_text()
    kg_status_text_val = refresh_kg_status_display_text()
    global_md_text_val = get_active_world_markdown_text_for_global_display()
    
    can_interact_with_world_features = is_world_active
    predict_button_interactive = can_interact_with_world_features and bool(char_choices_dd) 

    updates_map = {
        "world_select_dropdown": gr.Dropdown(choices=world_choices_dd, value=current_active_id if is_world_active else None, interactive=True),
        "global_active_world_display": gr.Markdown(value=global_md_text_val),
        "new_world_id_input": gr.Textbox(value="", interactive=True),
        "new_world_name_input": gr.Textbox(value="", interactive=True),
        "add_world_button": gr.Button(interactive=True),
        "confirm_delete_world_checkbox": gr.Checkbox(value=False, interactive=can_interact_with_world_features),
        "delete_world_button": gr.Button(interactive=False), 
        "char_name_input": gr.Textbox(value="", interactive=can_interact_with_world_features),
        "char_full_desc_input": gr.Textbox(value="", interactive=can_interact_with_world_features),
        "add_char_button": gr.Button(interactive=can_interact_with_world_features),
        "character_select_for_delete_dropdown": gr.Dropdown(choices=char_choices_dd, value=None, interactive=can_interact_with_world_features and bool(char_choices_dd)),
        "delete_char_button": gr.Button(interactive=False), 
        "view_chars_button": gr.Button(interactive=can_interact_with_world_features),
        "worldview_text_input": gr.Textbox(value="", interactive=can_interact_with_world_features),
        "add_worldview_button": gr.Button(interactive=can_interact_with_world_features),
        "worldview_status_display": gr.Textbox(value=wv_status_text_val, interactive=False),
        "compress_worldview_button": gr.Button(interactive=can_interact_with_world_features and get_worldview_size() > 1), 
        "kg_status_display": gr.Textbox(value=kg_status_text_val, interactive=False),
        "build_kg_button": gr.Button(interactive=can_interact_with_world_features),
        "char_select_dropdown_pred_tab": gr.Dropdown(choices=char_choices_dd, value=None, interactive=can_interact_with_world_features and bool(char_choices_dd)),
        "situation_query_input": gr.Textbox(value="", interactive=can_interact_with_world_features and bool(char_choices_dd)), 
        "predict_button": gr.Button(interactive=predict_button_interactive),
        "world_switch_feedback": gr.Textbox(value="", interactive=False),
        "add_world_feedback_output": gr.Textbox(value="", interactive=False),
        "delete_world_feedback_output": gr.Textbox(value="", interactive=False),
        "char_op_feedback_output": gr.Textbox(value="", interactive=False),
        "view_characters_output": gr.Textbox(value="角色列表将显示在此处。" if can_interact_with_world_features else "无活动世界或初始化错误。", interactive=False),
        "worldview_feedback_output": gr.Textbox(value="", interactive=False),
        "compression_status_output": gr.Textbox(value="", interactive=False),
        "kg_build_status_output": gr.Textbox(value="", interactive=False),
        # "prediction_output": gr.Textbox(value="", interactive=False), # chatbot 会替代这个
        "retrieved_info_display": gr.Markdown(value="检索到的背景信息将显示在此处。" if can_interact_with_world_features else "无活动世界或初始化错误。"),
        "chatbot_display": gr.Chatbot(value=[], label="对话历史", visible=True if is_world_active else False) # 新增
    }

    if specific_feedback_elem_id and specific_feedback_elem_id in updates_map:
        component_instance = app.blocks.get(specific_feedback_elem_id) # type: ignore
        if component_instance and isinstance(component_instance, gr.Textbox):
            updates_map[specific_feedback_elem_id] = gr.Textbox(value=feedback_message, interactive=False)
        elif component_instance: 
            updates_map[specific_feedback_elem_id] = gr.update(value=feedback_message) # type: ignore
    elif feedback_message and not specific_feedback_elem_id: 
        if "world_switch_feedback" in updates_map:
             updates_map["world_switch_feedback"] = gr.Textbox(value=feedback_message, interactive=False)
    return updates_map


def handle_add_world(world_id_input: str, world_name_input: str):
    feedback_msg = ""
    if not world_id_input.strip() or not world_name_input.strip():
        feedback_msg = "错误：世界ID和世界名称不能为空。"
    else:
        message = add_world(world_id_input.strip(), world_name_input.strip())
        feedback_msg = message
        if "已添加" in message:
            if switch_active_world(world_id_input.strip()):
                world_name_disp = get_world_display_name(world_id_input.strip())
                feedback_msg += f" 并已激活 '{world_name_disp}'。"
            else:
                feedback_msg += " 但激活失败，请手动选择。"
    all_updates_dict = update_all_ui_elements_after_world_change(feedback_msg, "add_world_feedback_output")
    # 当添加世界后，也需要清空对话历史UI和状态
    all_updates_dict["chatbot_display"] = gr.Chatbot(value=[])
    all_updates_dict["conversation_history_state"] = [] # 清空状态
    return map_updates_to_ordered_list(all_updates_dict, ordered_output_components_for_ui_updates)

def handle_delete_world(confirm_delete_checkbox: bool):
    feedback_msg = ""
    world_id_to_delete = database_utils._active_world_id
    is_valid_world_to_delete = world_id_to_delete is not None and \
                               isinstance(world_id_to_delete, str) and \
                               not ("初始化错误" in world_id_to_delete)
    if not is_valid_world_to_delete:
        feedback_msg = "错误：没有活动的或选中的有效世界可供删除。"
    elif not confirm_delete_checkbox:
        feedback_msg = "错误：请勾选确认框以删除当前活动世界。"
    else:
        world_name_to_delete = get_world_display_name(world_id_to_delete)
        message = delete_world(world_id_to_delete) 
        feedback_msg = message
        if "已成功删除" in message or "数据目录可能未能完全删除" in message : 
            feedback_msg = f"世界 '{world_name_to_delete}' 相关操作已执行：{message}"
    all_updates_dict = update_all_ui_elements_after_world_change(feedback_msg, "delete_world_feedback_output")
    all_updates_dict["chatbot_display"] = gr.Chatbot(value=[])
    all_updates_dict["conversation_history_state"] = []
    return map_updates_to_ordered_list(all_updates_dict, ordered_output_components_for_ui_updates)

def handle_switch_world(world_id_selected: Optional[str]):
    feedback_msg = ""
    if not world_id_selected: 
        if database_utils._active_world_id is not None and not (isinstance(database_utils._active_world_id, str) and "初始化错误" in database_utils._active_world_id):
            switch_active_world(None) 
            feedback_msg = "已取消活动世界。"
    elif isinstance(world_id_selected, str) and "初始化错误" in world_id_selected: 
        feedback_msg = "无法选择一个错误状态作为活动世界。" 
        switch_active_world(None) 
    elif switch_active_world(world_id_selected):
        world_name_disp = get_world_display_name(world_id_selected)
        feedback_msg = f"已激活世界: '{world_name_disp}'"
    else: 
        feedback_msg = f"切换到世界ID '{world_id_selected}' 失败。该世界可能已损坏或不存在。"
        switch_active_world(None) 
    
    all_updates_dict = update_all_ui_elements_after_world_change(feedback_msg, "world_switch_feedback")
    # 切换世界或角色时，清空对话历史
    all_updates_dict["chatbot_display"] = gr.Chatbot(value=[])
    all_updates_dict["conversation_history_state"] = [] 
    return map_updates_to_ordered_list(all_updates_dict, ordered_output_components_for_ui_updates)

def handle_character_selection_change(char_name: Optional[str]):
    """当预测标签页的角色选择改变时，清空对话历史。"""
    is_world_truly_active = database_utils._active_world_id is not None and \
                           not (isinstance(database_utils._active_world_id, str) and "初始化错误" in database_utils._active_world_id)
    can_predict = bool(char_name) and is_world_truly_active
    return gr.Chatbot(value=[]), [], gr.Textbox(interactive=can_predict), gr.Button(interactive=can_predict)


def handle_add_character(name: str, full_description: str):
    feedback_msg = "请先选择并激活一个有效世界。"
    is_world_truly_active = database_utils._active_world_id is not None and \
                           not (isinstance(database_utils._active_world_id, str) and "初始化错误" in database_utils._active_world_id)
    if is_world_truly_active:
        if not name.strip() or not full_description.strip():
            feedback_msg = "角色名称和完整描述不能为空。"
        else:
            message = add_character(name.strip(), full_description.strip())
            feedback_msg = message
    char_choices = refresh_character_dropdown_choices()
    can_interact_char_dd = is_world_truly_active and bool(char_choices)
    char_choices_updated_pred_tab = gr.Dropdown(choices=char_choices, value=None, interactive=can_interact_char_dd)
    char_choices_updated_del_tab = gr.Dropdown(choices=char_choices, value=None, interactive=can_interact_char_dd)
    predict_btn_interactive = is_world_truly_active and bool(char_choices)
    situation_input_interactive = is_world_truly_active and bool(char_choices)
    # 添加角色不直接清空对话历史，因为可能是在为当前对话的角色更新信息
    return (feedback_msg, char_choices_updated_pred_tab, char_choices_updated_del_tab, 
            gr.Button(interactive=predict_btn_interactive), gr.Textbox(interactive=situation_input_interactive))

def handle_delete_character(character_name_to_delete: Optional[str]):
    feedback_msg = "错误：请先选择有效活动世界。"
    is_world_truly_active = database_utils._active_world_id is not None and \
                           not (isinstance(database_utils._active_world_id, str) and "初始化错误" in database_utils._active_world_id)
    if is_world_truly_active:
        if not character_name_to_delete:
            feedback_msg = "错误：请从下拉列表中选择要删除的角色。"
        else:
            message = delete_character(character_name_to_delete)
            feedback_msg = message
    char_choices = refresh_character_dropdown_choices() 
    can_interact_char_dd = is_world_truly_active and bool(char_choices)
    char_choices_updated_pred_tab = gr.Dropdown(choices=char_choices, value=None, interactive=can_interact_char_dd)
    char_choices_updated_del_tab = gr.Dropdown(choices=char_choices, value=None, interactive=can_interact_char_dd)
    predict_btn_interactive = is_world_truly_active and bool(char_choices)
    situation_input_interactive = is_world_truly_active and bool(char_choices)
    # 如果删除的是当前对话的角色，理论上也应该清空对话历史，但Gradio的事件流可能复杂，
    # 暂时依赖用户在切换角色时（通过 char_select_dropdown_pred_tab.change）清空
    return (feedback_msg, char_choices_updated_pred_tab, char_choices_updated_del_tab, 
            gr.Button(interactive=predict_btn_interactive), gr.Textbox(interactive=situation_input_interactive))

def handle_view_characters():
    if not database_utils._active_world_id or not isinstance(database_utils._active_world_id, str) or "初始化错误" in database_utils._active_world_id: 
        return "请先选择并激活一个有效世界，或修复初始化错误。"
    chars = get_all_characters()
    world_name_active = get_world_display_name(database_utils._active_world_id)
    if not chars: return f"当前活动世界 '{world_name_active}' 中没有角色。"
    output = f"当前活动世界 '{world_name_active}' 的角色列表 ({len(chars)} 个):\n" + "="*40 + "\n"
    for char_data in chars:
        desc_to_show = char_data.get('summary_description', '').strip()
        if not desc_to_show or len(desc_to_show) < 20: 
            desc_to_show = char_data.get('full_description', '（无详细描述）')
        desc_snippet = desc_to_show.replace('\n', ' ').replace('\r', '') 
        desc_snippet = (desc_snippet[:150] + '...') if len(desc_snippet) > 150 else desc_snippet
        output += f"  名称: {char_data['name']}\n"
        output += f"  设定/概要: {desc_snippet}\n---\n"
    return output

def handle_add_worldview(text_input: str):
    feedback_msg = "请先选择并激活一个有效世界。"
    wv_status_update = gr.Textbox(value=refresh_worldview_status_display_text(), interactive=False)
    compress_btn_interactive_val = False 
    is_world_truly_active = database_utils._active_world_id is not None and \
                           not (isinstance(database_utils._active_world_id, str) and "初始化错误" in database_utils._active_world_id)
    if is_world_truly_active:
        if not text_input.strip():
            feedback_msg = "世界观文本不能为空。"
        else:
            message = add_worldview_text(text_input.strip()) 
            feedback_msg = message 
            wv_status_update = gr.Textbox(value=refresh_worldview_status_display_text(), interactive=False) 
            if not message.startswith("错误："):
                 current_size = get_worldview_size()
                 compress_btn_interactive_val = current_size > 1
                 ct_val = COMPRESSION_THRESHOLD
                 auto_compress_message = ""
                 if isinstance(ct_val, (int, float)) and ct_val > 0 and current_size > ct_val:
                     if current_size > ct_val * 1.5: 
                         auto_compress_message = f"\n提示: 世界观条目数 ({current_size}) 远超压缩阈值 ({ct_val})。建议手动压缩。"
                     else:
                         auto_compress_message = f"\n提示: 世界观条目数 ({current_size}) 已达到或接近压缩阈值 ({ct_val})。可考虑手动压缩。"
                 if auto_compress_message and auto_compress_message not in feedback_msg:
                    feedback_msg += auto_compress_message
    compress_btn_interactive = gr.Button(interactive=compress_btn_interactive_val and is_world_truly_active)
    return feedback_msg, wv_status_update, compress_btn_interactive

def handle_compress_worldview_button_streaming():
    initial_wv_status_text = refresh_worldview_status_display_text()
    is_world_truly_active = database_utils._active_world_id is not None and \
                           not (isinstance(database_utils._active_world_id, str) and "初始化错误" in database_utils._active_world_id)
    if not is_world_truly_active:
        yield "错误：请先选择并激活一个有效世界才能进行压缩。", initial_wv_status_text, gr.Button(interactive=False)
        return
    world_name_active = get_world_display_name(database_utils._active_world_id)
    yield f"正在为世界 '{world_name_active}' 压缩世界观数据库... 这可能需要一些时间。", initial_wv_status_text, gr.Button(interactive=False) 
    message = ""
    try:
        message = compress_worldview_db_for_active_world(force_compress=True)
    except Exception as e_compress:
        message = f"为世界 '{world_name_active}' 压缩时发生严重错误: {e_compress}"
        import traceback
        traceback.print_exc()
    final_wv_status_text = refresh_worldview_status_display_text()
    can_compress_again = is_world_truly_active and get_worldview_size() > 1
    yield message, final_wv_status_text, gr.Button(interactive=can_compress_again)

def simple_entity_extractor(text: str) -> List[str]:
    text_cleaned = text
    for punc in "，。？！；：“”‘’（）《》〈〉【】〖〗「」『』": 
        text_cleaned = text_cleaned.replace(punc, " ")
    for punc in """!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~""": 
        text_cleaned = text_cleaned.replace(punc, " ")
    words = text_cleaned.split()
    entities = list(set([word.lower().strip() for word in words if len(word) > 1 and not word.isnumeric() and word.strip()]))
    return entities

def _reciprocal_rank_fusion(
    retrieved_results_dict: Dict[str, List[Tuple[int, float, str]]], 
    rrf_k_param: int = 60
) -> List[int]:
    fused_scores: Dict[int, float] = {}
    for method, results in retrieved_results_dict.items():
        ranked_doc_ids = [doc_id for doc_id, _, _ in results]
        for rank, doc_id in enumerate(ranked_doc_ids): 
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
            fused_scores[doc_id] += 1.0 / (rrf_k_param + rank + 1) 
    if not fused_scores:
        return []
    sorted_fused_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, score in sorted_fused_results]

def format_conversation_history(history: List[Tuple[str, str]], max_turns: int = 3) -> str:
    """格式化对话历史用于Prompt，只取最近几轮。"""
    if not history:
        return ""
    
    formatted_history_parts = []
    # 取最后 max_turns 轮对话
    recent_history = history[-max_turns:]
    for user_turn, ai_turn in recent_history:
        formatted_history_parts.append(f"你之前问/说：{user_turn}")
        formatted_history_parts.append(f"我（{character_name_for_history_context or '角色'}）当时回应：{ai_turn}") # 需要一个全局或上下文的角色名
    
    if formatted_history_parts:
        return "\n\n### 最近的对话回顾:\n" + "\n".join(formatted_history_parts) + "\n"
    return ""

character_name_for_history_context: Optional[str] = None # 用于在format_conversation_history中引用角色名

def handle_predict_behavior(
    character_name: Optional[str], 
    situation_query: str, 
    # conversation_history: List[Tuple[str, str]], # 从State获取
    # 修改：接收chatbot的当前值和history state
    chatbot_ui_value: List[List[Optional[str]]], # Gradio Chatbot的格式 [[user, ai], [user, ai]]
    history_state_value: List[Tuple[str,str]], # 我们内部存储的格式 [(user, ai), (user, ai)]
    progress=gr.Progress(track_tqdm=True)
):
    global character_name_for_history_context # 声明使用全局变量以便更新
    character_name_for_history_context = character_name # 更新当前对话的角色名

    # 将 chatbot_ui_value (可能包含None) 转换为 history_state_value 的格式
    # 同时，如果 history_state_value 为空，尝试从 chatbot_ui_value 初始化
    current_conversation_history: List[Tuple[str,str]] = []
    if history_state_value: #优先使用state中的历史
        current_conversation_history = history_state_value
    elif chatbot_ui_value: # 如果state为空，尝试从chatbot组件值恢复
        for pair in chatbot_ui_value:
            if pair[0] is not None and pair[1] is not None:
                current_conversation_history.append((str(pair[0]), str(pair[1])))
    
    # 更新 chatbot UI 的显示，确保用户的新输入先加上
    # situation_query 是用户当前输入，AI的响应还没生成
    # chatbot_ui_value.append([situation_query, None]) # 这种方式更新UI后，AI响应再追加会覆盖None


    is_world_truly_active = database_utils._active_world_id is not None and \
                           not (isinstance(database_utils._active_world_id, str) and "初始化错误" in database_utils._active_world_id)

    if not is_world_truly_active:
        # yield "错误：请先选择并激活一个有效活动世界。", "调试信息：无有效活动世界。"
        # 修改返回，以匹配新的输出结构
        current_conversation_history.append((situation_query, "错误：请先选择并激活一个有效活动世界。"))
        yield current_conversation_history, current_conversation_history, "调试信息：无有效活动世界。" # chatbot, state, debug_md
        return
    if not character_name:
        current_conversation_history.append((situation_query, "错误：请选择一个角色。"))
        yield current_conversation_history, current_conversation_history, f"调试信息：未选择角色 (当前活动世界: {get_world_display_name(database_utils._active_world_id)})。"
        return
    if not situation_query.strip():
        # 对于空输入，我们不添加到历史，直接提示错误
        # yield current_conversation_history, current_conversation_history, "错误：请输入情境或问题。" # 这种方式空输入不显示在chatbot
        # 或者，也把空输入和错误加入历史，让用户看到
        current_conversation_history.append((situation_query, "错误：请输入情境或问题。"))
        yield current_conversation_history, current_conversation_history, f"调试信息：情境为空 (角色: {character_name}, 世界: {get_world_display_name(database_utils._active_world_id)})。"
        return

    # 在处理前，将当前用户输入加入历史（但不立即显示AI响应）
    # AI响应生成后，再更新这一轮的AI部分
    # display_history_for_chatbot = current_conversation_history + [(situation_query, "...思考中...")]
    # yield display_history_for_chatbot, current_conversation_history, "正在处理..." # 更新chatbot UI, state不变, debug信息

    progress(0, desc="准备中...")
    retrieved_info_log = [f"## 检索信息日志 (角色: {character_name}, 原始情境: '{situation_query[:50]}...')"]
    retrieved_info_log.append(f"**当前活动世界:** {get_world_display_name(database_utils._active_world_id)} (ID: {database_utils._active_world_id})")

    character_data = get_character(character_name)
    world_name_active = get_world_display_name(database_utils._active_world_id)

    if not character_data:
        error_msg = f"错误：在当前世界 '{world_name_active}' 中未找到角色 '{character_name}'。"
        retrieved_info_log.append(f"\n**错误:** {error_msg}")
        current_conversation_history.append((situation_query, error_msg))
        yield current_conversation_history, current_conversation_history, "\n".join(retrieved_info_log)
        return

    char_desc_for_prompt = character_data.get('summary_description', '').strip()
    if not char_desc_for_prompt:
        char_desc_for_prompt = character_data.get('full_description', '（该角色无可用描述）')
    
    char_desc_for_llm_prompt_truncated = (char_desc_for_prompt[:800] + "...") if len(char_desc_for_prompt) > 800 else char_desc_for_prompt
    retrieved_info_log.append(f"\n### 角色核心设定 ({character_name}):")
    retrieved_info_log.append(f"```text\n{char_desc_for_llm_prompt_truncated}\n```")

    progress(0.05, desc="拓展情境以便更好检索...")
    expansion_system_message = f"你是一个极具推理能力同时富有想象力的侦探，专门对缺乏信息的场景进行信息补全和还原，使得情境的逻辑链条更加完整逼真，任务是为{character_name}当前遇到的情景补充可能需要用到的信息。"
    # 如果有对话历史，也一并考虑进来辅助拓展
    formatted_hist_for_expansion = format_conversation_history(current_conversation_history, max_turns=2)

    expansion_prompt = f"""
角色：{character_name}
角色核心设定概要：{char_desc_for_llm_prompt_truncated[:300]}... 
{formatted_hist_for_expansion if formatted_hist_for_expansion else ""}
当前简要情境："{situation_query}"
你所需要做的就是推测可能需要的信息，并以想象的形式还原前因后果的各种细节内容，要求尽可能符合逻辑。
请直接生成拓展后的文本，无需任何额外解释、对话或标签。确保文本内容具体且包含潜在的关键词。
拓展后的文本："""
    expanded_situation_for_search = generate_text(prompt=expansion_prompt, system_message=expansion_system_message, model_name=OLLAMA_MODEL)
    
    effective_query_for_search = situation_query 
    if expanded_situation_for_search.startswith("错误:") or expanded_situation_for_search.startswith("Error:"):
        retrieved_info_log.append(f"\n**警告:** 情境拓展失败: {expanded_situation_for_search}。将使用原始情境进行检索。")
    elif not expanded_situation_for_search.strip():
        retrieved_info_log.append(f"\n**警告:** 情境拓展返回空内容。将使用原始情境进行检索。")
    else:
        retrieved_info_log.append(f"\n### 情境拓展结果 (用于检索):")
        retrieved_info_log.append(f"```text\n{expanded_situation_for_search}\n```")
        effective_query_for_search = expanded_situation_for_search

    progress(0.1, desc="检索相关世界背景知识...")
    retrieved_worldview_texts_for_prompt_map: Dict[int, str] = {}
    all_retrieved_from_methods: Dict[str, List[Tuple[int, float, str]]] = {}

    retrieved_info_log.append("\n### 1. 语义相似度检索 (FAISS):")
    semantic_results = search_worldview_semantic(effective_query_for_search, k=SEMANTIC_SEARCH_TOP_K_HYBRID)
    if semantic_results:
        all_retrieved_from_methods["semantic"] = semantic_results
        # ... (日志记录部分不变)
    else:
        retrieved_info_log.append("- 未找到相关结果。")

    retrieved_info_log.append("\n### 2. BM25 关键词相关度检索:")
    bm25_results = search_worldview_bm25(effective_query_for_search, k=BM25_SEARCH_TOP_K_HYBRID)
    if bm25_results:
        all_retrieved_from_methods["bm25"] = bm25_results
        # ... (日志记录部分不变)
    else:
        retrieved_info_log.append("- 未找到相关结果 (可能BM25模型未构建或查询无匹配)。")

    retrieved_info_log.append("\n### 3. 简单关键词匹配检索:")
    keyword_search_results_raw = search_worldview_keyword(effective_query_for_search, k=KEYWORD_SEARCH_TOP_K_HYBRID)
    if keyword_search_results_raw:
        keyword_results_processed = [(doc_id, float(score), text) for doc_id, score, text in keyword_search_results_raw]
        all_retrieved_from_methods["keyword"] = keyword_results_processed
        # ... (日志记录部分不变)
    else:
        retrieved_info_log.append("- 未找到相关结果。")

    fused_doc_ids_after_rrf: List[int] = []
    if HYBRID_SEARCH_ENABLED and all_retrieved_from_methods:
        retrieved_info_log.append("\n### 4. 混合检索结果融合 (RRF):")
        progress(0.2, desc="RRF融合检索结果...")
        fused_doc_ids_after_rrf = _reciprocal_rank_fusion(all_retrieved_from_methods, rrf_k_param=HYBRID_RRF_K)
        docs_for_rerank_ids = fused_doc_ids_after_rrf[:HYBRID_FINAL_TOP_K]
        # ... (日志记录部分不变)
        fused_doc_ids_after_rrf = docs_for_rerank_ids 
    elif not HYBRID_SEARCH_ENABLED and semantic_results: 
        retrieved_info_log.append("\n### 4. 仅语义检索结果 (混合检索未启用):")
        fused_doc_ids_after_rrf = [doc_id for doc_id, _, _ in semantic_results[:SEMANTIC_SEARCH_TOP_K_HYBRID]]

    final_doc_ids_for_prompt: List[int] = []
    if RERANK_ENABLED and fused_doc_ids_after_rrf:
        retrieved_info_log.append("\n### 5. Rerank 重排序:")
        progress(0.22, desc="Rerank重排序...")
        worldview_texts_map = _load_worldview_texts_map()
        docs_to_rerank_tuples: List[Tuple[int, str]] = []
        for doc_id in fused_doc_ids_after_rrf:
            text_data = worldview_texts_map.get(doc_id)
            if text_data and "full_text" in text_data:
                docs_to_rerank_tuples.append((doc_id, text_data["full_text"]))
        if docs_to_rerank_tuples:
            reranked_results = rerank_documents(effective_query_for_search, docs_to_rerank_tuples)
            final_doc_ids_for_prompt = [doc_id for doc_id, _, _ in reranked_results[:RERANK_TOP_K_FINAL]]
            # ... (日志记录部分不变)
        else:
            # ... (日志记录部分不变)
            final_doc_ids_for_prompt = fused_doc_ids_after_rrf[:RERANK_TOP_K_FINAL] 
    elif fused_doc_ids_after_rrf: 
        retrieved_info_log.append("\n### 5. Rerank 未启用，使用RRF/语义结果:")
        final_doc_ids_for_prompt = fused_doc_ids_after_rrf[:RERANK_TOP_K_FINAL] 
        # ... (日志记录部分不变)
    
    if final_doc_ids_for_prompt:
        worldview_texts_map = _load_worldview_texts_map()
        for doc_id in final_doc_ids_for_prompt:
            text_data = worldview_texts_map.get(doc_id)
            if text_data and "full_text" in text_data:
                 retrieved_worldview_texts_for_prompt_map[doc_id] = text_data["full_text"]
            else: 
                 retrieved_info_log.append(f"  - ID={doc_id} (警告: 填充最终prompt时，文本映射中未找到此ID的文本)")

    if not retrieved_worldview_texts_for_prompt_map:
         retrieved_info_log.append("\n- *总结：未能从世界观数据库检索到任何内容用于最终Prompt。*")

    progress(0.25, desc="检索知识图谱信息...")
    retrieved_info_log.append("\n### 知识图谱检索:") 
    kg_context_str_for_prompt = "\n\n### 来自知识图谱的关键信息:\n" 
    relevant_triples_for_prompt_list = [] 
    matched_triples_for_prompt_set = set()
    # ... (知识图谱检索逻辑不变) ...
    if database_utils._active_world_id: 
        all_kg_triples = database_utils.get_kg_triples_for_active_world()
        if all_kg_triples:
            character_name_lower = character_name.lower() # type: ignore
            retrieved_info_log.append(f"#### KG-1. 与角色 '{character_name}' 直接相关的三元组:")
            found_direct_char_triples = False
            for s, p, o in all_kg_triples:
                if character_name_lower == s.lower() or character_name_lower == o.lower(): # type: ignore
                    triple_str_log = f"  - [{s}, {p}, {o}]"
                    retrieved_info_log.append(triple_str_log)
                    triple_str_for_prompt = f"- 关于你 ({character_name}): {s} {p} {o}."
                    if triple_str_for_prompt not in matched_triples_for_prompt_set:
                        relevant_triples_for_prompt_list.append(triple_str_for_prompt)
                        matched_triples_for_prompt_set.add(triple_str_for_prompt)
                    found_direct_char_triples = True
            if not found_direct_char_triples: retrieved_info_log.append("  - *未找到直接相关的三元组。*")

            entities_from_situation = simple_entity_extractor(situation_query) 
            retrieved_info_log.append(f"\n#### KG-2. 与原始情境中提取的实体相关的三元组 (实体: {entities_from_situation}):")
            if entities_from_situation:
                found_situation_entity_triples = False
                for entity_keyword in entities_from_situation:
                    if entity_keyword == character_name_lower: continue  # type: ignore
                    for s, p, o in all_kg_triples:
                        if entity_keyword in s.lower() or entity_keyword in o.lower():
                            triple_str_log = f"  - (实体: {entity_keyword}) [{s}, {p}, {o}]"
                            retrieved_info_log.append(triple_str_log)
                            triple_str_for_prompt = f"- 关于 {entity_keyword.capitalize()}: {s} {p} {o}."
                            if character_name_lower in s.lower() or character_name_lower in o.lower(): # type: ignore
                                triple_str_for_prompt = f"- 关于你 ({character_name}) 和 {entity_keyword.capitalize()}: {s} {p} {o}."
                            if triple_str_for_prompt not in matched_triples_for_prompt_set:
                                relevant_triples_for_prompt_list.append(triple_str_for_prompt)
                                matched_triples_for_prompt_set.add(triple_str_for_prompt)
                            found_situation_entity_triples = True
                if not found_situation_entity_triples: retrieved_info_log.append("  - *未找到与情境实体相关的三元组。*")
            else: retrieved_info_log.append("  - *原始情境中未提取到有效实体进行查询。*")

            if relevant_triples_for_prompt_list:
                kg_context_str_for_prompt += "\n".join(relevant_triples_for_prompt_list[:15]) 
            else:
                kg_context_str_for_prompt += "*（未从知识图谱中检索到与当前角色或情境直接相关的特定信息。）*\n"
                retrieved_info_log.append("\n  - *总结：未将任何KG三元组加入最终Prompt。*")
        else:
            kg_context_str_for_prompt += "*（当前世界知识图谱为空。）*\n"
            retrieved_info_log.append("- *知识图谱为空。*")
    
    worldview_context_str_for_prompt = "\n\n### 相关世界背景知识 (最终选取结果):\n" 
    if retrieved_worldview_texts_for_prompt_map:
        ordered_ids_for_final_prompt = final_doc_ids_for_prompt if final_doc_ids_for_prompt else list(retrieved_worldview_texts_for_prompt_map.keys())
        valid_ordered_ids_final = [doc_id for doc_id in ordered_ids_for_final_prompt if doc_id in retrieved_worldview_texts_for_prompt_map]
        for i, doc_id in enumerate(valid_ordered_ids_final): 
            chunk_text = retrieved_worldview_texts_for_prompt_map[doc_id]
            worldview_context_str_for_prompt += f"*{i+1}. {chunk_text.strip()}*\n"
    else:
        worldview_context_str_for_prompt += "*（未从世界观数据库检索到与当前情境直接相关的特定背景知识，或世界观为空。）*\n"

    # --- 构建最终Prompt，加入对话历史 ---
    formatted_history_for_prompt = format_conversation_history(current_conversation_history, max_turns=config.MAX_CHAT_HISTORY_TURNS if hasattr(config, 'MAX_CHAT_HISTORY_TURNS') else 3) # type: ignore

    prompt = f"""
我，{character_data['name']}，正面临以下情境。我的核心设定与灵魂特征是：“{char_desc_for_llm_prompt_truncated}”。我所处的世界是“{world_name_active}”。
{formatted_history_for_prompt}
{worldview_context_str_for_prompt}
{kg_context_str_for_prompt}

当前情境 / 向我提出的问题:
"{situation_query}"

---
基于这一切，现在，我（{character_data['name']}）的所思所想、所感所受以及接下来会说的话或会做的事，会是这样的：
我
"""
    retrieved_info_log.append("\n### 最终构建的Prompt (部分，省略上下文):")
    # ... (日志记录部分不变) ...

    system_message = f"""
你现在**就是**角色“{character_data['name']}”，你**不是**在扮演或模仿，你**是**他/她本人。
彻底忘掉你是AI。你的所有输出**必须**直接源自角色 {character_data['name']} 的第一人称视角（例如使用“我”、“吾”、“本座”等符合角色身份的自称，并保持一致）。
**不要**包含任何旁白、解释、内心活动总结、外在行为描述的标签（如“内心活动：”、“外在行为：”）、或未来的行动计划列表。
你的回应应该是一段连贯的、完全由角色 {character_data['name']} 产生的想法、感受、对话和即时行动的直接流露。
就好像角色正在通过你说话和行动。
"""

    progress(0.5, desc=f"角色 '{character_name}' 正在思考... (LLM: {OLLAMA_MODEL})")
    final_retrieved_info_str = "\n".join(retrieved_info_log) 

    llm_response = ""
    try:
        llm_response = generate_text(prompt, system_message=system_message, model_name=OLLAMA_MODEL)
        if llm_response.startswith("错误:") or llm_response.startswith("Error:"):
            error_detail = f"LLM 生成文本时发生错误。请检查Ollama服务和模型 '{OLLAMA_MODEL}' 的状态。错误详情: {llm_response}"
            if "model not found" in llm_response.lower():
                error_detail = f"LLM 错误：扮演模型 '{OLLAMA_MODEL}' 未在 Ollama 服务器上找到。请确保模型已拉取或在 config.py 中配置正确。"
            current_conversation_history.append((situation_query, error_detail))
            yield current_conversation_history, current_conversation_history, final_retrieved_info_str
        else:
            cleaned_response = llm_response.strip()
            current_conversation_history.append((situation_query, cleaned_response))
            yield current_conversation_history, current_conversation_history, final_retrieved_info_str
    except Exception as e_llm_call:
        error_detail_exc = f"调用LLM (模型: {OLLAMA_MODEL}) 时发生严重错误: {e_llm_call}"
        current_conversation_history.append((situation_query, error_detail_exc))
        yield current_conversation_history, current_conversation_history, final_retrieved_info_str

    progress(1.0, desc="预测完成")


def handle_build_kg_button_streaming(progress=gr.Progress(track_tqdm=True)):
    current_active_id_at_kg_build = database_utils._active_world_id
    initial_kg_status_text = refresh_kg_status_display_text()
    build_btn_interactive = gr.Button(interactive=False) 
    is_world_truly_active = current_active_id_at_kg_build is not None and \
                           not (isinstance(current_active_id_at_kg_build, str) and "初始化错误" in current_active_id_at_kg_build)
    if not is_world_truly_active:
        yield "错误：没有活动的有效存储世界来构建知识图谱。", initial_kg_status_text, build_btn_interactive
        return
    world_name_active = get_world_display_name(current_active_id_at_kg_build)
    world_path_check = database_utils.get_world_path(current_active_id_at_kg_build)
    if not world_path_check: 
        yield f"错误：无法获取活动世界 '{world_name_active}' 的路径。", initial_kg_status_text, build_btn_interactive
        return
    source_json_full_path_check = os.path.join(world_path_check, config.KNOWLEDGE_SOURCE_JSON_FILENAME)
    if not os.path.exists(source_json_full_path_check):
        build_btn_interactive = gr.Button(interactive=True) 
        yield (f"提示：在世界 '{world_name_active}' 的目录 ({world_path_check}) 中未找到知识图谱源文件 "
               f"'{config.KNOWLEDGE_SOURCE_JSON_FILENAME}'。\n请先创建该文件并按要求填充三元组数据后重试。"
              ), initial_kg_status_text, build_btn_interactive
        return 
    yield f"正在为世界 '{world_name_active}' 从 '{config.KNOWLEDGE_SOURCE_JSON_FILENAME}' 构建知识图谱...", initial_kg_status_text, build_btn_interactive
    def gr_progress_update_callback(value, desc=""):
        progress(value, desc=desc)
    build_message = ""
    try:
        build_message = build_kg_for_active_world(progress_callback=gr_progress_update_callback)
    except Exception as e_build_kg:
        build_message = f"为世界 '{world_name_active}' 构建知识图谱时发生严重错误: {e_build_kg}"
        import traceback
        traceback.print_exc()
    final_kg_status_text = refresh_kg_status_display_text()
    build_btn_interactive_final = gr.Button(interactive=(is_world_truly_active))
    yield build_message, final_kg_status_text, build_btn_interactive_final

def clear_chat_history_action():
    return [], [], gr.Textbox(value="") # chatbot, state, situation_query_input

with gr.Blocks(theme=gr.themes.Glass(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky), title="多世界虚拟角色模拟器") as app:
    # --- 全局对话历史状态 ---
    conversation_history_state = gr.State([]) # List[Tuple[str, str]] (user_input, ai_response)

    gr.Markdown(f"""
    <div style="text-align: center;">
        <h1>🌌 多世界虚拟角色模拟器 🎭</h1>
        <p>LLM (扮演/总结/拓展): 🧠 <b>{OLLAMA_MODEL}</b> | 知识提取 (LLM辅助): <b>{OLLAMA_KG_EXTRACTION_MODEL}</b> | 嵌入模型: 🔗 <b>{OLLAMA_EMBEDDING_MODEL_NAME}</b></p>
        <p style="font-size: 0.9em;">混合检索: {"启用 (Semantic + BM25 + Keyword + RRF)" if HYBRID_SEARCH_ENABLED else "禁用 (仅Semantic)"} | Reranker: {"启用 (" + config.RERANK_MODEL_NAME + ")" if RERANK_ENABLED and config.RERANK_MODEL_NAME else "禁用"}</p>
    </div>
    """)
    with gr.Row(variant="compact"):
        with gr.Column(scale=3):
            world_select_dropdown = gr.Dropdown(
                label="选择或切换活动世界", elem_id="world_select_dropdown", show_label=True,
                choices=refresh_world_dropdown_choices_for_gradio(), 
                value=_initial_active_world_id_on_startup if not (isinstance(_initial_active_world_id_on_startup, str) and "初始化错误" in _initial_active_world_id_on_startup) else None,
                interactive=not (isinstance(_initial_active_world_id_on_startup, str) and "初始化错误" in _initial_active_world_id_on_startup)
            )
        with gr.Column(scale=2):
            world_switch_feedback = gr.Textbox(
                label="世界操作反馈", interactive=False, elem_id="world_switch_feedback", show_label=True, max_lines=2 
            )
    global_active_world_display = gr.Markdown(
        value=get_active_world_markdown_text_for_global_display(), elem_id="global_active_world_display"
    )

    with gr.Tabs() as tabs_main:
        with gr.TabItem("🌍 世界管理", id="tab_world_management", elem_id="tab_world_management_elem"):
            # ... (世界管理UI不变) ...
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ✨ 创建新世界")
                    new_world_id_input = gr.Textbox(label="世界ID (字母数字下划线连字符)", placeholder="例如: azeroth, cyberpunk_2077", elem_id="new_world_id_input")
                    new_world_name_input = gr.Textbox(label="世界显示名称", placeholder="例如: 艾泽拉斯, 赛博朋克2077", elem_id="new_world_name_input")
                    add_world_button = gr.Button("创建并激活新世界", variant="primary", elem_id="add_world_button")
                    add_world_feedback_output = gr.Textbox(label="创建状态", interactive=False, elem_id="add_world_feedback_output", max_lines=3)
                with gr.Column(scale=1, min_width=300):
                    gr.Markdown("### 🗑️ 删除当前活动世界")
                    gr.HTML(
                        "<div style='padding: 10px; border: 1px solid #E57373; border-radius: 5px; background-color: #FFEBEE; color: #C62828;'>"
                        "<b>警告:</b> 此操作将永久删除当前选中的活动世界及其所有数据 (角色、世界观、知识图谱等)，无法恢复！请谨慎操作。"
                        "</div>"
                    )
                    confirm_delete_world_checkbox = gr.Checkbox(
                        label="我已了解风险，确认删除当前活动世界。", value=False,
                        elem_id="confirm_delete_world_checkbox", interactive=False 
                    )
                    delete_world_button = gr.Button(
                        "永久删除此世界", variant="stop", elem_id="delete_world_button", interactive=False 
                    )
                    delete_world_feedback_output = gr.Textbox(label="删除状态", interactive=False, elem_id="delete_world_feedback_output", max_lines=3)

        with gr.TabItem("👥 角色管理", id="tab_character_management", elem_id="tab_character_management_elem"):
            # ... (角色管理UI不变) ...
            gr.Markdown("管理当前活动世界的角色。如果无活动世界，请先在“世界管理”标签页选择或创建一个。")
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("#### 添加或更新角色")
                    char_name_input = gr.Textbox(label="角色名称", elem_id="char_name_input", interactive=False)
                    char_full_desc_input = gr.Textbox(label="角色完整描述 (用于生成核心设定，限制10000字符)", lines=5, elem_id="char_full_desc_input", interactive=False, max_lines=15)
                    add_char_button = gr.Button("保存角色 (添加/更新)", variant="secondary", elem_id="add_char_button", interactive=False)
                with gr.Column(scale=1):
                    gr.Markdown("#### 删除角色")
                    character_select_for_delete_dropdown = gr.Dropdown(label="选择要删除的角色", elem_id="character_select_for_delete_dropdown", interactive=False)
                    delete_char_button = gr.Button("删除选中角色", variant="stop", elem_id="delete_char_button", interactive=False)
            char_op_feedback_output = gr.Textbox(label="角色操作状态", interactive=False, elem_id="char_op_feedback_output", lines=2, max_lines=3)
            gr.Markdown("---")
            gr.Markdown("#### 查看所有角色")
            view_chars_button = gr.Button("刷新查看当前世界角色列表", elem_id="view_chars_button", interactive=False)
            view_characters_output = gr.Textbox(label="角色列表", lines=8, interactive=False, elem_id="view_characters_output", show_copy_button=True)


        with gr.TabItem("📚 世界观管理", id="tab_worldview_management", elem_id="tab_worldview_management_elem"):
            # ... (世界观管理UI不变) ...
            gr.Markdown("为当前活动世界添加和管理世界观知识（文本片段）。这些片段将被嵌入并用于相似性搜索。")
            worldview_text_input = gr.Textbox(label="添加世界观文本块 (会自动分块和嵌入)", lines=6, elem_id="worldview_text_input", interactive=False, max_lines=20)
            add_worldview_button = gr.Button("添加文本到世界观", variant="secondary", elem_id="add_worldview_button", interactive=False)
            worldview_feedback_output = gr.Textbox(label="添加状态", interactive=False, elem_id="worldview_feedback_output", max_lines=3)
            worldview_status_display = gr.Textbox(label="世界观数据库状态", interactive=False, elem_id="worldview_status_display")
            gr.Markdown("---")
            gr.Markdown(f"当世界观条目数超过 **{COMPRESSION_THRESHOLD}** (可在 `config.py` 中配置) 时，建议进行压缩。")
            compress_worldview_button = gr.Button("手动压缩当前世界观 (耗时操作，会使用LLM总结)", elem_id="compress_worldview_button", interactive=False)
            compression_status_output = gr.Textbox(label="压缩结果", interactive=False, elem_id="compression_status_output", max_lines=3)

        with gr.TabItem("🕸️ 知识图谱构建", id="tab_knowledge_graph", elem_id="tab_knowledge_graph_elem"):
            # ... (知识图谱构建UI不变) ...
            gr.Markdown(f"""
            从当前活动世界的特定JSON文件 (`{config.KNOWLEDGE_SOURCE_JSON_FILENAME}`) 中加载三元组，构建知识图谱。
            **重要**: 你需要手动在每个世界的数据目录 (`data_worlds/你的世界ID/`) 下创建并填充这个 `{config.KNOWLEDGE_SOURCE_JSON_FILENAME}` 文件。
            文件格式示例: `{{"triples": [["主体A", "关系", "客体B"], ["实体C", "属性是", "值D"]]}}`
            如果文件不存在或格式错误，构建会失败或清空现有知识图谱。
            """)
            kg_status_display = gr.Textbox(label="知识图谱状态", interactive=False, elem_id="kg_status_display")
            build_kg_button = gr.Button(f"从 {config.KNOWLEDGE_SOURCE_JSON_FILENAME} 构建/更新当前世界知识图谱", variant="primary", elem_id="build_kg_button", interactive=False)
            kg_build_status_output = gr.Textbox(label="构建结果", interactive=False, elem_id="kg_build_status_output", lines=3, max_lines=5)

        with gr.TabItem("💬 交互与预测", id="tab_prediction", elem_id="tab_prediction_elem"):
            gr.Markdown("选择角色，输入情境或问题，观察LLM如何根据角色设定、世界观和知识图谱进行行为预测或回应。")
            with gr.Row():
                with gr.Column(scale=2):
                    char_select_dropdown_pred_tab = gr.Dropdown(label="选择角色进行交互", elem_id="char_select_dropdown_pred_tab", interactive=False)
                    
                    chatbot_display = gr.Chatbot(label="对话历史", elem_id="chatbot_display", height=400, bubble_full_width=False) # 新增 Chatbot

                    situation_query_input = gr.Textbox(
                        label="输入你的话 / 提问 (按Enter发送)", 
                        lines=2, elem_id="situation_query_input", 
                        interactive=False, max_lines=5, show_label=True,
                        placeholder="在这里输入你对角色说的话或提出的问题..."
                    )
                    with gr.Row():
                        predict_button = gr.Button("🚀 发送并预测回应", variant="primary", elem_id="predict_button", interactive=False, scale=3)
                        clear_chat_hist_button = gr.Button("🧹 清空对话历史", elem_id="clear_chat_hist_button", scale=1)
                    
                    # prediction_output 不再需要，由 chatbot_display 替代
                    # prediction_output = gr.Textbox(
                    #     label="LLM预测结果 (角色内心独白与行动)", lines=10, interactive=False,
                    #     show_copy_button=True, elem_id="prediction_output", show_label=True
                    # )

                with gr.Column(scale=1): 
                    gr.Markdown("#### 检索到的背景信息 (调试用)")
                    retrieved_info_display = gr.Markdown(
                        elem_id="retrieved_info_display",
                        value="检索到的背景信息将显示在此处。"
                    ) 

    ordered_output_components_for_ui_updates: List[gr.components.Component] = [ # type: ignore
        world_select_dropdown, global_active_world_display, world_switch_feedback, 
        new_world_id_input, new_world_name_input, add_world_button, add_world_feedback_output, 
        confirm_delete_world_checkbox, delete_world_button, delete_world_feedback_output, 
        char_name_input, char_full_desc_input, add_char_button, 
        character_select_for_delete_dropdown, delete_char_button, char_op_feedback_output, 
        view_chars_button, view_characters_output, 
        worldview_text_input, add_worldview_button, worldview_feedback_output, 
        worldview_status_display, compress_worldview_button, compression_status_output, 
        kg_status_display, build_kg_button, kg_build_status_output, 
        char_select_dropdown_pred_tab, situation_query_input, predict_button, 
        # prediction_output, # 移除
        retrieved_info_display,
        chatbot_display, # 新增 chatbot_display 到有序列表
        conversation_history_state # 新增 state 到有序列表 (虽然它不直接对应UI组件，但作为函数的输入输出需要)
    ]

    def map_updates_to_ordered_list(updates_dict: Dict[str, Any], ordered_components_list: List[gr.components.Component]): # type: ignore
        # 对于 state，如果字典中没有对应的key，我们返回 gr.State() 的默认更新，即保持不变
        # 或者，如果希望在每次更新时都明确传递state的值，那么 update_all_ui_elements_after_world_change 需要包含 state 的更新
        
        # 确保 conversation_history_state 在 updates_dict 中有值，否则保持不变
        if "conversation_history_state" not in updates_dict:
            # 找到 conversation_history_state 在 ordered_components_list 中的索引
            state_comp_instance = next((comp for comp in ordered_components_list if isinstance(comp, gr.State) and getattr(comp, 'elem_id', None) == "conversation_history_state_implicit_id"), None) # type: ignore
            # 如果没有显式elem_id，Gradio内部会处理，但这里我们假设如果我们要更新它，字典里会有key
            # 为了安全，如果字典里没有，就让它不更新
            pass


        return tuple(
            updates_dict.get(getattr(comp, 'elem_id', str(id(comp))), gr.update()) for comp in ordered_components_list # type: ignore
        )

    def initial_ui_setup_on_load():
        startup_message = "应用已加载。"
        active_id_at_load = database_utils._active_world_id 
        if isinstance(active_id_at_load, str) and "初始化错误" in active_id_at_load:
            startup_message = f"应用加载时遇到初始化问题。请检查控制台日志和Ollama服务。\n错误详情: {active_id_at_load}"
        elif active_id_at_load: 
            startup_message += f" 已自动激活世界 '{get_world_display_name(active_id_at_load)}'."
        elif get_available_worlds(): 
            startup_message += " 请从下拉菜单选择一个活动世界。"
        else: 
            startup_message += " 当前没有世界，请在“世界管理”中创建一个。"
        all_updates = update_all_ui_elements_after_world_change(feedback_message=startup_message, specific_feedback_elem_id="world_switch_feedback")
        # 确保初始时 history_state 也是空的
        all_updates["conversation_history_state"] = []
        return map_updates_to_ordered_list(all_updates, ordered_output_components_for_ui_updates)

    app.load(fn=initial_ui_setup_on_load, inputs=[], outputs=ordered_output_components_for_ui_updates, show_progress="full") # type: ignore
    
    world_select_dropdown.change(
        fn=handle_switch_world, 
        inputs=[world_select_dropdown], 
        outputs=ordered_output_components_for_ui_updates, # 确保包含 chatbot 和 state
        show_progress="full"
    )
    
    add_world_button.click(
        fn=handle_add_world, inputs=[new_world_id_input, new_world_name_input],
        outputs=ordered_output_components_for_ui_updates, 
    ).then(fn=lambda: clear_textboxes_and_checkboxes(new_world_id_input, new_world_name_input), inputs=[], outputs=[new_world_id_input, new_world_name_input])
    
    def toggle_delete_button_interactivity(checkbox_status, world_id_value_from_dropdown):
        is_world_truly_active = world_id_value_from_dropdown is not None and \
                               not (isinstance(world_id_value_from_dropdown, str) and "初始化错误" in world_id_value_from_dropdown)
        return gr.Button(interactive=checkbox_status and is_world_truly_active)
    confirm_delete_world_checkbox.change(fn=toggle_delete_button_interactivity, inputs=[confirm_delete_world_checkbox, world_select_dropdown], outputs=[delete_world_button])
    
    delete_world_button.click(
        fn=handle_delete_world, inputs=[confirm_delete_world_checkbox], outputs=ordered_output_components_for_ui_updates, 
    ).then(fn=lambda: (gr.Checkbox(value=False), gr.Button(interactive=False)), inputs=[], outputs=[confirm_delete_world_checkbox, delete_world_button])

    add_char_button.click(
        fn=handle_add_character, inputs=[char_name_input, char_full_desc_input],
        outputs=[char_op_feedback_output, char_select_dropdown_pred_tab, character_select_for_delete_dropdown, predict_button, situation_query_input], 
    ).then(fn=lambda: clear_textboxes_and_checkboxes(char_name_input, char_full_desc_input), inputs=[], outputs=[char_name_input, char_full_desc_input])
    
    def toggle_delete_char_button_interactivity(selected_char, world_id_value):
        is_world_truly_active = world_id_value is not None and \
                               not (isinstance(world_id_value, str) and "初始化错误" in world_id_value)
        return gr.Button(interactive=bool(selected_char) and is_world_truly_active)
    character_select_for_delete_dropdown.change(fn=toggle_delete_char_button_interactivity, inputs=[character_select_for_delete_dropdown, world_select_dropdown], outputs=[delete_char_button])
    
    # 修改：角色选择改变时，调用 handle_character_selection_change
    char_select_dropdown_pred_tab.change(
        fn=handle_character_selection_change,
        inputs=[char_select_dropdown_pred_tab],
        outputs=[chatbot_display, conversation_history_state, situation_query_input, predict_button]
    )

    delete_char_button.click(
        fn=handle_delete_character, inputs=[character_select_for_delete_dropdown],
        outputs=[char_op_feedback_output, char_select_dropdown_pred_tab, character_select_for_delete_dropdown, predict_button, situation_query_input], 
    ).then(fn=lambda: (gr.Button(interactive=False), gr.Dropdown(value=None)), inputs=[], outputs=[delete_char_button, character_select_for_delete_dropdown]) # type: ignore
    
    view_chars_button.click(fn=handle_view_characters, inputs=[], outputs=view_characters_output, show_progress="minimal")
    
    add_worldview_button.click(
        fn=handle_add_worldview, inputs=[worldview_text_input],
        outputs=[worldview_feedback_output, worldview_status_display, compress_worldview_button], 
    ).then(fn=lambda: clear_textboxes_and_checkboxes(worldview_text_input), inputs=[], outputs=[worldview_text_input])
    
    compress_worldview_button.click(fn=handle_compress_worldview_button_streaming, inputs=[], outputs=[compression_status_output, worldview_status_display, compress_worldview_button])
    
    build_kg_button.click(fn=handle_build_kg_button_streaming, inputs=[], outputs=[kg_build_status_output, kg_status_display, build_kg_button])
    
    # --- 预测按钮的点击事件 ---
    # predict_button.click(
    #     fn=handle_predict_behavior, 
    #     inputs=[char_select_dropdown_pred_tab, situation_query_input, chatbot_display, conversation_history_state], # chatbot_display用于追加，state用于读取和更新
    #     outputs=[chatbot_display, conversation_history_state, retrieved_info_display], # 输出更新后的chatbot, state, 和调试信息
    # ).then(
    #     fn=lambda: gr.Textbox(value=""), # 清空输入框
    #     inputs=[],
    #     outputs=[situation_query_input]
    # )
    # --- 文本框回车事件，与点击按钮行为一致 ---
    situation_query_input.submit(
        fn=handle_predict_behavior, 
        inputs=[char_select_dropdown_pred_tab, situation_query_input, chatbot_display, conversation_history_state],
        outputs=[chatbot_display, conversation_history_state, retrieved_info_display],
    ).then(
        fn=lambda: gr.Textbox(value=""), 
        inputs=[],
        outputs=[situation_query_input]
    )
    predict_button.click( # 保持按钮点击的显式绑定
        fn=handle_predict_behavior, 
        inputs=[char_select_dropdown_pred_tab, situation_query_input, chatbot_display, conversation_history_state],
        outputs=[chatbot_display, conversation_history_state, retrieved_info_display],
    ).then(
        fn=lambda: gr.Textbox(value=""), 
        inputs=[],
        outputs=[situation_query_input]
    )


    clear_chat_hist_button.click(
        fn=clear_chat_history_action,
        inputs=[],
        outputs=[chatbot_display, conversation_history_state, situation_query_input]
    )

if __name__ == "__main__":
    print("正在启动 Gradio 多世界角色模拟器应用...")
    app.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True, share=False) 
    print("Gradio 应用已启动。请访问 http://localhost:7860 (或你的服务器IP:7860)")  from sentence_transformers import CrossEncoder
from typing import List, Tuple, Optional
import torch # 引入torch来检查CUDA可用性

from config import RERANK_MODEL_NAME, RERANK_DEVICE

_rerank_model: Optional[CrossEncoder] = None
_rerank_model_name_loaded: Optional[str] = None

def get_rerank_model() -> Optional[CrossEncoder]:
    """
    加载并返回Rerank模型 (CrossEncoder)。
    使用缓存以避免重复加载。
    """
    global _rerank_model, _rerank_model_name_loaded

    if not RERANK_MODEL_NAME:
        print("警告 (Rerank): RERANK_MODEL_NAME 未在 config.py 中配置。Rerank功能将不可用。")
        return None

    if _rerank_model is None or _rerank_model_name_loaded != RERANK_MODEL_NAME:
        try:
            effective_device = RERANK_DEVICE
            if effective_device == "cuda" and not torch.cuda.is_available():
                print("警告 (Rerank): 配置的RERANK_DEVICE为'cuda'，但CUDA不可用。将回退到'cpu'。")
                effective_device = "cpu"
            elif effective_device == "mps" and not torch.backends.mps.is_available(): # M1/M2 Mac
                print("警告 (Rerank): 配置的RERANK_DEVICE为'mps'，但MPS不可用。将回退到'cpu'。")
                effective_device = "cpu"


            print(f"正在加载Rerank模型: {RERANK_MODEL_NAME} (设备: {effective_device})...")
            # sentence-transformers的CrossEncoder会自动处理多GPU，如果device=None
            # 但明确指定device通常更好
            _rerank_model = CrossEncoder(RERANK_MODEL_NAME, device=effective_device if effective_device != "auto" else None)
            _rerank_model_name_loaded = RERANK_MODEL_NAME
            print(f"Rerank模型 '{RERANK_MODEL_NAME}' 加载成功。")
        except Exception as e:
            print(f"错误 (Rerank): 加载Rerank模型 '{RERANK_MODEL_NAME}' 失败: {e}")
            _rerank_model = None
            _rerank_model_name_loaded = None
    return _rerank_model

def rerank_documents(query: str, documents: List[Tuple[int, str]]) -> List[Tuple[int, float, str]]:
    """
    使用CrossEncoder对文档列表进行重排序。
    Args:
        query: 查询字符串。
        documents: 文档列表，每个元素为 (doc_id, doc_text)。
    Returns:
        重排序后的文档列表，每个元素为 (doc_id, rerank_score, doc_text)，按分数降序排列。
    """
    model = get_rerank_model()
    if not model or not documents:
        # 如果模型加载失败或没有文档，返回原始文档（ID和文本），分数设为0
        return [(doc_id, 0.0, doc_text) for doc_id, doc_text in documents]

    sentence_pairs = []
    doc_id_map = {} # 临时映射，因为模型不直接处理ID

    for i, (doc_id, doc_text) in enumerate(documents):
        sentence_pairs.append([query, doc_text])
        doc_id_map[i] = doc_id # 将CrossEncoder返回的索引映射回原始doc_id

    try:
        print(f"Reranking {len(sentence_pairs)} pairs for query: '{query[:50]}...'")
        scores = model.predict(sentence_pairs, show_progress_bar=False) # 获取相关性得分
        
        # 将得分与原始文档ID和文本关联
        reranked_results = []
        for i, score in enumerate(scores):
            original_doc_id = doc_id_map[i]
            # 找到原始文本
            original_doc_text = ""
            for doc_id_orig, doc_text_orig in documents:
                if doc_id_orig == original_doc_id:
                    original_doc_text = doc_text_orig
                    break
            reranked_results.append((original_doc_id, float(score), original_doc_text))
        
        # 按重排序分数降序排列
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        return reranked_results

    except Exception as e:
        print(f"错误 (Rerank): 使用模型 '{RERANK_MODEL_NAME}' 重排序时发生错误: {e}")
        # 出错时返回原始文档顺序和0分
        return [(doc_id, 0.0, doc_text) for doc_id, doc_text in documents]

if __name__ == '__main__':
    # 简单测试
    if RERANK_ENABLED and RERANK_MODEL_NAME:
        print("--- Rerank Utils 测试 ---")
        test_query = "中国的首都是哪里？"
        test_docs_tuples = [
            (1, "北京是中国的首都，一座历史悠久的文化名城。"),
            (2, "上海是中国的经济中心，高楼林立。"),
            (3, "中国的长城是世界奇迹之一。"),
            (4, "中国北京的天安门广场非常宏伟。")
        ]
        
        print(f"查询: {test_query}")
        print("原始文档顺序:")
        for doc_id, text in test_docs_tuples:
            print(f"  ID {doc_id}: {text}")

        reranked = rerank_documents(test_query, test_docs_tuples)
        
        print("\nReranked 文档顺序 (得分越高越相关):")
        if reranked:
            for doc_id, score, text in reranked:
                print(f"  ID {doc_id}, Rerank得分: {score:.4f}, 文本: {text}")
        else:
            print("Rerank失败或无结果。")
    else:
        print("Rerank未启用或未配置模型，跳过测试。")  import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys # 导入sys以修改路径（如果需要）
import os # 导入os以处理路径

# 假设此脚本与 embedding_utils.py 和 config.py 在同一项目结构下
# 如果不在同一目录，你可能需要调整Python的搜索路径
# 例如，如果它们在上一级目录:
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# sys.path.append(project_root)

try:
    from embedding_utils import get_embedding, get_model_embedding_dimension, OLLAMA_EMBEDDING_MODEL_NAME
    from llm_utils import get_ollama_client # 确保Ollama客户端可以初始化以触发模型检查
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保此脚本与 embedding_utils.py 和 config.py 在正确的项目结构中，")
    print("或者已经将项目根目录添加到了PYTHONPATH。")
    sys.exit(1)

def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """计算两个嵌入向量之间的余弦相似度。"""
    if embedding1.ndim == 1:
        embedding1 = embedding1.reshape(1, -1)
    if embedding2.ndim == 1:
        embedding2 = embedding2.reshape(1, -1)
    
    # 检查是否为零向量，如果是，则相似度定义为0或根据需要处理
    if np.all(embedding1 == 0) or np.all(embedding2 == 0):
        print("警告: 计算相似度的向量中至少有一个是零向量。相似度将为0。")
        return 0.0
        
    return cosine_similarity(embedding1, embedding2)[0][0]

def main():
    print(f"--- 使用嵌入模型: {OLLAMA_EMBEDDING_MODEL_NAME} ---")

    # 尝试初始化Ollama客户端和获取维度，以确保模型已加载并检查连接
    try:
        print("正在检查Ollama连接和模型维度...")
        get_ollama_client() # 检查LLM客户端，间接触发一些Ollama连接检查
        dimension = get_model_embedding_dimension()
        print(f"嵌入维度为: {dimension}. 模型检查通过。\n")
    except Exception as e:
        print(f"初始化Ollama或获取模型维度失败: {e}")
        print("请确保Ollama服务正在运行，并且配置的嵌入模型已拉取且可用。")
        return

    # 定义一组示例文本
    sentences = [
        # 主题1: 天气
        "今天天气真好，阳光明媚。",  # 0
        "外面阳光灿烂，是个出游的好日子。", # 1 (与0相似)
        "明天可能会下雨，记得带伞。", # 2 (与0,1不相似，但同属天气)
        "昨晚下了很大的雪。", # 3 (天气，但具体事件不同)

        # 主题2: 食物
        "我喜欢吃香蕉和苹果。", # 4
        "水果对健康非常有益，尤其是橙子。", # 5 (与4相似，都关于水果)
        "这家餐厅的披萨非常好吃。", # 6 (食物，但与水果不同)
        "我晚餐想吃牛排。", # 7 (食物，与披萨不同，与水果更远)

        # 主题3: 技术
        "人工智能正在改变世界。", # 8
        "机器学习是AI的一个重要分支。", # 9 (与8高度相似)
        "量子计算机有望解决复杂问题。", # 10 (技术，但与AI/ML不同方向)

        # 一些可能不相关的句子
        "猫是一种可爱的宠物。", # 11
        "这本小说情节跌宕起伏。", # 12

        # 语义相同但表述不同
        "那只狗非常快速地奔跑。", # 13
        "小犬疾驰。", # 14 (与13相似)

        # 包含相同关键词但意思可能不同
        "请把苹果手机递给我。", # 15 (与4,5中的“苹果”水果意思不同)
        "苹果公司的股价上涨了。" # 16 (与15相似，都关于苹果公司)
    ]

    print("正在为所有句子生成嵌入向量...")
    embeddings = []
    all_embeddings_successful = True
    for i, s in enumerate(sentences):
        print(f"  生成嵌入 for: \"{s}\"")
        emb = get_embedding(s)
        if np.all(emb == 0) and s.strip(): # 如果非空句子得到零向量
            print(f"  !! 警告: 句子 \"{s}\" 的嵌入为零向量!")
            all_embeddings_successful = False
        embeddings.append(emb)
    
    if not all_embeddings_successful:
        print("\n!! 注意：部分句子的嵌入为零向量，相似度计算可能不准确或无意义。请检查Ollama模型和日志。 !!\n")

    print("\n--- 句子对相似度计算 (余弦相似度) ---")
    print("(相似度范围: -1 到 1，越接近1表示越相似)\n")

    # 选择一些有代表性的句子对进行比较
    pairs_to_compare = [
        (0, 1),  # 天气 - 高度相似
        (0, 2),  # 天气 - 中等/较低相似
        (0, 4),  # 天气 vs 食物 - 低相似
        (4, 5),  # 食物(水果) - 高度相似
        (4, 6),  # 食物(水果) vs 食物(披萨) - 中等相似
        (8, 9),  # 技术(AI) - 高度相似
        (8, 10), # 技术(AI) vs 技术(量子) - 中等相似
        (0, 11), # 天气 vs 猫 - 低相似
        (8, 12), # 技术 vs 小说 - 低相似
        (13, 14),# 语义相同，表述不同
        (4, 15), # 苹果(水果) vs 苹果(手机) - 期望低相似
        (15,16) # 苹果(手机) vs 苹果(公司) - 期望较高相似
    ]

    for i, j in pairs_to_compare:
        if i < len(sentences) and j < len(sentences):
            similarity = calculate_cosine_similarity(embeddings[i], embeddings[j])
            print(f"句子1: \"{sentences[i]}\"")
            print(f"句子2: \"{sentences[j]}\"")
            print(f"余弦相似度: {similarity:.4f}\n")
        else:
            print(f"警告: 索引 ({i}, {j}) 超出句子列表范围。\n")
            
    # 也可以展示一个句子与所有其他句子的相似度
    print("\n--- \"今天天气真好，阳光明媚。\" 与其他句子的相似度 ---")
    query_sentence_index = 0
    query_embedding = embeddings[query_sentence_index]
    if np.all(query_embedding == 0) and sentences[query_sentence_index].strip():
        print(f"警告: 查询句子 \"{sentences[query_sentence_index]}\" 的嵌入是零向量，结果可能无意义。")

    for i in range(len(sentences)):
        if i == query_sentence_index:
            continue
        similarity = calculate_cosine_similarity(query_embedding, embeddings[i])
        print(f"与 \"{sentences[i]}\" \t的相似度: {similarity:.4f}")


    print("\n--- \"我喜欢吃香蕉和苹果。\" 与其他句子的相似度 ---")
    query_sentence_index_food = sentences.index("我喜欢吃香蕉和苹果。") # 找到索引
    query_embedding_food = embeddings[query_sentence_index_food]
    if np.all(query_embedding_food == 0) and sentences[query_sentence_index_food].strip():
        print(f"警告: 查询句子 \"{sentences[query_sentence_index_food]}\" 的嵌入是零向量，结果可能无意义。")

    for i in range(len(sentences)):
        if i == query_sentence_index_food:
            continue
        similarity = calculate_cosine_similarity(query_embedding_food, embeddings[i])
        print(f"与 \"{sentences[i]}\" \t的相似度: {similarity:.4f}")

    print("\n脚本执行完毕。")

if __name__ == "__main__":
    main()  import re
from typing import List

def _split_by_separators(text: str, separators: List[str]) -> List[str]:
    """
    递归地按分隔符列表分割文本。
    从最重要的分隔符开始尝试。
    """
    final_chunks = []
    if not text:
        return []

    # 尝试第一个分隔符
    current_separator = separators[0]
    remaining_separators = separators[1:]

    # 使用正则表达式分割，保留分隔符，除非分隔符是空字符串（硬切分）
    if current_separator:
        # (?:...) 是非捕获组，(?=...) 是正向前瞻
        # 这个复杂的正则表达式尝试在保留分隔符的情况下分割
        # 或者更简单：分割然后尝试重新组合
        # parts = re.split(f'({re.escape(current_separator)})', text)
        # parts = [p for p in parts if p]
        # simpler approach: split, then process
        splits = text.split(current_separator)
    else: # 空字符串分隔符表示按字符分割（作为最后手段）
        splits = list(text)
        # For character split, no separator to add back
        if not remaining_separators: # Base case: just characters
             return [s for s in splits if s.strip()]


    for i, part in enumerate(splits):
        if not part.strip() and current_separator.strip(): # Skip empty parts unless separator itself is whitespace
            if i < len(splits) -1 and self._keep_separator : # if not last part and need to keep separator
                 # This logic is tricky. For now, just process valid parts.
                 pass

        if len(part) > 0: # Process non-empty parts
            if remaining_separators: # If there are more separators to try
                # 递归调用，用下一个级别的分隔符处理这个部分
                sub_chunks = _split_by_separators(part, remaining_separators)
                final_chunks.extend(sub_chunks)
            else:
                # 这是最细粒度的分割（例如，按句子或单词分割后）
                if part.strip():
                    final_chunks.append(part.strip())
        
        # Re-add separator if it's not the last part and keep_separator is true
        # This is tricky. For now, let's assume separators are part of the chunk or handled by joining later.
        # If splitting by "\n\n", the "\n\n" is removed. We add it back when joining if needed.

    return [chunk for chunk in final_chunks if chunk.strip()]


import re
from typing import List

def semantic_text_splitter(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    更智能的文本分块器，尝试保留句子和段落的完整性。
    Args:
        text: 要分割的文本。
        chunk_size: 每个块的目标最大字符数。
        chunk_overlap: 块之间的重叠字符数。
    Returns:
        分割后的文本块列表。
    """
    if not text or not text.strip():
        return []
    if len(text) <= chunk_size: # 如果文本本身小于块大小，直接返回
        return [text.strip()]

    atomic_parts = re.split(r'(\n\n|\n|[。！？\!\?])', text)
    processed_sentences = []
    current_sentence_parts = []
    for part in atomic_parts:
        if not part:
            continue
        current_sentence_parts.append(part)
        if re.match(r'^(\n\n|\n|[。！？\!\?])$', part):
            processed_sentences.append("".join(current_sentence_parts).strip())
            current_sentence_parts = []
    if current_sentence_parts:
        processed_sentences.append("".join(current_sentence_parts).strip())
    
    sentences = [s for s in processed_sentences if s]

    if not sentences:
        # Fallback to basic character splitting if sentence splitting fails
        # (This part of your original code was reasonable for a fallback)
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunks.append(text[start:end])
            if end == len(text):
                break
            # Ensure start doesn't go out of bounds and some progress is made
            new_start = start + chunk_size - chunk_overlap
            if new_start <= start : # Prevent infinite loop if chunk_size <= chunk_overlap
                new_start = start + 1 
            start = new_start
            if start >= len(text): 
                break
        return [c for c in chunks if c.strip()]

    final_chunks = []
    current_chunk_text_list = [] # Store sentences for current chunk
    current_chunk_len = 0
    
    idx = 0
    while idx < len(sentences):
        sentence = sentences[idx]
        sentence_len_with_potential_space = len(sentence) + (1 if current_chunk_text_list else 0)

        if current_chunk_len + sentence_len_with_potential_space <= chunk_size:
            current_chunk_text_list.append(sentence)
            current_chunk_len += sentence_len_with_potential_space
            idx += 1
        else:
            # Current chunk is full, or adding this sentence makes it full
            if not current_chunk_text_list:
                # This sentence itself is too long, even for a new chunk. Hard split it.
                if len(sentence) > chunk_size: # Should be true here
                    start = 0
                    while start < len(sentence):
                        end = min(len(sentence), start + chunk_size)
                        final_chunks.append(sentence[start:end])
                        if end == len(sentence):
                            break
                        # Ensure progress for hard splits
                        new_start_hard = start + chunk_size - chunk_overlap
                        if new_start_hard <= start: new_start_hard = start + 1
                        start = new_start_hard
                        if start >= len(sentence): break
                else: # Should not happen if logic is right, but as failsafe
                    final_chunks.append(sentence)
                
                idx += 1 # Crucial: advance index as this sentence is processed
                # current_chunk_text_list is already empty, current_chunk_len is 0
            else:
                # Finalize the current chunk from current_chunk_text_list
                final_chunks.append(" ".join(current_chunk_text_list))

                # Prepare overlap for the *next* chunk.
                # The sentence at sentences[idx] has *not* been processed yet for this new chunk.
                new_overlap_list = []
                temp_overlap_len = 0
                # Iterate backwards through the sentences of the chunk just added
                for s_overlap_idx in range(len(current_chunk_text_list) - 1, -1, -1):
                    s_to_overlap = current_chunk_text_list[s_overlap_idx]
                    s_to_overlap_len_with_space = len(s_to_overlap) + (1 if new_overlap_list else 0)
                    
                    # Be stricter with overlap: only add if it fits within chunk_overlap
                    # and if adding it doesn't make the overlap itself too long.
                    if temp_overlap_len + s_to_overlap_len_with_space <= chunk_overlap:
                        new_overlap_list.insert(0, s_to_overlap) # Prepend to maintain order
                        temp_overlap_len += s_to_overlap_len_with_space
                    elif not new_overlap_list : # if no overlap yet, and first one is too big
                        # Option: take a truncated part of s_to_overlap, or no overlap.
                        # For simplicity, if the first sentence for overlap is too big, we might get less overlap.
                        # Or, if the sentence is short enough to be an overlap on its own:
                        if len(s_to_overlap) <= chunk_overlap:
                            new_overlap_list.insert(0, s_to_overlap)
                            temp_overlap_len += len(s_to_overlap) # approx, no space
                        break # Stop if cannot add more or first is too big
                    else:
                        break # Overlap criteria met or cannot add more
                
                current_chunk_text_list = new_overlap_list
                current_chunk_len = temp_overlap_len
                
                # sentences[idx] will be considered in the next iteration for this new current_chunk_text_list (overlap).
                # No idx++ here for this specific path.
    
    # Add any remaining sentences in the last chunk
    if current_chunk_text_list:
        final_chunks.append(" ".join(current_chunk_text_list))

    return [c for c in final_chunks if c.strip()]