# config.py
import os

# --- Ollama 配置 ---
OLLAMA_BASE_URL = "http://localhost:11434" # 确保这是你的Ollama服务地址

# --- 模型配置 (根据你的可用模型调整) ---
# 主要用于角色扮演和复杂推理，效果优先
OLLAMA_MODEL = "deepseek-r1:14b"
# 备选: 如果 deepseek-r1:14b 太慢，可以尝试 "deepseek-r1:8b"

# 用于总结、简单提取、上下文剪裁、历史处理等，平衡效果与效率
# phi3 通常在这些任务上表现良好且高效
OLLAMA_SUMMARY_MODEL = "phi3:latest"
OLLAMA_CONTEXT_SNIPPING_MODEL = "phi3:latest"
OLLAMA_HISTORY_PROCESSING_MODEL = "phi3:latest"
OLLAMA_KG_REPHRASE_MODEL = "phi3:latest"

# 用于知识图谱三元组提取、角色特质提取、LLM评估等需要结构化输出的任务
# deepseek-coder 或 phi3 都可以，phi3 可能更通用一些
OLLAMA_KG_EXTRACTION_MODEL = "phi3:latest" # 或 "deepseek-coder:6.7b"
OLLAMA_TRAIT_EXTRACTION_MODEL = "phi3:latest" # 或 "deepseek-coder:6.7b"
OLLAMA_EVALUATION_MODEL = "phi3:latest"    # 或 "deepseek-coder:6.7b"

# 嵌入模型
OLLAMA_EMBEDDING_MODEL_NAME = "quentinz/bge-large-zh-v1.5"

# --- LLM 生成参数 ---
OLLAMA_DEFAULT_TEMPERATURE = 0.75 # 主要LLM的默认温度，用于角色扮演等创造性任务
OLLAMA_DEFAULT_TOP_P = 0.9       # 主要LLM的默认top_p

OLLAMA_SUMMARY_TEMPERATURE = 0.2 # 用于总结任务的温度，偏向事实性
OLLAMA_CONTEXT_SNIPPING_TEMPERATURE = 0.3 # 上下文剪裁温度
OLLAMA_HISTORY_PROCESSING_TEMPERATURE = 0.3 # 用于历史处理的温度
OLLAMA_KG_REPHRASE_TEMPERATURE = 0.4 # 用于KG改写的温度，需要一点创造性但仍基于事实
OLLAMA_KG_EXTRACTION_TEMPERATURE = 0.1 # KG提取，尽量精确
OLLAMA_TRAIT_EXTRACTION_TEMPERATURE = 0.1 # 特质提取，尽量精确
OLLAMA_EVALUATION_TEMPERATURE = 0.1 # LLM评估，尽量客观

# --- 数据存储配置 ---
DATA_DIR = "data_worlds"
WORLDS_METADATA_FILE = os.path.join(DATA_DIR, "worlds_metadata.json")

# --- 世界观数据库配置 ---
WORLD_CHARACTERS_DB_FILENAME = "characters.json"
WORLD_WORLDVIEW_TEXTS_FILENAME = "worldview_texts.json"
WORLD_WORLDVIEW_FAISS_INDEX_FILENAME = "worldview_index.faiss"
WORLD_WORLDVIEW_BM25_FILENAME = "worldview_bm25.pkl"
WORLD_KNOWLEDGE_GRAPH_FILENAME = "knowledge_graph.json"
KNOWLEDGE_SOURCE_JSON_FILENAME = "knowledge_source.json" # KG的JSON源文件

# --- 文本分块配置 ---
CHUNK_SIZE = 300 # 语义分块的目标块大小（字符数）
CHUNK_OVERLAP = 50 # 语义分块的重叠字符数

# --- 检索配置 (这些 TOP_K 值将来可考虑动态调整) ---
HYBRID_SEARCH_ENABLED = True # 是否启用混合检索
SEMANTIC_SEARCH_TOP_K_HYBRID = 5 # 混合检索中语义搜索召回数
BM25_SEARCH_TOP_K_HYBRID = 7 # 混合检索中BM25召回数
KEYWORD_SEARCH_TOP_K_HYBRID = 7 # 混合检索中关键词搜索召回数
HYBRID_RRF_K = 60 # RRF融合算法中的k值
HYBRID_FINAL_TOP_K = 10 # RRF融合后，送入Reranker或LLM评估的候选数量上限

# --- Rerank 配置 ---
RERANK_ENABLED = True # 是否启用Reranker
RERANK_MODEL_NAME = "BAAI/bge-reranker-base" # Rerank模型 (这个你需要通过pip安装sentence-transformers并确保能下载)
RERANK_DEVICE = "cpu"  # Rerank设备: "cpu", "cuda", "mps", "auto" (auto会尝试cuda, mps, then cpu)
RERANK_TOP_K_FINAL = 3 # Rerank后最终保留的结果数 (如果RERANK_ENABLED=False, 则此值也用于从HYBRID_FINAL_TOP_K中选取)

# --- 世界观压缩配置 ---
COMPRESSION_THRESHOLD = 100 # 世界观条目数超过此值时，UI提示建议压缩
COMPRESSION_TARGET_CLUSTERS = 20 # K-Means聚类目标簇数
COMPRESSION_FALLBACK_SUMMARY_LENGTH = 1000 # LLM总结失败时，备用摘要的最大长度

# --- 文本长度限制 (LLM输入) ---
MAX_CHAR_FULL_DESC_LEN = 10000 # 角色完整描述的最大长度 (数据库存储层面)
MAX_SUMMARY_INPUT_LEN_LLM = 3500 # LLM生成摘要时的最大输入文本长度 (phi3等模型上下文窗口通常4k-8k, 留些余地)
MAX_KG_TEXT_INPUT_LEN_LLM = 3000 # LLM提取KG三元组时的最大输入文本长度
MAX_EVAL_TEXT_INPUT_LEN_LLM = 1500 # LLM评估单个文本片段相关性时的最大输入长度
MAX_TRAIT_EXTRACTION_INPUT_LEN_LLM = 3000 # LLM提取角色结构化特质时的最大输入长度
MAX_CHAR_DESC_INPUT_FOR_CONTEXTUAL_SNIPPING_LLM = 2500 # LLM剪裁角色描述用于Prompt时的最大输入长度
MAX_HISTORY_TEXT_LENGTH_FOR_SUMMARY_INPUT = 3000 # LLM处理对话历史时的最大输入长度
MAX_KG_REPHRASE_INPUT_TRIPLES = 15 # LLM改写KG三元组时，一次处理的最大三元组数量 (减少以适应小模型)

# --- Prompt构建内容长度/数量限制 ---
TARGET_CHAR_DESC_OUTPUT_FOR_PROMPT_LLM = 400 # LLM剪裁角色描述后，期望输出用于Prompt的长度
MAX_WORLDVIEW_SNIPPETS_FOR_PROMPT = 3 # 构建最终Prompt时，最多包含的世界观片段数量
WORLDVIEW_LLM_EVAL_SCORE_THRESHOLD = 3 # LLM评估世界观片段的相关性得分阈值 (>= 此值才考虑用于Prompt)
MAX_KG_TRIPLES_FOR_PROMPT = 7 # 构建最终Prompt时，最多包含的（改写后的）KG信息片段数 (减少以适应小模型)

# --- 对话历史配置 ---
MAX_CHAT_HISTORY_TURNS = 5 # 最终Prompt中包含的最近原始对话历史轮数 (如果LLM历史处理未启用或失败)
MAX_HISTORY_TURNS_FOR_SUMMARY_THRESHOLD = 7 # 对话历史超过此轮数，则尝试LLM辅助处理（总结+相关轮次挑选）
TARGET_HISTORY_SUMMARY_LENGTH = 250 # LLM生成的对话历史摘要的目标长度 (phi3可能生成不了太长摘要)

# --- LLM评估配置 (除了文本相关性评估外，可以扩展) ---
MAX_RETRIEVED_TEXTS_FOR_LLM_EVALUATION = 5 # Rerank/融合后，最多送多少条文本给LLM进行相关性评估
# LLM_EVALUATION_BATCH_SIZE = 1 # (当前逐条评估，预留)

# 确保数据目录存在
os.makedirs(DATA_DIR, exist_ok=True)
print(f"Config loaded. Data directory: '{os.path.abspath(DATA_DIR)}'")
if not OLLAMA_BASE_URL or not OLLAMA_MODEL or not OLLAMA_EMBEDDING_MODEL_NAME:
    print("关键Ollama配置 (OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL_NAME) 未完全设置，程序可能无法正常运行。")