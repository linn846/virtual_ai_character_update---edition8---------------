from sentence_transformers import CrossEncoder
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
        print("Rerank未启用或未配置模型，跳过测试。")