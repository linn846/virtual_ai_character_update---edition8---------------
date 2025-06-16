import numpy as np
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
    main()import re
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

