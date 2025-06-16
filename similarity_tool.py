# similarity_tool.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys 
import os 

# Adjust path to import from parent directory if needed
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir) # Assuming similarity_tool is in a subdirectory
# if project_root not in sys.path:
#    sys.path.append(project_root)

try:
    import config # Import config first
    from embedding_utils import get_embedding, get_model_embedding_dimension # OLLAMA_EMBEDDING_MODEL_NAME is now in config
    from llm_utils import get_ollama_client 
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保此脚本与 embedding_utils.py, llm_utils.py 和 config.py 在正确的项目结构中，")
    print("或者已经将项目根目录添加到了PYTHONPATH。")
    sys.exit(1)

def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """计算两个嵌入向量之间的余弦相似度。"""
    if embedding1.ndim == 1:
        embedding1 = embedding1.reshape(1, -1)
    if embedding2.ndim == 1:
        embedding2 = embedding2.reshape(1, -1)
    
    if np.all(embedding1 == 0) or np.all(embedding2 == 0):
        # print("警告: 计算相似度的向量中至少有一个是零向量。相似度将为0。") # Less verbose in tool
        return 0.0
        
    return cosine_similarity(embedding1, embedding2)[0][0]

def main():
    print(f"--- 使用嵌入模型: {config.OLLAMA_EMBEDDING_MODEL_NAME} (来自 config.py) ---")

    try:
        print("正在检查Ollama连接和模型维度...")
        get_ollama_client() 
        dimension = get_model_embedding_dimension()
        print(f"嵌入维度为: {dimension}. 模型检查通过。\n")
    except Exception as e:
        print(f"初始化Ollama或获取模型维度失败: {e}")
        print("请确保Ollama服务正在运行，并且配置的嵌入模型已拉取且可用。")
        return

    sentences = [
        "今天天气真好，阳光明媚。",  
        "外面阳光灿烂，是个出游的好日子。", 
        "明天可能会下雨，记得带伞。", 
        "昨晚下了很大的雪。", 
        "我喜欢吃香蕉和苹果。", 
        "水果对健康非常有益，尤其是橙子。", 
        "这家餐厅的披萨非常好吃。", 
        "我晚餐想吃牛排。", 
        "人工智能正在改变世界。", 
        "机器学习是AI的一个重要分支。", 
        "量子计算机有望解决复杂问题。", 
        "猫是一种可爱的宠物。", 
        "这本小说情节跌宕起伏。", 
        "那只狗非常快速地奔跑。", 
        "小犬疾驰。", 
        "请把苹果手机递给我。", 
        "苹果公司的股价上涨了。" 
    ]

    print("正在为所有句子生成嵌入向量...")
    embeddings = []
    all_embeddings_successful = True
    for i, s in enumerate(sentences):
        # print(f"  生成嵌入 for: \"{s}\"") # Can be verbose
        emb = get_embedding(s)
        if np.all(emb == 0) and s.strip(): 
            print(f"  !! 警告: 句子 \"{s}\" 的嵌入为零向量!")
            all_embeddings_successful = False
        embeddings.append(emb)
    
    if not all_embeddings_successful:
        print("\n!! 注意：部分句子的嵌入为零向量，相似度计算可能不准确或无意义。请检查Ollama模型和日志。 !!\n")

    print("\n--- 句子对相似度计算 (余弦相似度) ---")
    print("(相似度范围: -1 到 1，越接近1表示越相似)\n")

    pairs_to_compare = [
        (0, 1), (0, 2), (0, 4), (4, 5), (4, 6), (8, 9), (8, 10),
        (0, 11), (8, 12), (13, 14), (4, 15), (15,16)
    ]

    for i, j in pairs_to_compare:
        if i < len(sentences) and j < len(sentences):
            similarity = calculate_cosine_similarity(embeddings[i], embeddings[j])
            print(f"句子1: \"{sentences[i]}\"")
            print(f"句子2: \"{sentences[j]}\"")
            print(f"余弦相似度: {similarity:.4f}\n")
        else:
            print(f"警告: 索引 ({i}, {j}) 超出句子列表范围。\n")
            
    print("\n--- \"今天天气真好，阳光明媚。\" 与其他句子的相似度 ---")
    query_sentence_index = 0
    query_embedding = embeddings[query_sentence_index]
    if np.all(query_embedding == 0) and sentences[query_sentence_index].strip():
        print(f"警告: 查询句子 \"{sentences[query_sentence_index]}\" 的嵌入是零向量，结果可能无意义。")

    for i in range(len(sentences)):
        if i == query_sentence_index:
            continue
        similarity = calculate_cosine_similarity(query_embedding, embeddings[i])
        print(f"与 \"{sentences[i][:30].ljust(30)}...\" \t的相似度: {similarity:.4f}")


    print("\n--- \"我喜欢吃香蕉和苹果。\" 与其他句子的相似度 ---")
    try:
        query_sentence_index_food = sentences.index("我喜欢吃香蕉和苹果。") 
        query_embedding_food = embeddings[query_sentence_index_food]
        if np.all(query_embedding_food == 0) and sentences[query_sentence_index_food].strip():
            print(f"警告: 查询句子 \"{sentences[query_sentence_index_food]}\" 的嵌入是零向量，结果可能无意义。")

        for i in range(len(sentences)):
            if i == query_sentence_index_food:
                continue
            similarity = calculate_cosine_similarity(query_embedding_food, embeddings[i])
            print(f"与 \"{sentences[i][:30].ljust(30)}...\" \t的相似度: {similarity:.4f}")
    except ValueError:
        print("错误：测试句子 '我喜欢吃香蕉和苹果。' 未在列表中找到。")


    print("\n脚本执行完毕。")