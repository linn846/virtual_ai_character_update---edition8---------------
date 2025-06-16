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

    print("\n压缩工具 (compression_utils.py) 测试运行完毕。")