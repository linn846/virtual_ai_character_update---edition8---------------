# 可以放在 database_utils.py 或者一个专门的 utils.py 文件中
import jieba
import json
import os
from config import KNOWLEDGE_SOURCE_JSON_FILENAME # 假设你的KG文件名在config中

# 全局或类变量来跟踪哪些世界的词典已经加载，避免重复加载
_loaded_world_dictionaries = set()

def load_kg_entities_to_jieba_dict_for_world(world_id: str, world_path: str):
    """
    从指定世界的知识图谱源JSON文件中提取实体（第一个和第三个词），
    并将其加载到Jieba的自定义词典中。
    """
    global _loaded_world_dictionaries
    if not world_id or not world_path:
        return

    if world_id in _loaded_world_dictionaries:
        # print(f"世界 '{world_id}' 的自定义词典已加载，跳过。")
        return

    kg_source_file = os.path.join(world_path, KNOWLEDGE_SOURCE_JSON_FILENAME)
    custom_dict_file_for_world = os.path.join(world_path, f"{world_id}_jieba_custom_dict.txt") # 为每个世界生成一个独立的词典文件

    entities_to_add = set()

    if os.path.exists(kg_source_file):
        try:
            with open(kg_source_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and "triples" in data and isinstance(data["triples"], list):
                for triple in data["triples"]:
                    if isinstance(triple, list) and len(triple) == 3:
                        # 提取第一个和第三个词作为实体
                        entity1 = str(triple[0]).strip()
                        entity3 = str(triple[2]).strip()
                        if entity1:
                            entities_to_add.add(entity1)
                        if entity3:
                            entities_to_add.add(entity3)
            
            if entities_to_add:
                # 创建或覆盖这个世界专用的自定义词典文件
                with open(custom_dict_file_for_world, 'w', encoding='utf-8') as f_dict:
                    for entity in entities_to_add:
                        # Jieba 自定义词典格式通常是：词语 [词频] [词性]
                        # 这里我们只提供词语，词频和词性可以省略，Jieba会使用默认值
                        # 为了确保分词，可以给一个较高的词频，例如： entity 100 nz
                        # 但简单起见，先只写入词语
                        f_dict.write(f"{entity}\n")
                
                # 加载这个新生成的自定义词典
                jieba.load_userdict(custom_dict_file_for_world)
                print(f"已为世界 '{world_id}' 加载了 {len(entities_to_add)} 个实体到Jieba自定义词典 (从 {custom_dict_file_for_world})。")
                _loaded_world_dictionaries.add(world_id)
            else:
                print(f"世界 '{world_id}' 的知识图谱源文件 '{kg_source_file}' 中未找到可提取的实体。")

        except json.JSONDecodeError:
            print(f"错误：解析世界 '{world_id}' 的知识图谱源文件 '{kg_source_file}' 失败。")
        except Exception as e:
            print(f"错误：加载世界 '{world_id}' 的KG实体到Jieba时发生错误: {e}")
    else:
        print(f"警告：世界 '{world_id}' 的知识图谱源文件 '{kg_source_file}' 未找到，无法加载自定义实体。")

# 在 main_app.py 或 database_utils.py 的初始化部分或切换世界时调用：
# 例如，在 database_utils.py 的 switch_active_world 函数成功切换后：


# 或者在 main_app.py 的 initial_ui_setup_on_load 和 handle_switch_world 后调用
# def update_jieba_dict_for_current_world():
#     if database_utils._active_world_id:
#         world_path = database_utils.get_world_path(database_utils._active_world_id)
#         if world_path:
#             load_kg_entities_to_jieba_dict_for_world(database_utils._active_world_id, world_path)
#
# app.load(...)
# .then(update_jieba_dict_for_current_world) # 在加载后更新
#
# world_select_dropdown.change(...)
# .then(update_jieba_dict_for_current_world) # 在切换世界后更新