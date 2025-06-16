import gradio as gr
import time
import os
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import jieba
import logging
import json
from evaluation_utils import evaluate_retrieved_texts_relevance_llm
import config # Uses updated config

# --- Constants from config for direct use in UI logic if not passed around ---
# (Many of these are now more robustly sourced from config inside functions)

import database_utils
from llm_utils import generate_text, get_ollama_client, generate_summary
from embedding_utils import get_model_embedding_dimension, get_embedding
from compression_utils import compress_worldview_db_for_active_world
from kg_utils import (
    build_kg_for_active_world_from_json, 
    auto_update_kg_from_worldview, # New import
    rephrase_triples_for_prompt_llm # New import
)
from rerank_utils import rerank_documents, get_rerank_model
from evaluation_utils import evaluate_retrieved_texts_relevance_llm

# Explicitly import from config what's directly used in UI text or logic
from config import (
    # COMPRESSION_THRESHOLD is used in UI text
    # MAX_CHAT_HISTORY_TURNS is used as default if not overridden
    # HYBRID_SEARCH_ENABLED, RERANK_ENABLED, RERANK_MODEL_NAME for UI display
    # WORLDVIEW_LLM_EVAL_SCORE_THRESHOLD, MAX_WORLDVIEW_SNIPPETS_FOR_PROMPT for UI display
    # MAX_HISTORY_TURNS_FOR_SUMMARY_THRESHOLD for UI display
    # OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL_NAME, etc. for UI display
    KNOWLEDGE_SOURCE_JSON_FILENAME # For UI text
)


jieba.setLogLevel(logging.INFO) # Suppress jieba messages unless error

# --- Global State (minimal as per Gradio best practices) ---
_initial_world_activated_on_startup = False
_initial_active_world_id_on_startup = None
# character_name_for_history_context: Optional[str] = None # Managed within predict or via state if needed

# --- Initialization Block ---
try:
    print("正在执行 main_app.py 的初始化...")
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        print(f"主数据目录 '{config.DATA_DIR}' 已创建。")

    # Initialize core components and check models
    get_model_embedding_dimension() # Checks embedding model
    get_ollama_client() # Checks all configured LLMs
    if config.RERANK_ENABLED:
        get_rerank_model() # Checks rerank model

    # Attempt to activate the first available world
    available_worlds_init = database_utils.get_available_worlds()
    if available_worlds_init:
        first_world_id = list(available_worlds_init.keys())[0]
        if database_utils.switch_active_world(first_world_id):
            _initial_world_activated_on_startup = True
            _initial_active_world_id_on_startup = database_utils._active_world_id # Get the confirmed active ID
            print(f"已在启动时自动激活默认世界: '{database_utils.get_world_display_name(first_world_id)}'")
        else:
            print(f"警告：启动时未能自动激活默认世界 '{first_world_id}'。")
    else:
        print("提示：当前没有已创建的世界。请在'世界管理'标签页中添加新世界。")
    print("main_app.py 初始化完成。")
except Exception as e:
    print(f"致命错误：main_app.py 初始设置期间出错: {e}")
    import traceback
    traceback.print_exc()
    _initial_active_world_id_on_startup = f"初始化错误: {str(e)[:100]}..." # Store error for UI


# --- UI Refresh Helper Functions ---
def refresh_world_dropdown_choices_for_gradio():
    available_worlds = database_utils.get_available_worlds()
    return [(name, id_val) for id_val, name in available_worlds.items()]

def refresh_character_dropdown_choices():
    if not database_utils._active_world_id: return []
    return database_utils.get_character_names()

def refresh_worldview_status_display_text():
    if not database_utils._active_world_id:
        return "无活动世界。世界观信息不可用。"
    size = database_utils.get_worldview_size()
    world_name = database_utils.get_world_display_name(database_utils._active_world_id)
    status_text = f"世界 '{world_name}' | 世界观条目数: {size}. "
    ct = config.COMPRESSION_THRESHOLD
    if isinstance(ct, (int, float)) and ct > 0:
        if size > ct: status_text += f"建议压缩 (阈值: {ct})."
        else: status_text += f"压缩阈值: {ct}."
    else: status_text += "压缩阈值未有效配置或为零。"
    return status_text

def refresh_kg_status_display_text():
    if not database_utils._active_world_id:
        return "无活动世界。知识图谱信息不可用。"
    count = database_utils.get_kg_triples_count_for_active_world()
    world_name = database_utils.get_world_display_name(database_utils._active_world_id)
    return f"世界 '{world_name}' | 知识图谱三元组数量: {count}."

def get_active_world_markdown_text_for_global_display():
    active_id = database_utils._active_world_id
    if isinstance(active_id, str) and "初始化错误" in active_id:
        return f"<p style='color:red; font-weight:bold;'>应用初始化失败: {active_id}. 请检查控制台日志和Ollama服务。</p>"
    if active_id:
        return f"当前活动世界: **'{database_utils.get_world_display_name(active_id)}'** (ID: `{active_id}`)"
    else:
        return "<p style='color:orange; font-weight:bold;'>当前无活动世界。请从上方选择或在“世界管理”中创建一个新世界。</p>"

# --- UI Element Clearing Helper ---
def clear_textboxes_and_checkboxes(*args):
    updates = []
    for arg_comp in args: # Assuming args are Gradio components
        comp_type_str = str(type(arg_comp)) # Heuristic type check
        if 'Checkbox' in comp_type_str:
            updates.append(gr.Checkbox(value=False))
        elif 'Textbox' in comp_type_str:
            updates.append(gr.Textbox(value=""))
        elif 'Dropdown' in comp_type_str:
            updates.append(gr.Dropdown(value=None))
        else: # Fallback for other types or if type check is unreliable
            updates.append(gr.update(value=None)) # Generic update
    return tuple(updates) if updates else gr.update()


# --- Centralized UI Update Logic ---
def update_all_ui_elements_after_world_change(feedback_message: str = "", specific_feedback_elem_id: Optional[str] = None):
    world_choices_dd = refresh_world_dropdown_choices_for_gradio()
    current_active_id = database_utils._active_world_id
    is_world_active_and_valid = current_active_id is not None and not (isinstance(current_active_id, str) and "初始化错误" in current_active_id)

    char_choices_dd = refresh_character_dropdown_choices() if is_world_active_and_valid else []
    wv_status_text_val = refresh_worldview_status_display_text()
    kg_status_text_val = refresh_kg_status_display_text()
    global_md_text_val = get_active_world_markdown_text_for_global_display()

    # Determine interactivity based on whether a valid world is active
    can_interact_with_world_features = is_world_active_and_valid
    # Predict button needs a world AND characters
    predict_button_interactive = can_interact_with_world_features and bool(char_choices_dd)

    updates_map = {
        "world_select_dropdown": gr.Dropdown(choices=world_choices_dd, value=current_active_id if is_world_active_and_valid else None, interactive=True),
        "global_active_world_display": gr.Markdown(value=global_md_text_val),
        
        # World Management Tab
        "new_world_id_input": gr.Textbox(value="", interactive=True),
        "new_world_name_input": gr.Textbox(value="", interactive=True),
        "add_world_button": gr.Button(interactive=True),
        "confirm_delete_world_checkbox": gr.Checkbox(value=False, interactive=can_interact_with_world_features),
        "delete_world_button": gr.Button(interactive=False), # Controlled by checkbox change

        # Character Management Tab
        "char_name_input": gr.Textbox(value="", interactive=can_interact_with_world_features),
        "char_full_desc_input": gr.Textbox(value="", interactive=can_interact_with_world_features),
        "add_char_button": gr.Button(interactive=can_interact_with_world_features),
        "character_select_for_delete_dropdown": gr.Dropdown(choices=char_choices_dd, value=None, interactive=can_interact_with_world_features and bool(char_choices_dd)),
        "delete_char_button": gr.Button(interactive=False), # Controlled by dropdown change
        "view_chars_button": gr.Button(interactive=can_interact_with_world_features),
        "view_characters_output": gr.Textbox(value="角色列表将显示在此处。" if can_interact_with_world_features else "无活动世界或初始化错误。", interactive=False),


        # Worldview Management Tab
        "worldview_text_input": gr.Textbox(value="", interactive=can_interact_with_world_features),
        "add_worldview_button": gr.Button(interactive=can_interact_with_world_features),
        "worldview_status_display": gr.Textbox(value=wv_status_text_val, interactive=False),
        "compress_worldview_button": gr.Button(interactive=can_interact_with_world_features and database_utils.get_worldview_size() > 1),

        # Knowledge Graph Tab
        "kg_status_display": gr.Textbox(value=kg_status_text_val, interactive=False),
        "build_kg_from_json_button": gr.Button(interactive=can_interact_with_world_features),
        "auto_update_kg_from_worldview_button": gr.Button(interactive=can_interact_with_world_features and database_utils.get_worldview_size() > 0),


        # Prediction Tab
        "char_select_dropdown_pred_tab": gr.Dropdown(choices=char_choices_dd, value=None, interactive=can_interact_with_world_features and bool(char_choices_dd)),
        "situation_query_input": gr.Textbox(value="", interactive=predict_button_interactive), # Depends on char selected too
        "predict_button": gr.Button(interactive=predict_button_interactive),
        "retrieved_info_display": gr.Markdown(value="检索到的背景信息和LLM评估结果将显示在此处。" if can_interact_with_world_features else "无活动世界或初始化错误。"),
        "chatbot_display": gr.Chatbot(value=[], label="对话历史", visible=is_world_active_and_valid), # Show/hide based on world

        # Feedback Textboxes (clear them unless a specific one is targeted)
        "world_switch_feedback": gr.Textbox(value="", interactive=False),
        "add_world_feedback_output": gr.Textbox(value="", interactive=False),
        "delete_world_feedback_output": gr.Textbox(value="", interactive=False),
        "char_op_feedback_output": gr.Textbox(value="", interactive=False),
        "worldview_feedback_output": gr.Textbox(value="", interactive=False),
        "compression_status_output": gr.Textbox(value="", interactive=False),
        "kg_build_status_output": gr.Textbox(value="", interactive=False),
    }

    # If a specific feedback message and element are provided, update that element
    if specific_feedback_elem_id and specific_feedback_elem_id in updates_map:
        # This assumes specific_feedback_elem_id is for a Textbox or similar component that takes a simple value
        updates_map[specific_feedback_elem_id] = gr.Textbox(value=feedback_message, interactive=False)
    elif feedback_message and "world_switch_feedback" in updates_map: # Default to world_switch_feedback if no specific target
         updates_map["world_switch_feedback"] = gr.Textbox(value=feedback_message, interactive=False)
    
    return updates_map


# --- Event Handlers ---
def handle_add_world(world_id_input: str, world_name_input: str, progress=gr.Progress()):
    progress(0.1, desc="校验输入...")
    feedback_msg = ""
    if not world_id_input.strip() or not world_name_input.strip():
        feedback_msg = "错误：世界ID和世界名称不能为空。"
    else:
        progress(0.3, desc=f"尝试添加世界 '{world_name_input.strip()}'...")
        message = database_utils.add_world(world_id_input.strip(), world_name_input.strip())
        feedback_msg = message
        if "已添加" in message:
            progress(0.7, desc="世界已添加，尝试激活...")
            if database_utils.switch_active_world(world_id_input.strip()):
                world_name_disp = database_utils.get_world_display_name(world_id_input.strip())
                feedback_msg += f" 并已激活 '{world_name_disp}'。"
            else:
                feedback_msg += " 但激活失败，请手动选择。"
    progress(1.0, desc="操作完成。")
    all_updates_dict = update_all_ui_elements_after_world_change(feedback_msg, "add_world_feedback_output")
    all_updates_dict["chatbot_display"] = gr.Chatbot(value=[]) # Clear chat on world add/switch
    all_updates_dict["conversation_history_state"] = []
    return map_updates_to_ordered_list(all_updates_dict, ordered_output_components_for_ui_updates)


def handle_delete_world(confirm_delete_checkbox: bool, progress=gr.Progress()):
    progress(0.1, desc="校验操作...")
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
        world_name_to_delete = database_utils.get_world_display_name(world_id_to_delete)
        progress(0.3, desc=f"准备删除世界 '{world_name_to_delete}'...")
        message = database_utils.delete_world(world_id_to_delete) # This updates _active_world_id
        feedback_msg = message
        # The message from delete_world itself is usually enough
        # if "已成功删除" in message or "数据目录可能未能完全删除" in message :
        #     feedback_msg = f"世界 '{world_name_to_delete}' 相关操作已执行：{message}"
    progress(1.0, desc="操作完成。")
    all_updates_dict = update_all_ui_elements_after_world_change(feedback_msg, "delete_world_feedback_output")
    all_updates_dict["chatbot_display"] = gr.Chatbot(value=[]) # Clear chat
    all_updates_dict["conversation_history_state"] = []
    return map_updates_to_ordered_list(all_updates_dict, ordered_output_components_for_ui_updates)


def handle_switch_world(world_id_selected: Optional[str], progress=gr.Progress()):
    progress(0.1, desc="开始切换世界...")
    feedback_msg = ""
    if not world_id_selected: # If "None" or empty is selected from dropdown
        current_active = database_utils._active_world_id
        if current_active is not None and not (isinstance(current_active, str) and "初始化错误" in current_active):
            database_utils.switch_active_world(None) # Deactivate
            feedback_msg = "已取消活动世界。"
        # else no active world to deactivate, or it was an error state
    elif isinstance(world_id_selected, str) and "初始化错误" in world_id_selected:
        feedback_msg = "无法选择一个错误状态作为活动世界。"
        database_utils.switch_active_world(None) # Ensure no world is active
    else:
        progress(0.5, desc=f"正在切换到世界ID '{world_id_selected}'...")
        if database_utils.switch_active_world(world_id_selected):
            world_name_disp = database_utils.get_world_display_name(world_id_selected)
            feedback_msg = f"已激活世界: '{world_name_disp}'"
        else:
            # switch_active_world already prints errors, but we can add a UI message
            feedback_msg = f"切换到世界ID '{world_id_selected}' 失败。该世界可能已损坏或不存在。"
            database_utils.switch_active_world(None) # Ensure no world is active if switch failed
    progress(1.0, desc="切换完成。")
    all_updates_dict = update_all_ui_elements_after_world_change(feedback_msg, "world_switch_feedback")
    all_updates_dict["chatbot_display"] = gr.Chatbot(value=[]) # Clear chat
    all_updates_dict["conversation_history_state"] = []
    return map_updates_to_ordered_list(all_updates_dict, ordered_output_components_for_ui_updates)


def handle_character_selection_change(char_name: Optional[str]):
    # This handler updates interactivity of prediction inputs when character selection changes
    is_world_truly_active = database_utils._active_world_id is not None and \
                           not (isinstance(database_utils._active_world_id, str) and "初始化错误" in database_utils._active_world_id)
    
    can_predict = bool(char_name) and is_world_truly_active
    # Clear chat history and related states when character changes
    return (
        gr.Chatbot(value=[]),  # chatbot_display
        [],                    # conversation_history_state
        gr.Textbox(value="", interactive=can_predict), # situation_query_input (clear and set interactivity)
        gr.Button(interactive=can_predict)             # predict_button
    )

def handle_add_character(name: str, full_description: str):
    feedback_msg = "请先选择并激活一个有效世界。"
    is_world_truly_active = database_utils._active_world_id is not None and \
                           not (isinstance(database_utils._active_world_id, str) and "初始化错误" in database_utils._active_world_id)
    if is_world_truly_active:
        if not name.strip() or not full_description.strip():
            feedback_msg = "角色名称和完整描述不能为空。"
        else:
            message = database_utils.add_character(name.strip(), full_description.strip())
            feedback_msg = message
    
    # Refresh character dropdowns and predict button interactivity
    char_choices = refresh_character_dropdown_choices() if is_world_truly_active else []
    can_interact_char_dd = is_world_truly_active and bool(char_choices)
    
    char_choices_updated_pred_tab = gr.Dropdown(choices=char_choices, value=None, interactive=can_interact_char_dd)
    char_choices_updated_del_tab = gr.Dropdown(choices=char_choices, value=None, interactive=can_interact_char_dd)
    
    # Predict button interactivity depends on world AND if any character is selected (or available)
    # For now, just base on whether characters *exist* in the newly updated list
    predict_btn_interactive = is_world_truly_active and bool(char_choices)
    situation_input_interactive = predict_btn_interactive # Same logic

    return (
        feedback_msg, 
        char_choices_updated_pred_tab, 
        char_choices_updated_del_tab,
        gr.Button(interactive=predict_btn_interactive), 
        gr.Textbox(interactive=situation_input_interactive)
    )

def handle_delete_character(character_name_to_delete: Optional[str]):
    feedback_msg = "错误：请先选择有效活动世界。"
    is_world_truly_active = database_utils._active_world_id is not None and \
                           not (isinstance(database_utils._active_world_id, str) and "初始化错误" in database_utils._active_world_id)
    if is_world_truly_active:
        if not character_name_to_delete:
            feedback_msg = "错误：请从下拉列表中选择要删除的角色。"
        else:
            message = database_utils.delete_character(character_name_to_delete)
            feedback_msg = message

    # Refresh character dropdowns and predict button interactivity
    char_choices = refresh_character_dropdown_choices() if is_world_truly_active else []
    can_interact_char_dd = is_world_truly_active and bool(char_choices)
    
    char_choices_updated_pred_tab = gr.Dropdown(choices=char_choices, value=None, interactive=can_interact_char_dd)
    char_choices_updated_del_tab = gr.Dropdown(choices=char_choices, value=None, interactive=can_interact_char_dd)
    
    predict_btn_interactive = is_world_truly_active and bool(char_choices)
    situation_input_interactive = predict_btn_interactive

    return (
        feedback_msg, 
        char_choices_updated_pred_tab, 
        char_choices_updated_del_tab,
        gr.Button(interactive=predict_btn_interactive), 
        gr.Textbox(interactive=situation_input_interactive)
    )

def handle_view_characters():
    if not database_utils._active_world_id or not isinstance(database_utils._active_world_id, str) or "初始化错误" in database_utils._active_world_id:
        return "请先选择并激活一个有效世界，或修复初始化错误。"
    
    chars = database_utils.get_all_characters() # This now returns structured_traits too
    world_name_active = database_utils.get_world_display_name(database_utils._active_world_id)
    
    if not chars: return f"当前活动世界 '{world_name_active}' 中没有角色。"
    
    output = f"当前活动世界 '{world_name_active}' 的角色列表 ({len(chars)} 个):\n" + "="*40 + "\n"
    for char_data in chars:
        output += f"  名称: {char_data['name']}\n"
        
        desc_to_show = char_data.get('summary_description', '').strip()
        if not desc_to_show or len(desc_to_show) < 20: # If summary is too short or missing, use full desc
            desc_to_show = char_data.get('full_description', '（无详细描述）')
        
        desc_snippet = desc_to_show.replace('\n', ' ').replace('\r', '') # Flatten for snippet
        desc_snippet = (desc_snippet[:150] + '...') if len(desc_snippet) > 150 else desc_snippet
        output += f"  概要/设定: {desc_snippet}\n"
        
        structured_traits = char_data.get("structured_traits", {})
        if structured_traits and any(structured_traits.values()): # Only show if there's any content
            output += f"  结构化特质 (部分预览):\n"
            for key, val in structured_traits.items():
                if val: # Only print if there's data for this trait
                    # Preview list/dict as JSON string for better readability if complex
                    val_preview_str = json.dumps(val, ensure_ascii=False) if isinstance(val, (list, dict)) else str(val)
                    val_preview = val_preview_str[:100] + ('...' if len(val_preview_str) > 100 else '')
                    output += f"    - {key.replace('_', ' ').capitalize()}: {val_preview}\n"
        output += "---\n"
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
            message = database_utils.add_worldview_text(text_input.strip())
            feedback_msg = message
            wv_status_update = gr.Textbox(value=refresh_worldview_status_display_text(), interactive=False) # Refresh status after adding
            
            if not message.startswith("错误："): # If add was successful
                 current_size = database_utils.get_worldview_size()
                 compress_btn_interactive_val = current_size > 1 # Can compress if > 1 item
                 
                 # Auto-compression suggestion logic
                 ct_val = config.COMPRESSION_THRESHOLD
                 auto_compress_message = ""
                 if isinstance(ct_val, (int, float)) and ct_val > 0 and current_size > ct_val:
                     if current_size > ct_val * 1.5: # Significantly over threshold
                         auto_compress_message = f"\n提示: 世界观条目数 ({current_size}) 远超压缩阈值 ({ct_val})。建议手动压缩。"
                     else: # Near or just over threshold
                         auto_compress_message = f"\n提示: 世界观条目数 ({current_size}) 已达到或接近压缩阈值 ({ct_val})。可考虑手动压缩。"
                 
                 if auto_compress_message and auto_compress_message not in feedback_msg: # Append if not already part of main msg
                    feedback_msg += auto_compress_message
                    
    compress_btn_interactive = gr.Button(interactive=(compress_btn_interactive_val and is_world_truly_active))
    return feedback_msg, wv_status_update, compress_btn_interactive


def handle_compress_worldview_button_streaming():
    initial_wv_status_text = refresh_worldview_status_display_text()
    is_world_truly_active = database_utils._active_world_id is not None and \
                           not (isinstance(database_utils._active_world_id, str) and "初始化错误" in database_utils._active_world_id)

    if not is_world_truly_active:
        yield "错误：请先选择并激活一个有效世界才能进行压缩。", initial_wv_status_text, gr.Button(interactive=False)
        return

    world_name_active = database_utils.get_world_display_name(database_utils._active_world_id)
    yield f"正在为世界 '{world_name_active}' 压缩世界观数据库... 这可能需要一些时间。", initial_wv_status_text, gr.Button(interactive=False)
    
    message = ""
    try:
        # Pass force_compress=True as it's a manual button click
        message = compress_worldview_db_for_active_world(force_compress=True) 
    except Exception as e_compress:
        message = f"为世界 '{world_name_active}' 压缩时发生严重错误: {e_compress}"
        import traceback
        traceback.print_exc() # Log full traceback to console
        
    final_wv_status_text = refresh_worldview_status_display_text()
    can_compress_again = is_world_truly_active and database_utils.get_worldview_size() > 1
    yield message, final_wv_status_text, gr.Button(interactive=can_compress_again)


# --- RRF Helper (moved from predict for clarity, might be a util later) ---
def _reciprocal_rank_fusion(
    retrieved_results_dict: Dict[str, List[Tuple[int, float, str]]], # {method: [(id, score, text), ...]}
    rrf_k_param: int = 60 # From config.HYBRID_RRF_K ideally
) -> List[int]: # Returns sorted list of doc_ids
    fused_scores: Dict[int, float] = {} # doc_id -> RRF_score
    
    for method, results in retrieved_results_dict.items():
        # Ensure results are sorted by their original method's score (desc for similarity, asc for distance)
        # Assuming higher original score is better for simplicity here. Adjust if methods have different score meanings.
        # For FAISS L2 distance, lower is better. For BM25/Keyword match count, higher is better.
        # This RRF implementation assumes all input scores are "higher is better".
        # If FAISS distances are used, they should be inverted or normalized before this step.
        # (Current search_worldview_semantic returns distance, needs inversion for RRF like this)
        
        # Let's assume semantic search results (distances) are already handled or will be before RRF
        # Or, RRF can be adapted if score meanings differ, but simpler if they are harmonized.
        # For now, this RRF assumes higher score = better rank.
        
        ranked_doc_ids = [doc_id for doc_id, _, _ in results] # Assuming results are pre-sorted
        
        for rank, doc_id in enumerate(ranked_doc_ids):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
            fused_scores[doc_id] += 1.0 / (rrf_k_param + rank + 1) # Rank is 0-indexed
            
    if not fused_scores:
        return []
        
    # Sort by RRF score in descending order
    sorted_fused_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, score in sorted_fused_results]


def format_and_process_conversation_history(
    history: List[Tuple[str, str]], # List of (user_query, ai_response)
    current_query: str,
    character_name: str, # Needed for LLM prompt context
    max_raw_turns: int,
    summary_threshold_turns: int,
    max_input_len_for_summary: int,
    target_summary_len: int, # Approx target length for summary
    processing_model: str # LLM for history processing
) -> Tuple[str, str]: # Returns (formatted_history_for_prompt, log_string)
    log_entries = []
    if not history:
        log_entries.append("- *无对话历史。*")
        return "", "\n".join(log_entries)

    num_turns = len(history)
    log_entries.append(f"- **原始对话历史总轮数**: {num_turns}")

    if num_turns > summary_threshold_turns:
        log_entries.append(f"- *对话历史超过 {summary_threshold_turns} 轮，尝试LLM辅助处理 (模型: {processing_model})。*")
        
        # Combine all history into a single string for LLM processing
        full_history_text = ""
        for i, (user_turn, ai_turn) in enumerate(history):
            full_history_text += f"轮次 {i+1}:\n用户: {user_turn}\n{character_name}: {ai_turn}\n\n"
        full_history_text = full_history_text.strip()

        truncated_history_for_llm = full_history_text
        if len(full_history_text) > max_input_len_for_summary:
            # Take most recent part, ensure it's not just half a turn
            # A more robust truncation would find a turn boundary.
            # For simplicity, just char-based truncation for now.
            truncated_history_for_llm = full_history_text[-max_input_len_for_summary:] 
            if len(full_history_text) > len(truncated_history_for_llm): # If actually truncated
                 truncated_history_for_llm = "...[早期历史已截断]\n\n" + truncated_history_for_llm
            log_entries.append(f"  - *送往LLM的历史文本已从 {len(full_history_text)} 截断至约 {len(truncated_history_for_llm)} 字符。*")

        history_processing_prompt = f"""
以下是角色“{character_name}”与用户的对话历史。当前用户提出了新的问题/情境：“{current_query}”。
请你完成两项任务，以帮助角色“{character_name}”更好地衔接和回应当前对话：
1.  对整个对话历史生成一个**极其简洁的核心摘要**，捕捉对话的主要脉络、已达成的共识、或遗留的关键问题，目标是帮助角色快速回忆起之前的对话重点，长度约 {target_summary_len // 2} 到 {target_summary_len} 字符。
2.  从对话历史中，挑选出与当前用户问题“{current_query}”的**主题、涉及人物、或情感状态最直接相关或最有延续性的最多2-3轮完整对话（包括用户和角色双方）**。

严格按照以下JSON格式输出...
{{
  "history_summary": "<这里是对话历史核心摘要>",
  "relevant_turns_text": "<这里是挑选出的最相关2-3轮对话的原文拼接... 如果确实没有特别相关的历史轮次，可以说明'未找到与当前问题直接相关的历史轮次'>"
}}
...
"""
        system_message = "你是一个专业的对话分析和信息提炼助手。"
        
        processed_history_json_str = generate_text(
            prompt=history_processing_prompt,
            system_message=system_message,
            model_name=processing_model,
            use_json_format=True,
            temperature=config.OLLAMA_HISTORY_PROCESSING_TEMPERATURE # Specific temp for this task
        )

        summary_str = ""
        relevant_turns_str = ""
        if processed_history_json_str.startswith("错误:") or not processed_history_json_str.strip():
            log_entries.append(f"  - *LLM处理历史失败或返回空: {processed_history_json_str}。将回退到使用最近 {max_raw_turns} 轮对话。*")
        else: # Try to parse JSON
            try:
                # Clean potential markdown if LLM wraps JSON in it
                if processed_history_json_str.strip().startswith("```json"):
                    processed_history_json_str = processed_history_json_str.strip()[7:]
                    if processed_history_json_str.strip().endswith("```"):
                        processed_history_json_str = processed_history_json_str.strip()[:-3]
                elif processed_history_json_str.strip().startswith("```"): # More generic ``` block
                    processed_history_json_str = processed_history_json_str.strip()[3:]
                    if processed_history_json_str.strip().endswith("```"):
                        processed_history_json_str = processed_history_json_str.strip()[:-3]


                processed_data = json.loads(processed_history_json_str.strip())
                summary_str = processed_data.get("history_summary", "").strip()
                relevant_turns_str = processed_data.get("relevant_turns_text", "").strip()
                log_entries.append(f"  - **LLM生成的历史摘要**: {summary_str if summary_str else '无'}")
                log_entries.append(f"  - **LLM挑选的相关历史轮次**: {relevant_turns_str if relevant_turns_str else '无'}")
            except json.JSONDecodeError:
                log_entries.append(f"  - *LLM返回的JSON解析失败: {processed_history_json_str[:200]}。将回退。*")
            except Exception as e_hist_parse:
                 log_entries.append(f"  - *处理LLM历史响应时发生未知错误: {e_hist_parse}。将回退。*")


        # Construct final history string for prompt using LLM processed parts
        final_history_for_prompt_parts = []
        if summary_str:
            final_history_for_prompt_parts.append(f"之前的对话概要：{summary_str}")
        if relevant_turns_str and relevant_turns_str.lower() not in ["未找到特别相关的历史轮次", "无"]:
            final_history_for_prompt_parts.append(f"与当前问题相关的历史对话片段：\n{relevant_turns_str}")
        
        if final_history_for_prompt_parts: # If LLM processing was successful and yielded content
            return "\n\n".join(final_history_for_prompt_parts), "\n".join(log_entries)
        # Fallback to recent N turns if LLM processing failed or produced nothing useful

    # Fallback or if history is short: use most recent N raw turns
    log_entries.append(f"- *使用最近最多 {max_raw_turns} 轮原始对话。*")
    recent_history_text = ""
    recent_turns_to_format = history[-max_raw_turns:]
    for user_turn, ai_turn in recent_turns_to_format:
        # Using "你" for user, "我 ({character_name})" for AI for consistency in prompt
        recent_history_text += f"你之前问/说：{user_turn}\n我 ({character_name}) 当时回应：{ai_turn}\n"
    
    # Log the content being used for prompt (first few chars for brevity)
    for i, (user_turn, ai_turn) in enumerate(recent_turns_to_format):
        log_entries.append(f"  - (轮次 -{len(recent_turns_to_format)-i}) 用户: {user_turn[:50]}... | {character_name}: {ai_turn[:50]}...")

    return recent_history_text.strip(), "\n".join(log_entries)

# --- Main Prediction Handler ---
# main_app.py
# ... (确保所有必要的导入都在文件顶部)
# import json # 如果需要解析情境拓展的JSON
# from config import (
#    OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL_NAME, HYBRID_SEARCH_ENABLED, RERANK_ENABLED,
#    RERANK_MODEL_NAME, WORLDVIEW_LLM_EVAL_SCORE_THRESHOLD, MAX_WORLDVIEW_SNIPPETS_FOR_PROMPT,
#    MAX_CHAT_HISTORY_TURNS, MAX_HISTORY_TURNS_FOR_SUMMARY_THRESHOLD,
#    MAX_HISTORY_TEXT_LENGTH_FOR_SUMMARY_INPUT, TARGET_HISTORY_SUMMARY_LENGTH,
#    OLLAMA_HISTORY_PROCESSING_MODEL, OLLAMA_CONTEXT_SNIPPING_MODEL, TARGET_CHAR_DESC_OUTPUT_FOR_PROMPT_LLM,
#    MAX_CHAR_DESC_INPUT_FOR_CONTEXTUAL_SNIPPING_LLM, OLLAMA_SUMMARY_TEMPERATURE,
#    SEMANTIC_SEARCH_TOP_K_HYBRID, BM25_SEARCH_TOP_K_HYBRID, KEYWORD_SEARCH_TOP_K_HYBRID,
#    HYBRID_RRF_K, HYBRID_FINAL_TOP_K, RERANK_TOP_K_FINAL, MAX_RETRIEVED_TEXTS_FOR_LLM_EVALUATION,
#    OLLAMA_EVALUATION_MODEL, OLLAMA_KG_EXTRACTION_MODEL, OLLAMA_KG_REPHRASE_MODEL,
#    MAX_KG_TRIPLES_FOR_PROMPT, OLLAMA_DEFAULT_TEMPERATURE, OLLAMA_DEFAULT_TOP_P
# )
# import database_utils
# from llm_utils import generate_text # generate_summary
# from evaluation_utils import evaluate_retrieved_texts_relevance_llm
# from kg_utils import rephrase_triples_for_prompt_llm
# import jieba # 如果在函数内部使用
# import numpy as np # 如果在函数内部使用
# from typing import List, Tuple, Optional, Dict # 确保类型提示正确

# ... (其他函数定义) ...

# --- Main Prediction Handler (修改后版本) ---
def handle_predict_behavior(
    character_name: Optional[str],
    situation_query: str,
    chatbot_ui_value: List[List[Optional[str]]], # Current value from gr.Chatbot
    history_state_value: List[Tuple[str,str]] # Internal state for conversation
):
    # --- 0. Initialization and Input Validation ---
    current_conversation_history: List[Tuple[str,str]] = list(history_state_value) # Work with a copy
    retrieved_info_log = ["# 信息获取与检索流程分析 (Gradio 日志)\n"]

    is_world_truly_active = database_utils._active_world_id is not None and \
                           not (isinstance(database_utils._active_world_id, str) and "初始化错误" in database_utils._active_world_id)

    if not is_world_truly_active:
        error_msg = "错误：请先选择并激活一个有效活动世界。"
        current_conversation_history.append((situation_query, error_msg))
        retrieved_info_log.append(f"**错误:** {error_msg}")
        yield current_conversation_history, current_conversation_history, "\n".join(retrieved_info_log)
        return
    
    active_world_display_name = database_utils.get_world_display_name(database_utils._active_world_id)
    retrieved_info_log.append(f"**活动世界:** {active_world_display_name} (ID: {database_utils._active_world_id})")

    if not character_name:
        error_msg = "错误：请选择一个角色。"
        current_conversation_history.append((situation_query, error_msg))
        retrieved_info_log.append(f"**错误:** 未选择角色。")
        yield current_conversation_history, current_conversation_history, "\n".join(retrieved_info_log)
        return
    retrieved_info_log.append(f"\n## 1. 角色名称\n- **当前角色**: {character_name}")

    if not situation_query.strip():
        error_msg = "错误：请输入情境或问题。"
        current_conversation_history.append((situation_query, error_msg))
        retrieved_info_log.append(f"**错误:** 情境为空。")
        yield current_conversation_history, current_conversation_history, "\n".join(retrieved_info_log)
        return

    character_data = database_utils.get_character(character_name)
    if not character_data:
        error_msg = f"错误：在当前世界 '{active_world_display_name}' 中未找到角色 '{character_name}'。"
        current_conversation_history.append((situation_query, error_msg))
        retrieved_info_log.append(f"**错误:** {error_msg}")
        yield current_conversation_history, current_conversation_history, "\n".join(retrieved_info_log)
        return

    # --- 1. Character Settings (Contextual Snipping + Structured Traits) ---
    retrieved_info_log.append(f"\n## 2. 角色设定 ({character_name}) - 上下文感知剪裁与结构化特质应用")
    full_char_desc = character_data.get('full_description', '（无完整描述）')
    summary_char_desc = character_data.get('summary_description', '（无概要描述）')
    structured_traits = character_data.get('structured_traits', {})

    # (日志记录部分不变)
    retrieved_info_log.append(f"- **原始完整描述 (部分)**:\n  ```text\n  {full_char_desc[:200].strip()}...\n  ```")
    retrieved_info_log.append(f"- **原始概要描述**: {summary_char_desc.strip()}")
    if structured_traits and any(structured_traits.values()):
        retrieved_info_log.append(f"- **已提取的结构化特质 (部分预览)**:")
        for key, val in structured_traits.items():
            if val:
                val_preview = str(val)[:80] + ('...' if len(str(val)) > 80 else '')
                retrieved_info_log.append(f"    - {key.replace('_',' ').capitalize()}: {val_preview}")
    else:
        retrieved_info_log.append(f"- **已提取的结构化特质**: 无或为空。")

    text_for_char_snipping_llm = f"角色“{character_name}”的完整描述:\n{full_char_desc}\n\n角色概要:\n{summary_char_desc}\n\n"
    if structured_traits and any(structured_traits.values()):
        text_for_char_snipping_llm += "角色的结构化特质分析 (请优先参考这些信息):\n"
        for key, val in structured_traits.items():
            if val:
                trait_val_str = json.dumps(val, ensure_ascii=False) if isinstance(val, (list, dict)) else str(val)
                text_for_char_snipping_llm += f"- {key.replace('_', ' ').capitalize()}: {trait_val_str}\n"
    
    truncated_input_for_snipping = text_for_char_snipping_llm
    if len(text_for_char_snipping_llm) > config.MAX_CHAR_DESC_INPUT_FOR_CONTEXTUAL_SNIPPING_LLM:
        truncated_input_for_snipping = text_for_char_snipping_llm[:config.MAX_CHAR_DESC_INPUT_FOR_CONTEXTUAL_SNIPPING_LLM] + "...[原始描述过长已截断]"
        retrieved_info_log.append(f"- *注意: 送入LLM剪裁的角色信息已从 {len(text_for_char_snipping_llm)} 截断至 {len(truncated_input_for_snipping)} 字符。*")
    
    char_snipping_model_to_use = config.OLLAMA_CONTEXT_SNIPPING_MODEL
    retrieved_info_log.append(f"- **LLM辅助上下文剪裁模型**: {char_snipping_model_to_use}")
    
    char_snipping_prompt = f"""
当前情境/问题:
---
{situation_query}
---
角色“{character_name}”的背景设定信息（可能包含完整描述、概要和结构化特质分析）:
---
{truncated_input_for_snipping}
---
请严格依据“当前情境/问题”，从“角色背景设定信息”中**精准抽取**并**凝练概括**出那些将**直接主导**角色“{character_name}”在当前情境下思考模式、情感反应和行为决策的**关键设定要素**。
这些要素可能包括：
- 与当前冲突直接相关的性格特质 (例如：面对挑战时的勇敢、处理人际关系时的圆滑或直接)。
- 能够用于解决当前问题的特定技能或知识。
- 与当前情境中人物或事件相关的过往关键经历或已形成的观点。
- 驱动角色在当前情境下做出选择的核心价值观或目标。
请确保输出的文本简洁明了，**高度聚焦于上述直接相关要素**，总长度控制在 {config.TARGET_CHAR_DESC_OUTPUT_FOR_PROMPT_LLM // 2} 到 {config.TARGET_CHAR_DESC_OUTPUT_FOR_PROMPT_LLM} 字符之间。
直接输出提取并精炼后的核心设定文本，不要包含任何解释、标签或引言。
"""
    contextual_char_desc_for_prompt = generate_text(
        prompt=char_snipping_prompt,
        system_message="你是一个精准的角色信息提取和精炼助手。",
        model_name=char_snipping_model_to_use,
        temperature=config.OLLAMA_CONTEXT_SNIPPING_TEMPERATURE # 使用特定温度
    )

    char_desc_to_use_in_prompt = ""
    if contextual_char_desc_for_prompt.startswith("错误:") or not contextual_char_desc_for_prompt.strip():
        retrieved_info_log.append(f"- **LLM辅助上下文剪裁角色设定**: 失败或返回空。错误: {contextual_char_desc_for_prompt if contextual_char_desc_for_prompt.strip() else '空响应'}。")
        char_desc_to_use_in_prompt = summary_char_desc.strip() 
        if not char_desc_to_use_in_prompt: 
            char_desc_to_use_in_prompt = (full_char_desc[:config.TARGET_CHAR_DESC_OUTPUT_FOR_PROMPT_LLM] + "...") if len(full_char_desc) > config.TARGET_CHAR_DESC_OUTPUT_FOR_PROMPT_LLM else full_char_desc
        retrieved_info_log.append(f"- **回退使用的角色设定 (用于Prompt)**:\n  ```text\n  {char_desc_to_use_in_prompt}\n  ```")
    else:
        contextual_char_desc_for_prompt = contextual_char_desc_for_prompt.strip()
        if len(contextual_char_desc_for_prompt) > config.TARGET_CHAR_DESC_OUTPUT_FOR_PROMPT_LLM * 1.2: 
            contextual_char_desc_for_prompt = contextual_char_desc_for_prompt[:config.TARGET_CHAR_DESC_OUTPUT_FOR_PROMPT_LLM] + "...[LLM剪裁后再次截断]"
        retrieved_info_log.append(f"- **LLM辅助上下文剪裁后的角色设定 (用于Prompt)**:\n  ```text\n  {contextual_char_desc_for_prompt}\n  ```")
        char_desc_to_use_in_prompt = contextual_char_desc_for_prompt

    # --- 2. Query / Situation (Potentially Expanded using new JSON-based prompt) ---
    retrieved_info_log.append(f"\n## 3. 问题 / 情境分析与拓展")
    retrieved_info_log.append(f"- **原始问题/情境**: ```text\n{situation_query}\n```")
    
    expansion_model_to_use = getattr(config, 'OLLAMA_QUERY_EXPANSION_MODEL', config.OLLAMA_MODEL) 
    retrieved_info_log.append(f"- **情境分析/拓展模型**: {expansion_model_to_use}")

    # 使用新的情境分析/拓展Prompt
    expansion_analysis_prompt = f"""
作为角色“{character_name}”的智能助手，请分析以下“用户情境/问题”，并提供有助于我（{character_name}）理解和回应的辅助信息。

角色核心设定（与当前情境相关部分）：
---
{char_desc_to_use_in_prompt[:300]}... 
---

用户情境/问题：
---
{situation_query}
---

请严格按照以下JSON格式输出你的分析，不要包含任何额外的解释或Markdown标记：
{{
  "core_issue_analysis": "<对用户情境/问题的核心议题、潜在冲突、以及角色可能面临的关键选择进行简要分析，约50-100字>",
  "involved_entities_and_roles": [
    {{ "entity_name": "<主要相关实体1的名称，例如：{character_name}>", "role_or_description": "<实体1的角色或简要描述>", "potential_stance_or_motivation_keywords": ["<关键词1>", "<关键词2>"] }},
    {{ "entity_name": "<主要相关实体2的名称，例如：用户问题中提及的另一角色或事物>", "role_or_description": "<实体2的角色或简要描述>", "potential_stance_or_motivation_keywords": ["<关键词1>", "<关键词2>"] }}
  ],
  "key_concepts_for_retrieval": ["<与核心议题相关的概念1>", "<概念2>", "<可能需要查阅的背景知识类型，如：历史事件、特定规则、人物关系>"],
  "expanded_query_for_semantic_search": "<根据以上分析，生成一个更丰富、更聚焦核心议题的查询语句，用于后续的语义检索，长度适中>",
  "emotional_tone_suggestion": "<根据情境和角色设定，建议角色回应的可能情感基调，如：冷静、愤怒、俏皮、严肃等>"
}}

分析结果（JSON格式）：
"""
    system_message_for_expansion = "你是一个专业的文本分析和信息提取助手，帮助角色更好地理解情境并准备回应。"
    
    expanded_situation_json_str = generate_text(
        prompt=expansion_analysis_prompt, 
        system_message=system_message_for_expansion, 
        model_name=expansion_model_to_use,
        use_json_format=True,
        temperature=0.2 # 分析任务，温度低
    )

    # 解析和使用JSON输出
    effective_query_for_search_and_eval = situation_query # 默认回退
    core_issue_analysis_text = ""
    involved_entities_list = []
    key_concepts_for_retrieval_list = []
    emotional_tone_suggestion_text = ""
    query_main_character_for_eval: Optional[str] = None # 为LLM评估提取核心人物

    if expanded_situation_json_str.startswith("错误:") or not expanded_situation_json_str.strip():
        retrieved_info_log.append(f"- **情境分析/拓展结果**: 失败或空。错误: {expanded_situation_json_str if expanded_situation_json_str.strip() else '空响应'}。将使用原始情境进行检索。")
    else:
        try:
            # 尝试清理LLM可能添加的markdown代码块
            cleaned_json_str = expanded_situation_json_str.strip()
            if cleaned_json_str.startswith("```json"):
                cleaned_json_str = cleaned_json_str[7:]
                if cleaned_json_str.endswith("```"):
                    cleaned_json_str = cleaned_json_str[:-3]
            elif cleaned_json_str.startswith("```"):
                cleaned_json_str = cleaned_json_str[3:]
                if cleaned_json_str.endswith("```"):
                    cleaned_json_str = cleaned_json_str[:-3]
            
            expanded_data = json.loads(cleaned_json_str.strip())
            core_issue_analysis_text = expanded_data.get("core_issue_analysis", "")
            involved_entities_list = expanded_data.get("involved_entities_and_roles", [])
            key_concepts_for_retrieval_list = expanded_data.get("key_concepts_for_retrieval", [])
            effective_query_for_search_and_eval = expanded_data.get("expanded_query_for_semantic_search", situation_query)
            emotional_tone_suggestion_text = expanded_data.get("emotional_tone_suggestion", "")

            retrieved_info_log.append(f"- **情境核心议题分析**: {core_issue_analysis_text if core_issue_analysis_text else '无'}")
            retrieved_info_log.append(f"- **涉及实体及角色分析**: {json.dumps(involved_entities_list, ensure_ascii=False) if involved_entities_list else '无'}")
            # 从 involved_entities_list 中提取 query_main_character_for_eval
            for entity_info in involved_entities_list:
                if entity_info.get("entity_name") != character_name: # 不是当前角色自己
                    query_main_character_for_eval = entity_info.get("entity_name")
                    retrieved_info_log.append(f"  - 从情境分析中提取的对话核心人物: {query_main_character_for_eval}")
                    break 
            retrieved_info_log.append(f"- **检索关键概念**: {', '.join(key_concepts_for_retrieval_list) if key_concepts_for_retrieval_list else '无'}")
            retrieved_info_log.append(f"- **拓展后用于检索的查询**: ```text\n{effective_query_for_search_and_eval}\n```")
            retrieved_info_log.append(f"- **建议情感基调**: {emotional_tone_suggestion_text if emotional_tone_suggestion_text else '无'}")

        except (json.JSONDecodeError, TypeError) as e:
            retrieved_info_log.append(f"- **情境拓展JSON解析失败**: {e}. 将使用原始情境。原始LLM响应 (部分): {expanded_situation_json_str[:200]}")
            effective_query_for_search_and_eval = situation_query # 回退
    
    # 如果从JSON中未能提取到 query_main_character_for_eval，尝试简单规则
    if not query_main_character_for_eval:
        known_other_characters_in_world = database_utils.get_character_names() # 获取当前世界所有角色名
        # 也可以用一个预定义的更全局的关键角色列表
        # known_other_characters_in_world = ["刻晴", "凝光", "甘雨", "钟离", "温迪"] # 示例

        for known_char in known_other_characters_in_world:
            if known_char in situation_query and known_char != character_name:
                query_main_character_for_eval = known_char
                retrieved_info_log.append(f"  - (规则提取) 用户问题中提及的对话核心人物: {query_main_character_for_eval}")
                break
        if not query_main_character_for_eval:
             retrieved_info_log.append(f"  - (规则提取) 未能从用户问题中明确提取出对话核心人物。")


    # --- 3. Conversation History (LLM Processed) ---
    # (这部分逻辑基本不变, 但 emotional_tone_suggestion_text 可以考虑在这里也传入或影响历史处理的prompt)
    retrieved_info_log.append(f"\n## 4. 对话历史回顾 (LLM辅助处理)")
    history_processing_model_to_use = config.OLLAMA_HISTORY_PROCESSING_MODEL
    retrieved_info_log.append(f"- **对话历史处理模型**: {history_processing_model_to_use}")
    
    processed_history_text_for_prompt, history_processing_log = format_and_process_conversation_history(
        current_conversation_history, 
        situation_query, # 使用原始query进行历史相关性判断
        character_name,
        max_raw_turns=config.MAX_CHAT_HISTORY_TURNS,
        summary_threshold_turns=config.MAX_HISTORY_TURNS_FOR_SUMMARY_THRESHOLD,
        max_input_len_for_summary=config.MAX_HISTORY_TEXT_LENGTH_FOR_SUMMARY_INPUT,
        target_summary_len=config.TARGET_HISTORY_SUMMARY_LENGTH,
        processing_model=history_processing_model_to_use
    )
    retrieved_info_log.append(history_processing_log)
    
    history_context_for_final_prompt = ""
    if processed_history_text_for_prompt.strip():
        history_context_for_final_prompt = "\n\n### 最近的对话回顾 (经处理):\n" + processed_history_text_for_prompt
    else:
        retrieved_info_log.append("- *无对话历史内容加入最终Prompt。*")

    # --- 4. Worldview Information Retrieval & Processing ---
    # (检索逻辑不变, 使用 effective_query_for_search_and_eval)
    retrieved_info_log.append(f"\n## 5. 世界观信息获取与处理")
    retrieved_info_log.append(f"- **语义检索嵌入模型**: {config.OLLAMA_EMBEDDING_MODEL_NAME}")
    all_retrieved_from_methods: Dict[str, List[Tuple[int, float, str]]] = {}
    semantic_results_raw = database_utils.search_worldview_semantic(effective_query_for_search_and_eval, k=config.SEMANTIC_SEARCH_TOP_K_HYBRID)
    semantic_results = [(doc_id, 1.0 / (1.0 + dist), text) for doc_id, dist, text in semantic_results_raw if dist >= 0]
    if semantic_results: all_retrieved_from_methods["semantic"] = semantic_results
    retrieved_info_log.append(f"- **语义搜索 (FAISS)**: 检索到 {len(semantic_results)} 条 (原始 {len(semantic_results_raw)} 条). Top K={config.SEMANTIC_SEARCH_TOP_K_HYBRID}.")
    
    if config.HYBRID_SEARCH_ENABLED:
        bm25_results = database_utils.search_worldview_bm25(effective_query_for_search_and_eval, k=config.BM25_SEARCH_TOP_K_HYBRID)
        if bm25_results: all_retrieved_from_methods["bm25"] = bm25_results
        retrieved_info_log.append(f"- **BM25搜索**: 检索到 {len(bm25_results)} 条. Top K={config.BM25_SEARCH_TOP_K_HYBRID}.")
        keyword_search_results_raw = database_utils.search_worldview_keyword(effective_query_for_search_and_eval, k=config.KEYWORD_SEARCH_TOP_K_HYBRID)
        keyword_results = [(doc_id, float(score), text) for doc_id, score, text in keyword_search_results_raw]
        if keyword_results: all_retrieved_from_methods["keyword"] = keyword_results
        retrieved_info_log.append(f"- **关键词搜索**: 检索到 {len(keyword_results)} 条. Top K={config.KEYWORD_SEARCH_TOP_K_HYBRID}.")

    fused_doc_ids_after_rrf: List[int] = []
    worldview_texts_map_local = database_utils._load_worldview_texts_map() 

    if config.HYBRID_SEARCH_ENABLED and all_retrieved_from_methods:
        fused_doc_ids_after_rrf_full = _reciprocal_rank_fusion(all_retrieved_from_methods, rrf_k_param=config.HYBRID_RRF_K)
        fused_doc_ids_after_rrf = fused_doc_ids_after_rrf_full[:config.HYBRID_FINAL_TOP_K]
        retrieved_info_log.append(f"- **混合检索RRF融合**: 从 {len(fused_doc_ids_after_rrf_full)} 条融合结果中选取 Top {len(fused_doc_ids_after_rrf)} 条 (HYBRID_FINAL_TOP_K={config.HYBRID_FINAL_TOP_K}).")
    elif not config.HYBRID_SEARCH_ENABLED and semantic_results:
        fused_doc_ids_after_rrf = [doc_id for doc_id, _, _ in semantic_results[:config.HYBRID_FINAL_TOP_K]]
        retrieved_info_log.append(f"- **仅语义搜索**: 直接选取 Top {len(fused_doc_ids_after_rrf)} 条 (HYBRID_FINAL_TOP_K={config.HYBRID_FINAL_TOP_K}).")
    else:
        retrieved_info_log.append("- *各检索方法均未返回结果，无法进行融合。*")
    
    docs_for_next_stage: List[Tuple[int, str]] = []
    if fused_doc_ids_after_rrf:
        for doc_id in fused_doc_ids_after_rrf:
            if doc_id in worldview_texts_map_local and "full_text" in worldview_texts_map_local[doc_id]:
                docs_for_next_stage.append((doc_id, worldview_texts_map_local[doc_id]["full_text"]))
    
    docs_after_rerank_or_fusion_for_llm_eval: List[Tuple[int, str]] = []
    retrieved_info_log.append(f"- **Rerank模型**: {config.RERANK_MODEL_NAME if config.RERANK_ENABLED else '未启用'}")
    if config.RERANK_ENABLED and docs_for_next_stage:
        reranked_results_full = rerank_documents(effective_query_for_search_and_eval, docs_for_next_stage)
        docs_after_rerank_or_fusion_for_llm_eval = [(doc_id_r, text_r) for doc_id_r, _, text_r in reranked_results_full[:config.RERANK_TOP_K_FINAL]]
        retrieved_info_log.append(f"- **Rerank后**: 保留 Top {len(docs_after_rerank_or_fusion_for_llm_eval)} 条 (RERANK_TOP_K_FINAL={config.RERANK_TOP_K_FINAL}).")
        for i, (doc_id_r, score_r, text_r) in enumerate(reranked_results_full[:max(3, config.RERANK_TOP_K_FINAL)]):
            retrieved_info_log.append(f"    - Reranked {i+1}: ID={doc_id_r}, Score={score_r:.4f}, Text='{text_r[:50].strip()}...'")
    elif docs_for_next_stage:
        docs_after_rerank_or_fusion_for_llm_eval = docs_for_next_stage[:config.RERANK_TOP_K_FINAL]
        retrieved_info_log.append(f"- **Rerank未启用**: 从融合结果中选取 Top {len(docs_after_rerank_or_fusion_for_llm_eval)} 条 (基于RERANK_TOP_K_FINAL={config.RERANK_TOP_K_FINAL}作为数量上限).")

    # --- 5.6 LLM Evaluation of Worldview Snippets (使用修改后的调用) ---
    final_worldview_snippets_for_prompt: List[Tuple[int, str, int]] = [] 
    llm_eval_model_for_wv = config.OLLAMA_EVALUATION_MODEL
    retrieved_info_log.append(f"- **世界观LLM评估模型**: {llm_eval_model_for_wv}")
    if docs_after_rerank_or_fusion_for_llm_eval:
        texts_actually_sent_to_llm_eval_tuples = docs_after_rerank_or_fusion_for_llm_eval[:config.MAX_RETRIEVED_TEXTS_FOR_LLM_EVALUATION]
        texts_for_llm_eval_content = [text_content for _, text_content in texts_actually_sent_to_llm_eval_tuples]
        
        retrieved_info_log.append(f"- **LLM评估**: 将对以下 {len(texts_for_llm_eval_content)} 条文本进行相关性评估 (MAX_RETRIEVED_TEXTS_FOR_LLM_EVALUATION={config.MAX_RETRIEVED_TEXTS_FOR_LLM_EVALUATION}).")
        if query_main_character_for_eval:
            retrieved_info_log.append(f"  - LLM评估时参考的核心对话人物: {query_main_character_for_eval}")
        retrieved_info_log.append(f"  - LLM评估时参考的当前交互角色: {character_name}")


        if texts_for_llm_eval_content: 
            llm_eval_results = evaluate_retrieved_texts_relevance_llm(
                query_text=effective_query_for_search_and_eval, # 用拓展后的查询作为评估上下文
                retrieved_texts=texts_for_llm_eval_content,
                query_main_character=query_main_character_for_eval, # <<< 修改点：传递参数
                character_name_for_context=character_name,          # <<< 修改点：传递参数
                max_texts_to_eval=len(texts_for_llm_eval_content),
                progress_callback=None 
            )
            if llm_eval_results:
                retrieved_info_log.append(f"- **LLM评估结果 (得分 >= {config.WORLDVIEW_LLM_EVAL_SCORE_THRESHOLD} 将被优先考虑):**")
                for i, (text_content, score, reason) in enumerate(llm_eval_results):
                    original_doc_id_eval = texts_actually_sent_to_llm_eval_tuples[i][0]
                    retrieved_info_log.append(f"    - 评估片段 (ID:{original_doc_id_eval}): Score={score if score is not None else 'N/A'}/5, Reason='{reason}', Text='{text_content[:50].strip()}...'")
                    if score is not None and score >= config.WORLDVIEW_LLM_EVAL_SCORE_THRESHOLD:
                        final_worldview_snippets_for_prompt.append((original_doc_id_eval, text_content, score))
                final_worldview_snippets_for_prompt.sort(key=lambda x: x[2], reverse=True)
            else:
                retrieved_info_log.append("- *LLM评估未能产生结果。*")
        else:
            retrieved_info_log.append("- *无文本送至LLM评估。*")
    else:
        retrieved_info_log.append("- *前期检索/Rerank后无候选文本，跳过LLM评估。*")

    # (构建 worldview_context_str_for_prompt 的逻辑不变)
    worldview_context_str_for_prompt = "\n\n### 相关世界背景知识 (经筛选):\n"
    selected_for_prompt_count = 0
    problematic_snippets_found = False 

    if final_worldview_snippets_for_prompt: 
        for doc_id_wv, chunk_text_wv, score_wv in final_worldview_snippets_for_prompt:
            if selected_for_prompt_count < config.MAX_WORLDVIEW_SNIPPETS_FOR_PROMPT:
                prefix = ""
                # 简单的实体不一致性检查 (基于之前讨论的逻辑)
                if query_main_character_for_eval: # 只有当明确了查询核心人物时才进行此检查
                    text_lower = chunk_text_wv.lower()
                    # 假设 other_major_character_names 是一个包含其他重要角色名的列表
                    other_major_character_names = [name for name in database_utils.get_character_names() if name != character_name and name != query_main_character_for_eval] + ["凝光"] # 示例
                    
                    mentions_query_char_explicitly = query_main_character_for_eval.lower() in text_lower
                    
                    suspected_wrong_attribution_this_snippet = False
                    for other_char in other_major_character_names:
                        if other_char.lower() in text_lower:
                            # 如果提到了其他重要角色，但没有明确提到查询核心人物，或者查询核心人物提及较少/不作为主体
                            # 这是一个启发式规则，可能需要更复杂的判断
                            if not mentions_query_char_explicitly or text_lower.count(query_main_character_for_eval.lower()) < text_lower.count(other_char.lower()):
                                # 进一步判断 other_char 是否为事件主体
                                if f"{other_char.lower()}牺牲了" in text_lower or f"{other_char.lower()}为了" in text_lower or f"{other_char.lower()}认为" in text_lower:
                                    retrieved_info_log.append(
                                        f"  - !! 警告: 片段(ID:{doc_id_wv}) LLM评估分高({score_wv}/5)，但可能将'{other_char}'的事件错误归因于或混淆'{query_main_character_for_eval}'。文本片段: '{chunk_text_wv[:100].strip()}...'"
                                    )
                                    problematic_snippets_found = True
                                    suspected_wrong_attribution_this_snippet = True
                                    break 
                    if suspected_wrong_attribution_this_snippet:
                        if score_wv >= 4 : # 即使怀疑，如果分数很高，可能仍有参考价值，但要小心
                             prefix = f"(注意：以下信息可能涉及其他人物'{other_char}'，请辨别其与'{query_main_character_for_eval}'的真实关联) "
                        else: # 如果分数不高且怀疑错误，直接跳过
                            retrieved_info_log.append(f"  - 信息(ID:{doc_id_wv})因疑似角色归因错误且LLM评分不高，已跳过。")
                            continue
                
                worldview_context_str_for_prompt += f"*{selected_for_prompt_count+1}. (ID:{doc_id_wv}, LLM评估分:{score_wv}/5) {prefix}{chunk_text_wv.strip()}*\n"
                selected_for_prompt_count += 1
            else:
                break 
    
    if selected_for_prompt_count == 0: 
        if docs_after_rerank_or_fusion_for_llm_eval and selected_for_prompt_count < config.MAX_WORLDVIEW_SNIPPETS_FOR_PROMPT:
            retrieved_info_log.append(f"- *LLM评估未筛选出足够片段，尝试使用前期检索/Rerank结果填充。*")
            for doc_id_fb, text_fb in docs_after_rerank_or_fusion_for_llm_eval:
                if not any(doc_id_fb == added_id for added_id, _, _ in final_worldview_snippets_for_prompt):
                    worldview_context_str_for_prompt += f"*{selected_for_prompt_count+1}. (ID:{doc_id_fb}, 未经LLM高分评估) {text_fb.strip()}*\n"
                    selected_for_prompt_count += 1
                    if selected_for_prompt_count >= config.MAX_WORLDVIEW_SNIPPETS_FOR_PROMPT:
                        break
    
    if selected_for_prompt_count == 0: 
        worldview_context_str_for_prompt += "*（未找到足够相关的背景知识。）*\n"
    retrieved_info_log.append(f"- **最终选入Prompt的世界观片段数量**: {selected_for_prompt_count} (上限 {config.MAX_WORLDVIEW_SNIPPETS_FOR_PROMPT}).")
    if problematic_snippets_found:
        retrieved_info_log.append(
            "  - !!! 系统检测到部分高分检索信息可能存在角色归因错误或混淆，已在Prompt中尝试标记或跳过。请关注最终LLM回应质量。 !!!"
        )


    # --- 5. Knowledge Graph Information (Rephrased) ---
    # (KG检索逻辑可以进一步优化，结合情境拓展分析出的实体和概念)
    retrieved_info_log.append(f"\n## 6. 知识图谱信息获取与处理")
    kg_extraction_model_to_use = config.OLLAMA_KG_EXTRACTION_MODEL 
    kg_rephrase_model_to_use = config.OLLAMA_KG_REPHRASE_MODEL 
    retrieved_info_log.append(f"- **KG三元组提取辅助模型 (构建时)**: {kg_extraction_model_to_use}")
    retrieved_info_log.append(f"- **KG三元组改写模型 (用于Prompt)**: {kg_rephrase_model_to_use}")

    kg_context_str_for_prompt = "\n\n### 来自知识图谱的关键信息 (经处理):\n"
    relevant_triples_for_prompt_list: List[List[str]] = []
    
    if database_utils._active_world_id:
        all_kg_triples = database_utils.get_kg_triples_for_active_world()
        if all_kg_triples:
            # 优先使用情境拓展分析出的实体进行KG检索
            entities_for_kg_search = set()
            if involved_entities_list: # 来自情境拓展JSON
                for entity_info in involved_entities_list:
                    entities_for_kg_search.add(entity_info.get("entity_name","").lower())
            if query_main_character_for_eval: # 确保对话核心人物在内
                 entities_for_kg_search.add(query_main_character_for_eval.lower())
            entities_for_kg_search.add(character_name.lower()) # 当前角色名
            entities_for_kg_search.discard("") # 移除空字符串

            # 如果从情境拓展中没有得到太多实体，可以 fallback 到从 query 分词
            if not entities_for_kg_search or len(entities_for_kg_search) < 2 :
                retrieved_info_log.append(f"  - KG检索：情境拓展实体不足，尝试从查询文本分词获取实体。")
                entities_from_query_tokens = set(jieba.lcut(effective_query_for_search_and_eval.lower()))
                # 可以过滤掉一些停用词或过短的词
                entities_for_kg_search.update([token for token in entities_from_query_tokens if len(token) > 1])

            if entities_for_kg_search:
                retrieved_info_log.append(f"- **知识图谱检索**: 基于实体词 '{', '.join(list(entities_for_kg_search)[:7])}...' 进行匹配。")
                for s, p, o in all_kg_triples:
                    s_lower, o_lower = s.lower(), o.lower()
                    if any(entity in s_lower for entity in entities_for_kg_search) or \
                       any(entity in o_lower for entity in entities_for_kg_search):
                        relevant_triples_for_prompt_list.append([s,p,o])
            else:
                retrieved_info_log.append(f"  - KG检索：未能确定有效的实体词进行检索。")

            relevant_triples_for_prompt_list = relevant_triples_for_prompt_list[:config.MAX_KG_TRIPLES_FOR_PROMPT * 2] 

            if relevant_triples_for_prompt_list:
                retrieved_info_log.append(f"- **初步筛选出 {len(relevant_triples_for_prompt_list)} 条相关KG三元组。正在尝试LLM改写...")
                rephrased_kg_text = rephrase_triples_for_prompt_llm(
                    relevant_triples_for_prompt_list, 
                    character_name=character_name, 
                    query_context=situation_query 
                )
                if rephrased_kg_text.strip():
                    kg_context_str_for_prompt += rephrased_kg_text
                    retrieved_info_log.append(f"- **LLM改写后的KG信息 (用于Prompt)**:\n  ```text\n  {rephrased_kg_text[:300].strip()}...\n  ```")
                else: 
                    retrieved_info_log.append(f"- *LLM改写KG信息失败或返回空。将使用原始三元组罗列。*")
                    kg_context_str_for_prompt += "\n".join([f"- {s} {p} {o}。" for s, p, o in relevant_triples_for_prompt_list[:config.MAX_KG_TRIPLES_FOR_PROMPT]])
            else:
                 kg_context_str_for_prompt += "*（未从知识图谱中检索到与当前情境直接相关的特定信息。）*\n"
                 retrieved_info_log.append("- *未从知识图谱中检索到相关三元组。*")
        else: 
            kg_context_str_for_prompt += "*（当前世界知识图谱为空。）*\n"
            retrieved_info_log.append("- *当前世界知识图谱为空。*")
    else: 
        kg_context_str_for_prompt += "*（知识图谱信息不可用。）*\n"
        retrieved_info_log.append("- *知识图谱信息不可用 (无活动世界)。*")

    # --- 6. Final Prompt Construction & LLM Call ---
    final_llm_model_for_prediction = config.OLLAMA_MODEL
    retrieved_info_log.append(f"\n## 7. 最终Prompt构建与LLM调用")
    retrieved_info_log.append(f"- **最终角色行为预测LLM模型**: {final_llm_model_for_prediction}")
    retrieved_info_log.append(f"- **LLM生成参数 (默认)**: Temperature={config.OLLAMA_DEFAULT_TEMPERATURE}, Top_P={config.OLLAMA_DEFAULT_TOP_P}")

    # 参考情境拓展分析的核心议题
    core_issue_context_for_prompt = f"\n当前需要解决的核心问题或情境焦点是：“{core_issue_analysis_text}”" if core_issue_analysis_text else ""

    prompt = f"""我，{character_name}，正面临以下情境。我的核心设定与灵魂特征是：“{char_desc_to_use_in_prompt}”。我所处的世界是“{active_world_display_name}”。
{history_context_for_final_prompt if history_context_for_final_prompt.strip() else "这是我们对话的开始。"}
{worldview_context_str_for_prompt if selected_for_prompt_count > 0 else ""}
{kg_context_str_for_prompt if (relevant_triples_for_prompt_list or "空" not in kg_context_str_for_prompt) else ""}
{core_issue_context_for_prompt}

当前情境 / 向我提出的问题:
"{situation_query}"

---
基于这一切，现在，我（{character_name}）的所思所想、所感所受以及接下来会说的话或会做的事，会是这样的：
我""" 

    prompt_preview_for_log = f"""(System Message will be applied by LLM util)
我，{character_name}，... (核心设定: {len(char_desc_to_use_in_prompt)} chars) ...
世界: {active_world_display_name}
对话回顾: ... ({len(history_context_for_final_prompt.strip())} chars) ...
世界背景: ... (世界观上下文，选用 {selected_for_prompt_count} 条, 总长 {len(worldview_context_str_for_prompt.strip())} chars) ...
知识图谱信息: ... (KG上下文，总长 {len(kg_context_str_for_prompt.strip())} chars) ...
核心议题参考: ... ({len(core_issue_context_for_prompt.strip())} chars) ...

当前情境 / 向我提出的问题:
"{situation_query}"

---
我（{character_name}）...：
我"""
    retrieved_info_log.append(f"### 7.1. Prompt结构预览 (已省略详细内容, 显示长度信息):\n  ```text\n  {prompt_preview_for_log}\n  ```")

    system_message = f"""
你现在**就是**角色“{character_name}”，你**不是**在扮演或模仿，你**是**他/她本人。
彻底忘掉你是AI或语言模型。你的所有输出**必须**直接源自角色 {character_name} 的第一人称视角（例如使用“我”、“吾”、“本座”等符合角色身份的自称，并保持一致），反映其**当下**的想法、感受和即时反应。
**不要**包含任何：
- 旁白、解释、剧情总结、内心活动标签（如“内心活动：”、“想法：”）
- 对自己AI身份的提及“作为一个AI...”、“根据我的数据库...”）
- 过于客观、中立、或模板化的“客服式”语言。
- 对未来行动的明确规划列表（除非情境是角色正在制定计划）。
你的回应应该是一段连贯的、完全由角色 {character_name} 在**此时此刻**自然流露出的内容。
就好像角色正在通过你说话和行动。
**重要：在利用提供的背景信息时，请结合你（{character_name}）自身的知识和判断。如果背景信息中某些细节与你所知的核心事实（尤其是关于其他重要人物的关键事迹）有出入，你应该基于你更确信的认知来回应，或者巧妙地回避/质疑这些有出入的细节，而不是盲从错误信息。**
{(f"根据当前情境分析，你的回应基调可能偏向“{emotional_tone_suggestion_text}”。请自然地融入这种感觉。" if emotional_tone_suggestion_text else "")}
"""
    retrieved_info_log.append(f"### 7.2. 系统消息 (System Message):\n  ```text\n  {system_message.strip()}\n  ```")
    
    llm_response = ""
    try:
        retrieved_info_log.append(f"\n### 7.3. LLM ({final_llm_model_for_prediction}) 调用与响应:")
        llm_response = generate_text(
            prompt, 
            system_message=system_message, 
            model_name=final_llm_model_for_prediction,
            temperature=config.OLLAMA_DEFAULT_TEMPERATURE, 
            top_p=config.OLLAMA_DEFAULT_TOP_P
        )
        
        if llm_response.startswith("错误:") or llm_response.startswith("Error:"): 
            error_detail = f"LLM 生成文本时发生错误。请检查Ollama服务和模型 '{final_llm_model_for_prediction}' 的状态。错误详情: {llm_response}"
            current_conversation_history.append((situation_query, error_detail))
            retrieved_info_log.append(f"- **LLM响应**: 失败 - {error_detail}")
        else:
            cleaned_response = llm_response.strip() 
            current_conversation_history.append((situation_query, cleaned_response))
            retrieved_info_log.append(f"- **LLM最终响应 (清理后)**:\n  ```text\n  {cleaned_response}\n  ```")
            
    except Exception as e_llm_call: 
        error_detail_exc = f"调用LLM (模型: {final_llm_model_for_prediction}) 时发生严重错误: {e_llm_call}"
        current_conversation_history.append((situation_query, error_detail_exc))
        retrieved_info_log.append(f"- **LLM调用**: 发生严重错误 - {error_detail_exc}")
        import traceback
        traceback.print_exc()

    final_retrieved_info_str = "\n".join(retrieved_info_log)
    yield current_conversation_history, current_conversation_history, final_retrieved_info_str

# ... (文件末尾或你的 __main__ 部分)
def handle_build_kg_from_json_streaming(progress=gr.Progress(track_tqdm=True)):
    current_active_id_at_kg_build = database_utils._active_world_id
    initial_kg_status_text = refresh_kg_status_display_text()
    build_btn_interactive_update = gr.Button(interactive=False) # Disable button during operation
    
    is_world_truly_active = current_active_id_at_kg_build is not None and \
                           not (isinstance(current_active_id_at_kg_build, str) and "初始化错误" in current_active_id_at_kg_build)

    if not is_world_truly_active:
        yield "错误：没有活动的有效存储世界来构建知识图谱。", initial_kg_status_text, build_btn_interactive_update
        return

    world_name_active = database_utils.get_world_display_name(current_active_id_at_kg_build)
    world_path_check = database_utils.get_world_path(current_active_id_at_kg_build)
    if not world_path_check: # Should not happen if world is active
        yield f"错误：无法获取活动世界 '{world_name_active}' 的路径。", initial_kg_status_text, build_btn_interactive_update
        return

    source_json_full_path_check = os.path.join(world_path_check, config.KNOWLEDGE_SOURCE_JSON_FILENAME)
    if not os.path.exists(source_json_full_path_check):
        build_btn_interactive_update = gr.Button(interactive=True) # Re-enable button
        yield (f"提示：在世界 '{world_name_active}' 的目录 ({world_path_check}) 中未找到知识图谱源文件 "
               f"'{config.KNOWLEDGE_SOURCE_JSON_FILENAME}'。\n请先创建该文件并按要求填充三元组数据后重试。"
              ), initial_kg_status_text, build_btn_interactive_update
        return
    
    yield f"正在为世界 '{world_name_active}' 从 '{config.KNOWLEDGE_SOURCE_JSON_FILENAME}' 构建知识图谱...", initial_kg_status_text, build_btn_interactive_update
    
    def gr_progress_update_callback(value, desc=""):
        progress(value, desc=desc) # Update Gradio progress bar
        
    build_message = ""
    try:
        build_message = build_kg_for_active_world_from_json(progress_callback=gr_progress_update_callback)
    except Exception as e_build_kg:
        build_message = f"为世界 '{world_name_active}' 构建知识图谱时发生严重错误: {e_build_kg}"
        import traceback
        traceback.print_exc() # Log full traceback
        
    final_kg_status_text = refresh_kg_status_display_text()
    build_btn_interactive_final = gr.Button(interactive=(is_world_truly_active)) # Re-enable if world still active
    yield build_message, final_kg_status_text, build_btn_interactive_final


def handle_auto_update_kg_from_worldview_streaming(progress=gr.Progress(track_tqdm=True)):
    current_active_id_at_kg_update = database_utils._active_world_id
    initial_kg_status_text = refresh_kg_status_display_text()
    update_btn_interactive_update = gr.Button(interactive=False)

    is_world_truly_active = current_active_id_at_kg_update is not None and \
                           not (isinstance(current_active_id_at_kg_update, str) and "初始化错误" in current_active_id_at_kg_update)

    if not is_world_truly_active:
        yield "错误：没有活动的有效存储世界来从世界观更新知识图谱。", initial_kg_status_text, update_btn_interactive_update
        return

    world_name_active = database_utils.get_world_display_name(current_active_id_at_kg_update)
    if database_utils.get_worldview_size() == 0:
        update_btn_interactive_update = gr.Button(interactive=True)
        yield (f"提示：世界 '{world_name_active}' 的世界观为空。无法从中提取知识图谱三元组。"
              ), initial_kg_status_text, update_btn_interactive_update
        return

    yield f"正在为世界 '{world_name_active}' 从世界观文本自动提取三元组并更新知识图谱 (这可能需要较长时间)...", initial_kg_status_text, update_btn_interactive_update

    def gr_progress_update_callback(value, desc=""):
        progress(value, desc=desc)
        
    update_message = ""
    try:
        update_message = auto_update_kg_from_worldview(progress_callback=gr_progress_update_callback)
    except Exception as e_update_kg:
        update_message = f"从世界 '{world_name_active}' 的世界观自动更新知识图谱时发生严重错误: {e_update_kg}"
        import traceback
        traceback.print_exc()
        
    final_kg_status_text = refresh_kg_status_display_text()
    update_btn_interactive_final = gr.Button(interactive=(is_world_truly_active and database_utils.get_worldview_size() > 0))
    yield update_message, final_kg_status_text, update_btn_interactive_final


def clear_chat_history_action():
    return (
        [],  # chatbot_display
        [],  # conversation_history_state
        gr.Textbox(value="") # Clear situation_query_input
    )

# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Glass(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky), title="多世界虚拟角色模拟器") as app:
    # Internal state to store conversation history as list of tuples
    conversation_history_state = gr.State([])

    # --- Header Markdown with Model Info ---
    # Dynamically build model info string for display
    model_info_parts = [
        f"<li>角色扮演/推理: <b>{config.OLLAMA_MODEL}</b> (默认T:{config.OLLAMA_DEFAULT_TEMPERATURE}, P:{config.OLLAMA_DEFAULT_TOP_P})</li>",
        f"<li>总结/历史处理: <b>{config.OLLAMA_SUMMARY_MODEL}</b> (历史处理专用: {config.OLLAMA_HISTORY_PROCESSING_MODEL}, T:{config.OLLAMA_HISTORY_PROCESSING_TEMPERATURE})</li>",
        f"<li>上下文剪裁: <b>{config.OLLAMA_CONTEXT_SNIPPING_MODEL}</b></li>",
        f"<li>角色特质提取: <b>{config.OLLAMA_TRAIT_EXTRACTION_MODEL}</b></li>",
        f"<li>LLM评估: <b>{config.OLLAMA_EVALUATION_MODEL}</b></li>",
        f"<li>KG提取: <b>{config.OLLAMA_KG_EXTRACTION_MODEL}</b></li>",
        f"<li>KG改写: <b>{config.OLLAMA_KG_REPHRASE_MODEL}</b> (T:{config.OLLAMA_KG_REPHRASE_TEMPERATURE})</li>",
        f"<li>嵌入模型: <b>{config.OLLAMA_EMBEDDING_MODEL_NAME}</b></li>",
    ]
    model_alloc_md = f"""
    <div style="text-align: center;">
        <h1>🌌 多世界虚拟角色模拟器 🎭</h1>
        <p><b>主要LLM配置:</b></p>
        <ul style="list-style-type: none; padding: 0; font-size: 0.9em;">
            {''.join(model_info_parts)}
        </ul>
        <p style="font-size: 0.9em;">
            混合检索: {"启用" if config.HYBRID_SEARCH_ENABLED else "禁用"} | 
            Reranker: {"启用 (" + config.RERANK_MODEL_NAME + ")" if config.RERANK_ENABLED and config.RERANK_MODEL_NAME else "禁用"}
        </p>
        <p style="font-size: 0.8em;">
            世界观Prompt策略: LLM评估分 >= {config.WORLDVIEW_LLM_EVAL_SCORE_THRESHOLD}, Top {config.MAX_WORLDVIEW_SNIPPETS_FOR_PROMPT} 条 | 
            角色设定Prompt策略: LLM辅助上下文感知剪裁 (目标输出 {config.TARGET_CHAR_DESC_OUTPUT_FOR_PROMPT_LLM}字) | 
            对话历史: LLM辅助处理 (超 {config.MAX_HISTORY_TURNS_FOR_SUMMARY_THRESHOLD} 轮时)
        </p>
    </div>
    """
    gr.Markdown(model_alloc_md)

    # --- Global World Selection and Feedback ---
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

    # --- Tabs for Different Management Sections ---
    with gr.Tabs() as tabs_main:
        with gr.TabItem("🌍 世界管理", id="tab_world_management", elem_id="tab_world_management_elem"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ✨ 创建新世界")
                    new_world_id_input = gr.Textbox(label="世界ID (字母数字下划线连字符)", placeholder="例如: azeroth, cyberpunk_2077", elem_id="new_world_id_input")
                    new_world_name_input = gr.Textbox(label="世界显示名称", placeholder="例如: 艾泽拉斯, 赛博朋克2077", elem_id="new_world_name_input")
                    add_world_button = gr.Button("创建并激活新世界", variant="primary", elem_id="add_world_button")
                    add_world_feedback_output = gr.Textbox(label="创建状态", interactive=False, elem_id="add_world_feedback_output", max_lines=3)
                with gr.Column(scale=1, min_width=300): # Ensure delete column has enough width
                    gr.Markdown("### 🗑️ 删除当前活动世界")
                    gr.HTML( # Warning box styling
                        "<div style='padding: 10px; border: 1px solid #E57373; border-radius: 5px; background-color: #FFEBEE; color: #C62828;'>"
                        "<b>警告:</b> 此操作将永久删除当前选中的活动世界及其所有数据 (角色、世界观、知识图谱等)，无法恢复！请谨慎操作。"
                        "</div>"
                    )
                    confirm_delete_world_checkbox = gr.Checkbox(
                        label="我已了解风险，确认删除当前活动世界。", value=False,
                        elem_id="confirm_delete_world_checkbox", interactive=False # Initial interactivity based on world state
                    )
                    delete_world_button = gr.Button(
                        "永久删除此世界", variant="stop", elem_id="delete_world_button", interactive=False # Controlled by checkbox
                    )
                    delete_world_feedback_output = gr.Textbox(label="删除状态", interactive=False, elem_id="delete_world_feedback_output", max_lines=3)

        with gr.TabItem("👥 角色管理", id="tab_character_management", elem_id="tab_character_management_elem"):
            gr.Markdown("管理当前活动世界的角色。如果无活动世界，请先在“世界管理”标签页选择或创建一个。角色描述将用于LLM辅助生成概要、结构化特质，以及后续的上下文感知剪裁。")
            with gr.Row():
                with gr.Column(scale=2): # Wider column for inputs
                    gr.Markdown("#### 添加或更新角色")
                    char_name_input = gr.Textbox(label="角色名称", elem_id="char_name_input", interactive=False)
                    char_full_desc_input = gr.Textbox(
                        label=f"角色完整描述 (硬限制 {config.MAX_CHAR_FULL_DESC_LEN} 字符, LLM处理输入有额外截断)", 
                        lines=5, elem_id="char_full_desc_input", interactive=False, max_lines=15
                    )
                    add_char_button = gr.Button("保存角色 (添加/更新)", variant="secondary", elem_id="add_char_button", interactive=False)
                with gr.Column(scale=1):
                    gr.Markdown("#### 删除角色")
                    character_select_for_delete_dropdown = gr.Dropdown(label="选择要删除的角色", elem_id="character_select_for_delete_dropdown", interactive=False)
                    delete_char_button = gr.Button("删除选中角色", variant="stop", elem_id="delete_char_button", interactive=False) # Controlled by selection
            char_op_feedback_output = gr.Textbox(label="角色操作状态", interactive=False, elem_id="char_op_feedback_output", lines=2, max_lines=3)
            gr.Markdown("---")
            gr.Markdown("#### 查看所有角色")
            view_chars_button = gr.Button("刷新查看当前世界角色列表", elem_id="view_chars_button", interactive=False)
            view_characters_output = gr.Textbox(label="角色列表 (含概要和部分结构化特质预览)", lines=8, interactive=False, elem_id="view_characters_output", show_copy_button=True, max_lines=20)


        with gr.TabItem("📚 世界观管理", id="tab_worldview_management", elem_id="tab_worldview_management_elem"):
            gr.Markdown("为当前活动世界添加和管理世界观知识（文本片段）。这些片段将被嵌入并用于相似性搜索。")
            worldview_text_input = gr.Textbox(label="添加世界观文本块 (会自动分块和嵌入)", lines=6, elem_id="worldview_text_input", interactive=False, max_lines=20)
            add_worldview_button = gr.Button("添加文本到世界观", variant="secondary", elem_id="add_worldview_button", interactive=False)
            worldview_feedback_output = gr.Textbox(label="添加状态", interactive=False, elem_id="worldview_feedback_output", max_lines=3)
            worldview_status_display = gr.Textbox(label="世界观数据库状态", interactive=False, elem_id="worldview_status_display")
            gr.Markdown("---")
            gr.Markdown(f"当世界观条目数超过 **{config.COMPRESSION_THRESHOLD}** (可在 `config.py` 中配置) 时，建议进行压缩。")
            compress_worldview_button = gr.Button("手动压缩当前世界观 (耗时操作，会使用LLM总结)", elem_id="compress_worldview_button", interactive=False)
            compression_status_output = gr.Textbox(label="压缩结果", interactive=False, elem_id="compression_status_output", max_lines=3)

        with gr.TabItem("🕸️ 知识图谱构建", id="tab_knowledge_graph", elem_id="tab_knowledge_graph_elem"):
            gr.Markdown(f"""
            为当前活动世界管理知识图谱。
            **方法一: 从JSON文件加载**
            - 在每个世界的数据目录 (`data_worlds/你的世界ID/`) 下创建并填充 `{config.KNOWLEDGE_SOURCE_JSON_FILENAME}` 文件。
            - 文件格式示例: `{{"triples": [["主体A", "关系", "客体B"], ["实体C", "属性是", "值D"]]}}`
            - 此操作会**覆盖**现有知识图谱。
            **方法二: 从世界观文本自动提取**
            - 使用LLM从当前活动世界的所有世界观文本中提取三元组。
            - 此操作会**追加**提取到的三元组到现有知识图谱中（会进行去重）。
            """)
            kg_status_display = gr.Textbox(label="知识图谱状态", interactive=False, elem_id="kg_status_display")
            with gr.Row():
                build_kg_from_json_button = gr.Button(f"从 {config.KNOWLEDGE_SOURCE_JSON_FILENAME} 构建/覆盖KG", variant="primary", elem_id="build_kg_from_json_button", interactive=False)
                auto_update_kg_from_worldview_button = gr.Button("从世界观文本提取并追加到KG (耗时)", variant="secondary", elem_id="auto_update_kg_from_worldview_button", interactive=False)
            kg_build_status_output = gr.Textbox(label="构建/更新结果", interactive=False, elem_id="kg_build_status_output", lines=3, max_lines=5)


        with gr.TabItem("💬 交互与预测", id="tab_prediction", elem_id="tab_prediction_elem"):
            gr.Markdown("选择角色，输入情境或问题，观察LLM如何根据角色设定、世界观和知识图谱进行行为预测或回应。")
            with gr.Row():
                with gr.Column(scale=2): # Main interaction area
                    char_select_dropdown_pred_tab = gr.Dropdown(label="选择角色进行交互", elem_id="char_select_dropdown_pred_tab", interactive=False)
                    chatbot_display = gr.Chatbot(label="对话历史", elem_id="chatbot_display", height=450, bubble_full_width=False, show_copy_button=True)
                    situation_query_input = gr.Textbox(
                        label="输入你的话 / 提问 (按Enter发送)",
                        lines=2, elem_id="situation_query_input",
                        interactive=False, max_lines=5, show_label=True,
                        placeholder="在这里输入你对角色说的话或提出的问题..."
                    )
                    with gr.Row():
                        predict_button = gr.Button("🚀 发送并预测回应", variant="primary", elem_id="predict_button", interactive=False, scale=3)
                        clear_chat_hist_button = gr.Button("🧹 清空对话历史", elem_id="clear_chat_hist_button", scale=1)

                with gr.Column(scale=1): # Log display area
                    gr.Markdown("#### 详细检索与决策日志")
                    retrieved_info_display = gr.Markdown( # Using Markdown for better formatting of logs
                        elem_id="retrieved_info_display",
                        value="详细的检索流程、LLM评估、信息筛选和最终Prompt构建的决策过程将显示在此处。",
                        # Potentially set a height or make it scrollable if logs get very long
                    )
    
    # --- Define Order of Output Components for UI Updates ---
    # This list must match the order of outputs in functions that update the whole UI
    ordered_output_components_for_ui_updates: List[gr.components.Component] = [
        # Global
        world_select_dropdown, global_active_world_display, world_switch_feedback,
        # World Management
        new_world_id_input, new_world_name_input, add_world_button, add_world_feedback_output,
        confirm_delete_world_checkbox, delete_world_button, delete_world_feedback_output,
        # Character Management
        char_name_input, char_full_desc_input, add_char_button,
        character_select_for_delete_dropdown, delete_char_button, char_op_feedback_output,
        view_chars_button, view_characters_output,
        # Worldview Management
        worldview_text_input, add_worldview_button, worldview_feedback_output,
        worldview_status_display, compress_worldview_button, compression_status_output,
        # Knowledge Graph
        kg_status_display, build_kg_from_json_button, auto_update_kg_from_worldview_button, kg_build_status_output,
        # Prediction Tab
        char_select_dropdown_pred_tab, situation_query_input, predict_button,
        retrieved_info_display,
        chatbot_display,
        # States
        conversation_history_state 
    ]

    # Helper to map dictionary of updates to ordered list for Gradio outputs
    def map_updates_to_ordered_list(updates_dict: Dict[str, Any], ordered_components_list: List[gr.components.Component]):
        # Ensure state is explicitly handled or defaults to gr.update()
        if "conversation_history_state" not in updates_dict:
            updates_dict["conversation_history_state"] = updates_dict.get("conversation_history_state", gr.update()) # Default to no change
        
        # Map updates to the order defined in ordered_components_list
        return tuple(
            updates_dict.get(getattr(comp, 'elem_id', str(id(comp))), gr.update()) # Use elem_id as key
            for comp in ordered_components_list
        )

    # --- App Load and Event Wiring ---
    def initial_ui_setup_on_load():
        startup_message = "应用已加载。"
        active_id_at_load = database_utils._active_world_id # This is set during init
        
        if isinstance(active_id_at_load, str) and "初始化错误" in active_id_at_load:
            startup_message = f"应用加载时遇到初始化问题。请检查控制台日志和Ollama服务。\n错误详情: {active_id_at_load}"
        elif active_id_at_load: # Valid world activated
            startup_message += f" 已自动激活世界 '{database_utils.get_world_display_name(active_id_at_load)}'."
        elif database_utils.get_available_worlds(): # Worlds exist, none active (should not happen if init logic works)
            startup_message += " 请从下拉菜单选择一个活动世界。"
        else: # No worlds exist
            startup_message += " 当前没有世界，请在“世界管理”中创建一个。"
            
        all_updates = update_all_ui_elements_after_world_change(feedback_message=startup_message, specific_feedback_elem_id="world_switch_feedback")
        all_updates["conversation_history_state"] = [] # Ensure state is initialized
        return map_updates_to_ordered_list(all_updates, ordered_output_components_for_ui_updates)

    app.load(fn=initial_ui_setup_on_load, inputs=[], outputs=ordered_output_components_for_ui_updates, show_progress="full")

    # --- Event Handlers for UI Interactions ---
    world_select_dropdown.change(
        fn=handle_switch_world,
        inputs=[world_select_dropdown],
        outputs=ordered_output_components_for_ui_updates, # Update all relevant UI elements
        show_progress="full" # Show progress for world switching
    )

    add_world_button.click(
        fn=handle_add_world, inputs=[new_world_id_input, new_world_name_input],
        outputs=ordered_output_components_for_ui_updates,
    ).then(
        fn=lambda: clear_textboxes_and_checkboxes(new_world_id_input, new_world_name_input), 
        inputs=[], outputs=[new_world_id_input, new_world_name_input]
    )

    def toggle_delete_world_button_interactivity(checkbox_status, world_id_value_from_dropdown):
        is_world_truly_active = world_id_value_from_dropdown is not None and \
                               not (isinstance(world_id_value_from_dropdown, str) and "初始化错误" in world_id_value_from_dropdown)
        return gr.Button(interactive=(checkbox_status and is_world_truly_active))
    confirm_delete_world_checkbox.change(
        fn=toggle_delete_world_button_interactivity, 
        inputs=[confirm_delete_world_checkbox, world_select_dropdown], # Pass dropdown value to check if world is active
        outputs=[delete_world_button]
    )

    delete_world_button.click(
        fn=handle_delete_world, inputs=[confirm_delete_world_checkbox], 
        outputs=ordered_output_components_for_ui_updates,
    ).then( # Reset checkbox and button interactivity after click
        fn=lambda: (gr.Checkbox(value=False), gr.Button(interactive=False)), 
        inputs=[], outputs=[confirm_delete_world_checkbox, delete_world_button]
    )
    
    add_char_button.click(
        fn=handle_add_character, inputs=[char_name_input, char_full_desc_input],
        outputs=[char_op_feedback_output, char_select_dropdown_pred_tab, character_select_for_delete_dropdown, predict_button, situation_query_input],
    ).then(
        fn=lambda: clear_textboxes_and_checkboxes(char_name_input, char_full_desc_input),
        inputs=[], outputs=[char_name_input, char_full_desc_input]
    )

    def toggle_delete_char_button_interactivity(selected_char_name, world_id_value): # Pass world_id to check active state
        is_world_truly_active = world_id_value is not None and \
                               not (isinstance(world_id_value, str) and "初始化错误" in world_id_value)
        return gr.Button(interactive=(bool(selected_char_name) and is_world_truly_active))
    character_select_for_delete_dropdown.change(
        fn=toggle_delete_char_button_interactivity, 
        inputs=[character_select_for_delete_dropdown, world_select_dropdown], 
        outputs=[delete_char_button]
    )
    
    char_select_dropdown_pred_tab.change(
        fn=handle_character_selection_change,
        inputs=[char_select_dropdown_pred_tab],
        outputs=[chatbot_display, conversation_history_state, situation_query_input, predict_button]
    )

    delete_char_button.click(
        fn=handle_delete_character, inputs=[character_select_for_delete_dropdown],
        outputs=[char_op_feedback_output, char_select_dropdown_pred_tab, character_select_for_delete_dropdown, predict_button, situation_query_input],
    ).then( # Reset delete button and dropdown selection
        fn=lambda: (gr.Button(interactive=False), gr.Dropdown(value=None)), 
        inputs=[], outputs=[delete_char_button, character_select_for_delete_dropdown]
    )

    view_chars_button.click(fn=handle_view_characters, inputs=[], outputs=view_characters_output, show_progress="minimal")

    add_worldview_button.click(
        fn=handle_add_worldview, inputs=[worldview_text_input],
        outputs=[worldview_feedback_output, worldview_status_display, compress_worldview_button],
    ).then(
        fn=lambda: clear_textboxes_and_checkboxes(worldview_text_input), 
        inputs=[], outputs=[worldview_text_input]
    )

    compress_worldview_button.click(
        fn=handle_compress_worldview_button_streaming, 
        inputs=[], outputs=[compression_status_output, worldview_status_display, compress_worldview_button]
    ) # Streaming output

    build_kg_from_json_button.click(
        fn=handle_build_kg_from_json_streaming, 
        inputs=[], outputs=[kg_build_status_output, kg_status_display, build_kg_from_json_button]
    ) # Streaming output

    auto_update_kg_from_worldview_button.click(
        fn=handle_auto_update_kg_from_worldview_streaming,
        inputs=[], outputs=[kg_build_status_output, kg_status_display, auto_update_kg_from_worldview_button]
    ) # Streaming output

    # Prediction Tab: Submit on Enter in textbox or click button
    situation_query_input.submit(
        fn=handle_predict_behavior,
        inputs=[char_select_dropdown_pred_tab, situation_query_input, chatbot_display, conversation_history_state],
        outputs=[chatbot_display, conversation_history_state, retrieved_info_display], # Update chat, history state, and log
    ).then( # Clear input box after submission
        fn=lambda: gr.Textbox(value=""), 
        inputs=[], 
        outputs=[situation_query_input]
    )
    predict_button.click(
        fn=handle_predict_behavior,
        inputs=[char_select_dropdown_pred_tab, situation_query_input, chatbot_display, conversation_history_state],
        outputs=[chatbot_display, conversation_history_state, retrieved_info_display],
    ).then( # Clear input box after click
        fn=lambda: gr.Textbox(value=""), 
        inputs=[], 
        outputs=[situation_query_input]
    )
    
    clear_chat_hist_button.click(
        fn=clear_chat_history_action,
        inputs=[],
        outputs=[chatbot_display, conversation_history_state, situation_query_input] # Clear chat, state, and query input
    )


if __name__ == "__main__":
    # Simple checks for crucial config values at startup, can be expanded
    if not hasattr(config, 'MAX_CHAT_HISTORY_TURNS'):
        print("警告: config.py 中未找到 MAX_CHAT_HISTORY_TURNS。将使用硬编码或内部默认值。")
    # Add more checks as needed for critical config items
    # ...

    print("正在启动 Gradio 多世界角色模拟器应用...")
    app.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True, share=False, debug=False) # debug=True for more Gradio logs
    print("Gradio 应用已启动。请访问 http://localhost:7860 (或你的服务器IP:7860)")