import ollama
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
            
    return all_embeddings