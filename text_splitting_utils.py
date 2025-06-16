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