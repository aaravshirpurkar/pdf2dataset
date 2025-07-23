"""Ensemble extraction and voting system for document parsing."""
from typing import List, Dict, Any, Union
import difflib
from collections import Counter

def compare_text_similarity(text1: str, text2: str) -> float:
    """Compare similarity between two text strings."""
    if not text1 or not text2:
        return 0.0
    return difflib.SequenceMatcher(None, text1, text2).ratio()

def vote_on_table_structure(tables: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Vote on the most reliable table structure."""
    if not tables:
        return {}
    
    # Count column occurrences
    column_counts = Counter()
    for table in tables:
        if isinstance(table.get('content'), dict):
            columns = table['content'].get('columns', [])
            column_counts.update(columns)
    
    # Select most common columns
    final_columns = [col for col, count in column_counts.most_common()
                    if count >= len(tables) * 0.5]  # At least 50% agreement
    
    # Merge cell values
    merged_data = []
    for table in tables:
        if isinstance(table.get('content'), dict):
            data = table['content'].get('data', [])
            for row in data:
                merged_row = {col: row.get(col) for col in final_columns}
                if any(merged_row.values()):  # Only add if row has some values
                    merged_data.append(merged_row)
    
    return {
        'columns': final_columns,
        'data': merged_data
    }

def ensemble_voting(extracted_contents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine and vote on extracted content from multiple extractors."""
    if not extracted_contents:
        return {}
    
    # Group content by type
    texts = []
    tables = []
    metadata = []
    
    for content in extracted_contents:
        content_type = content.get('type', '').lower()
        if content_type == 'text':
            texts.append(content.get('content', ''))
        elif content_type == 'table':
            tables.append(content)
        elif content_type == 'metadata':
            metadata.append(content.get('metadata', {}))
    
    # Vote on text content
    final_text = ""
    if texts:
        # Compare text similarities and select most common/reliable version
        text_scores = {}
        for i, text1 in enumerate(texts):
            score = sum(compare_text_similarity(text1, text2) 
                       for j, text2 in enumerate(texts) if i != j)
            text_scores[text1] = score
        
        if text_scores:
            final_text = max(text_scores.items(), key=lambda x: x[1])[0]
    
    # Vote on table structure and content
    final_tables = vote_on_table_structure(tables) if tables else {}
    
    # Merge metadata
    final_metadata = {}
    if metadata:
        for field in set().union(*(m.keys() for m in metadata)):
            values = [m.get(field) for m in metadata if field in m]
            if values:
                # Use most common value, or first if tie
                final_metadata[field] = Counter(values).most_common(1)[0][0]
    
    return {
        'text': final_text,
        'tables': final_tables,
        'metadata': final_metadata
    }
