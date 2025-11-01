import itertools
from pathlib import Path
from collections import defaultdict
from typing import List, Set, Tuple, Dict, Any

def _build_ground_truth(all_files: List[Path]) -> List[List[Path]]:
    gt_groups_dict = defaultdict(list)
    
    for file_path in all_files:
        parts = file_path.name.split('_')
        if len(parts) >= 2:
            group_id = parts[0]  
            gt_groups_dict[group_id].append(file_path)
        else:
            gt_groups_dict[file_path.name].append(file_path)
            
    return list(gt_groups_dict.values())

def _get_pairs(groups: List[List[Path]]) -> Set[Tuple[Path, Path]]:
    all_pairs = set()
    for group in groups:
        for file1, file2 in itertools.combinations(group, 2):
            canonical_pair = tuple(sorted((file1, file2)))
            all_pairs.add(canonical_pair)
    return all_pairs

def calculate_metrics(predicted_groups: List[List[Path]], 
                      all_processed_files: List[Path]) -> Dict[str, Any]:
    gt_groups = _build_ground_truth(all_processed_files)
    gt_pairs = _get_pairs(gt_groups)
    tp_plus_fn = len(gt_pairs) 

    pred_pairs = _get_pairs(predicted_groups)
    tp_plus_fp = len(pred_pairs)

    tp_pairs = gt_pairs.intersection(pred_pairs)
    tp = len(tp_pairs)

    precision = tp / tp_plus_fp if tp_plus_fp > 0 else 0.0
    recall = tp / tp_plus_fn if tp_plus_fn > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }