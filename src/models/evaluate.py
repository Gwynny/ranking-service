import math
import torch
import numpy as np
import pandas as pd


def dcg(ys_true: np.array, ys_pred: np.array, k: int) -> float:
    indices = np.argsort(-ys_pred)
    ys_true = ys_true[indices[:k]]

    sum_dcg = 0
    for i, y_true in enumerate(ys_true, 1):
        sum_dcg += (2 ** y_true - 1) / math.log2(i + 1)
    return sum_dcg


def ndcg_k(ys_true: np.array, ys_pred: np.array, ndcg_top_k: int = 10) -> float:
    ideal_dcg = dcg(ys_true, ys_true, ndcg_top_k)
    case_dcg = dcg(ys_true, ys_pred, ndcg_top_k)
    return float(case_dcg / ideal_dcg)


def check_ndcg_on_val_set(model: torch.nn.Module,
                          val_dataloader: torch.utils.data.DataLoader
                          ) -> float:
    labels_and_groups = val_dataloader.dataset.index_pairs_or_triplets
    labels_and_groups = pd.DataFrame(labels_and_groups,
                                     columns=['left_id', 'right_id', 'rel'])

    all_preds = []
    for batch in (val_dataloader):
        inp_1, y = batch
        preds = model.predict(inp_1)
        preds_np = preds.detach().numpy()
        all_preds.append(preds_np)
    all_preds = np.concatenate(all_preds, axis=0)
    labels_and_groups['preds'] = all_preds

    ndcgs = []
    for cur_id in labels_and_groups.left_id.unique():
        cur_df = labels_and_groups[labels_and_groups.left_id == cur_id]
        ndcg = ndcg_k(cur_df.rel.values.reshape(-1),
                      cur_df.preds.values.reshape(-1))
        if np.isnan(ndcg):
            ndcgs.append(0)
        else:
            ndcgs.append(ndcg)
    return np.mean(ndcgs)
