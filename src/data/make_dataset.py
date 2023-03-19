import itertools
import torch
import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from typing import Dict, List, Union, Callable


def get_idx_to_text_mapping(inp_df: pd.DataFrame) -> Dict[str, str]:
    left_dict = (
        inp_df[
            ['id_left', 'text_left']
        ].drop_duplicates()
        .set_index('id_left')
        ['text_left'].to_dict()
    )
    right_dict = (
        inp_df[
            ['id_right', 'text_right']
        ].drop_duplicates()
        .set_index('id_right')
        ['text_right'].to_dict()
    )
    left_dict.update(right_dict)
    return left_dict


def get_question_groups(inp_df: pd.DataFrame,
                        min_group_size: int = 3) -> DataFrameGroupBy:
    inp_df_select = inp_df[['id_left', 'id_right', 'label']]
    inf_df_group_sizes = inp_df_select.groupby('id_left').size()
    leftids_to_use = list(
        inf_df_group_sizes[inf_df_group_sizes >= min_group_size].index
    )
    groups = inp_df_select[
                inp_df_select['id_left'].isin(leftids_to_use)
             ].groupby('id_left')
    return groups


def sample_data_for_train_iter(inp_df: pd.DataFrame,
                               seed: int,
                               min_group_size: int = 3,
                               num_positive_examples: int = 4,
                               num_same_rel_examples: int = 2
                               ) -> List[List[Union[str, float]]]:
    np.random.seed(seed)
    groups = get_question_groups(inp_df, min_group_size)
    out_triplets = []
    for id_left, group in groups:
        if group['label'].sum() == 0:
            continue
        ones_df = group[group['label'] == 1]
        zeros_df = group[group['label'] == 0]

        if len(zeros_df) > 1:
            ones_ids = ones_df['id_right'].to_list()
            np.random.shuffle(ones_ids)
            zeros_ids = zeros_df['id_right'].to_list()
            pos_labels_permutations = [
                (one_id, zero_id) for one_id in ones_ids for zero_id in zeros_ids]
            np.random.shuffle(pos_labels_permutations)
            for ids in pos_labels_permutations[:num_positive_examples]:
                out_triplets.append([id_left, ids[0], ids[1], 1.0])

            zeros_permutations = list(itertools.combinations(zeros_ids, 2))
            np.random.shuffle(zeros_permutations)
            for ids in zeros_permutations[:num_same_rel_examples]:
                out_triplets.append([id_left, ids[0], ids[1], 0.5])
    return out_triplets


def create_val_pairs(inp_df: pd.DataFrame,
                     fill_top_to: int = 15,
                     min_group_size: int = 2,
                     seed: int = 0
                     ) -> List[List[Union[str, float]]]:
    np.random.seed(seed)
    groups = get_question_groups(inp_df, min_group_size)

    all_ids = set(inp_df['id_left']).union(set(inp_df['id_right']))
    out_pairs = []
    for id_left, group in groups:
        ones_ids = group[group.label > 0].id_right.values
        zeroes_ids = group[group.label == 0].id_right.values
        sum_len = len(ones_ids) + len(zeroes_ids)
        num_pad_items = max(0, fill_top_to - sum_len)

        if num_pad_items > 0:
            cur_chosen = set(ones_ids).union(
                set(zeroes_ids)).union({id_left})
            pad_sample = np.random.choice(
                list(all_ids - cur_chosen), num_pad_items,
                replace=False).tolist()
        else:
            pad_sample = []

        for i in ones_ids:
            out_pairs.append([id_left, i, 2])
        for i in zeroes_ids:
            out_pairs.append([id_left, i, 1])
        for i in pad_sample:
            out_pairs.append([id_left, i, 0])
    return out_pairs


class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, index_pairs_or_triplets: List[List[Union[str, float]]],
                 idx_to_text_mapping: Dict[int, str],
                 vocab: Dict[str, int],
                 oov_val: int,
                 preproc_func: Callable,
                 max_len: int = 30):
        self.index_pairs_or_triplets = index_pairs_or_triplets
        self.idx_to_text_mapping = idx_to_text_mapping
        self.vocab = vocab
        self.oov_val = oov_val
        self.preproc_func = preproc_func
        self.max_len = max_len

    def __len__(self):
        return len(self.index_pairs_or_triplets)

    def _tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        token_idxs = []
        text = tokenized_text[:self.max_len]
        for token in text:
            token_idxs.append(self.vocab.get(token, self.oov_val))
        return token_idxs

    def _convert_text_idx_to_token_idxs(self, idx: int) -> List[int]:
        text = self.idx_to_text_mapping[idx]
        tokenized_text = self.preproc_func(text)
        token_idxs = self._tokenized_text_to_index(tokenized_text)
        return token_idxs

    def __getitem__(self, idx: int):
        pass


class TrainTripletsDataset(RankingDataset):
    def __getitem__(self, idx):
        triplets = self.index_pairs_or_triplets[idx]
        query_tokens = self._convert_text_idx_to_token_idxs(str(triplets[0]))
        left_doc_tokens = self._convert_text_idx_to_token_idxs(str(triplets[1]))
        right_doc_tokens = self._convert_text_idx_to_token_idxs(str(triplets[2]))
        label = triplets[3]

        left_query_doc = {'query': query_tokens, 'document': left_doc_tokens}
        right_query_doc = {'query': query_tokens, 'document': right_doc_tokens}
        return left_query_doc, right_query_doc, label


class ValPairsDataset(RankingDataset):
    def __getitem__(self, idx):
        pairs = self.index_pairs_or_triplets[idx]
        query_tokens = self._convert_text_idx_to_token_idxs(str(pairs[0]))
        doc_tokens = self._convert_text_idx_to_token_idxs(str(pairs[1]))
        label = pairs[2]
        query_doc = {'query': query_tokens, 'document': doc_tokens}
        return query_doc, label
