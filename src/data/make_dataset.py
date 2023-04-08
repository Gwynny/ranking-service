import itertools
import torch
import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from typing import Dict, List, Union, Callable, Tuple


class RankingDataset(torch.utils.data.Dataset):
    OUT_DICT = Dict[str, torch.LongTensor]

    def __init__(self,
                 index_pairs_or_triplets: List[List[Union[str, float]]],
                 idx_to_text_mapping: Dict[int, str],
                 vocab: Dict[str, int],
                 preproc_func: Callable,
                 oov_token: str = 'OOV',
                 max_len: int = 30):
        self.index_pairs_or_triplets = index_pairs_or_triplets
        self.idx_to_text_mapping = idx_to_text_mapping
        self.vocab = vocab
        self.preproc_func = preproc_func
        self.oov_val = self.vocab[oov_token]
        self.max_len = max_len

    @classmethod
    def collate_fn(cls,
                   batch_objs: List[Union[Dict[str, torch.Tensor], torch.FloatTensor]]
                   ) -> Union[Tuple[OUT_DICT, OUT_DICT, torch.FloatTensor],
                              Tuple[OUT_DICT, torch.FloatTensor]]:
        is_triplets = False if len(batch_objs[0]) == 2 else True

        q1s, d1s = [], []
        q2s, d2s = [], []
        labels = []
        for elem in batch_objs:
            if is_triplets:
                left_elem, right_elem, label = elem
            else:
                left_elem, label = elem

            q1s.append(left_elem['query'])
            d1s.append(left_elem['document'])
            if is_triplets:
                q2s.append(right_elem['query'])
                d2s.append(right_elem['document'])
            labels.append([float(label)])

        q1s, d1s = torch.LongTensor(q1s), torch.LongTensor(d1s)
        if is_triplets:
            q2s, d2s = torch.LongTensor(q2s), torch.LongTensor(d2s)
        labels = torch.FloatTensor(labels)

        ret_left = {'query': q1s, 'document': d1s}
        if is_triplets:
            ret_right = {'query': q2s, 'document': d2s}
            return ret_left, ret_right, labels
        else:
            return ret_left, labels

    @classmethod
    def get_question_groups(cls, inp_df: pd.DataFrame, min_group_size: int = 3) -> DataFrameGroupBy:
        inp_df_select = inp_df[['id_left', 'id_right', 'label']]
        inf_df_group_sizes = inp_df_select.groupby('id_left').size()
        leftids_to_use = list(
            inf_df_group_sizes[inf_df_group_sizes >= min_group_size].index
        )
        groups = inp_df_select[
                    inp_df_select['id_left'].isin(leftids_to_use)
                ].groupby('id_left')
        return groups

    def __len__(self):
        return len(self.index_pairs_or_triplets)

    def _tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        input_len = len(tokenized_text)
        if input_len > 30:
            text = tokenized_text[:self.max_len]
        else:
            text = tokenized_text + (self.max_len - input_len) * [self.vocab['PAD']]
        token_idxs = []
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
    def __getitem__(self, idx: int):
        triplets = self.index_pairs_or_triplets[idx]
        query_tokens = self._convert_text_idx_to_token_idxs(triplets[0])
        left_doc_tokens = self._convert_text_idx_to_token_idxs(triplets[1])
        right_doc_tokens = self._convert_text_idx_to_token_idxs(triplets[2])
        label = triplets[3]

        left_query_doc = {'query': query_tokens, 'document': left_doc_tokens}
        right_query_doc = {'query': query_tokens, 'document': right_doc_tokens}
        return left_query_doc, right_query_doc, label

    @classmethod
    def create_train_triplets(cls,
                              inp_df: pd.DataFrame,
                              seed: int,
                              num_positive_examples: int = 4,
                              num_random_positive_examples: int = 2,
                              num_same_rel_examples: int = 2
                              ) -> List[List[Union[str, float]]]:
        MIN_GROUP_SIZE = 3
        np.random.seed(seed)
        groups = RankingDataset.get_question_groups(inp_df, MIN_GROUP_SIZE)
        all_right_ids = inp_df.id_right.values
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

                pos_labels_permutations = [(one_id, zero_id) for one_id in ones_ids for zero_id in zeros_ids]
                np.random.shuffle(pos_labels_permutations)
                for ids in pos_labels_permutations[:num_positive_examples]:
                    out_triplets.append([id_left, ids[0], ids[1], 1.0])

                random_neg_sample = np.random.choice(all_right_ids, num_random_positive_examples, replace=False)
                pos_sample_ids = np.random.choice(ones_ids, num_random_positive_examples, replace=True)
                for i in range(len(random_neg_sample)):
                    out_triplets.append([id_left, pos_sample_ids[i], random_neg_sample[i], 1.0])

                zeros_permutations = list(itertools.combinations(zeros_ids, 2))
                np.random.shuffle(zeros_permutations)
                for ids in zeros_permutations[:num_same_rel_examples]:
                    out_triplets.append([id_left, ids[0], ids[1], 0.5])
        return out_triplets


class ValPairsDataset(RankingDataset):
    def __getitem__(self, idx: int):
        pairs = self.index_pairs_or_triplets[idx]
        query_tokens = self._convert_text_idx_to_token_idxs(pairs[0])
        doc_tokens = self._convert_text_idx_to_token_idxs(pairs[1])
        label = pairs[2]
        query_doc = {'query': query_tokens, 'document': doc_tokens}
        return query_doc, label

    @classmethod
    def create_val_pairs(cls,
                         inp_df: pd.DataFrame,
                         fill_top_to: int = 15,
                         seed: int = 0
                         ) -> List[List[Union[str, float]]]:
        MIN_GROUP_SIZE = 2
        np.random.seed(seed)
        groups = RankingDataset.get_question_groups(inp_df, MIN_GROUP_SIZE)

        all_ids = set(inp_df['id_left']).union(set(inp_df['id_right']))
        out_pairs = []
        for id_left, group in groups:
            ones_ids = group[group.label > 0].id_right.values
            zeroes_ids = group[group.label == 0].id_right.values
            sum_len = len(ones_ids) + len(zeroes_ids)
            num_pad_items = max(0, fill_top_to - sum_len)

            if num_pad_items > 0:
                cur_chosen = set(ones_ids).union(set(zeroes_ids)).union({id_left})
                pad_sample = np.random.choice(list(all_ids - cur_chosen), num_pad_items, replace=False).tolist()
            else:
                pad_sample = []

            for i in ones_ids:
                out_pairs.append([id_left, i, 2])
            for i in zeroes_ids:
                out_pairs.append([id_left, i, 1])
            for i in pad_sample:
                out_pairs.append([id_left, i, 0])
        return out_pairs
