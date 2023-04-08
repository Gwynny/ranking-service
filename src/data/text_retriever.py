import itertools
import json
import nltk
import pickle
import string
import pandas as pd
from collections import Counter
from os.path import isfile
from typing import Dict, List


class TextRetriever:
    def __init__(self, train_path: str = None, val_path: str = None):
        if train_path:
            self.train_df = self.rename_cols_and_drop_na(
                                pd.read_csv(train_path, sep="\t"))
        if val_path:
            self.val_df = self.rename_cols_and_drop_na(
                            pd.read_csv(val_path, sep="\t"))

    def rename_cols_and_drop_na(self, quora_df: pd.DataFrame) -> pd.DataFrame:
        # TODO: add docstring
        quora_df = quora_df.dropna(axis=0, how="any").reset_index(drop=True)
        renamed_quora_df = pd.DataFrame(
            {
                "id_left": quora_df["qid1"],
                "id_right": quora_df["qid2"],
                "text_left": quora_df["question1"],
                "text_right": quora_df["question2"],
                "label": quora_df["is_duplicate"].astype(int),
            }
        )
        renamed_quora_df['id_left'] = renamed_quora_df['id_left'].astype(str)
        renamed_quora_df['id_right'] = renamed_quora_df['id_right'].astype(str)
        return renamed_quora_df

    def _handle_punctuation(self, input_str: str) -> str:
        # TODO: add docstring
        translator = str.maketrans(string.punctuation,
                                   " " * len(string.punctuation))
        new_str = input_str.translate(translator)
        return new_str

    def lower_and_tokenize_words(self, input_str: str) -> List[str]:
        # TODO: add docstring
        no_punctuation_str = self._handle_punctuation(input_str)
        lowered_str = no_punctuation_str.lower()
        splitted_doc = nltk.word_tokenize(lowered_str)
        return splitted_doc

    def _filter_rare_words(self,
                           vocab: Dict[str, int],
                           min_occurancies: int
                           ) -> Dict[str, int]:
        # TODO: add docstring
        filtered_vocab = {
            x: count for x, count in vocab.items() if count >= min_occurancies
        }
        return filtered_vocab

    def get_all_tokens(self,
                       min_occurancies: int,
                       save_path: str = None
                       ) -> List[str]:
        # TODO: add docstring
        preped_series = []
        for df in [self.train_df, self.val_df]:
            if isinstance(df, pd.DataFrame):
                preped_question1 = df["text_left"].apply(
                                    self.lower_and_tokenize_words)
                preped_question2 = df["text_right"].apply(
                                    self.lower_and_tokenize_words)
                preped_series.append(preped_question1)
                preped_series.append(preped_question2)

        concat_series = pd.concat(preped_series)
        one_list_of_tokens = list(
            itertools.chain.from_iterable(concat_series.to_list())
        )
        vocab = dict(Counter(one_list_of_tokens))
        vocab = self._filter_rare_words(vocab, min_occurancies)
        tokens = list(vocab.keys())
        if save_path:
            with open(save_path, 'wb') as fp:
                pickle.dump(tokens, fp)
        return tokens

    @classmethod
    def load_tokens(cls, load_path: str) -> List[str]:
        if isfile(load_path):
            with open(load_path, 'rb') as fp:
                tokens = pickle.load(fp)
            return tokens
        else:
            return None

    def get_idx_to_text_mapping(self, type_df: str) -> Dict[str, str]:
        assert type_df in ['train', 'val']
        if type_df == 'train':
            inp_df = self.train_df
        else:
            inp_df = self.val_df

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

    def get_and_save_documents(self, save_path: str):
        documents = dict()
        documents.update(self.get_idx_to_text_mapping('train'))
        documents.update(self.get_idx_to_text_mapping('val'))
        with open(save_path, 'w') as fp:
            json.dump(documents, fp, sort_keys=True, indent=4)

    @classmethod
    def load_documents(cls, load_path: str) -> Dict[str, str]:
        if isfile(load_path):
            with open(load_path, 'r') as fp:
                documents = json.load(fp)
            return documents
        else:
            return None
