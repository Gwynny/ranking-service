import itertools
import nltk
import string
import pandas as pd
from collections import Counter
from typing import Dict, List


class TextRetriever:
    def __init__(self, train_path: str, val_path: str):
        self.train_df = pd.read_csv(train_path, sep="\t")
        self.val_df = pd.read_csv(val_path, sep="\t")

    def _rename_cols_and_drop_na(self, quora_df: pd.DataFrame) -> pd.DataFrame:
        # TODO: add docstring
        quora_df = quora_df.dropna(axis=0, how="any").reset_index(drop=True)
        quora_df_cols_renamed = pd.DataFrame(
            {
                "id_left": quora_df["qid1"],
                "id_right": quora_df["qid2"],
                "text_left": quora_df["question1"],
                "text_right": quora_df["question2"],
                "label": quora_df["is_duplicate"].astype(int),
            }
        )
        return quora_df_cols_renamed

    def _handle_punctuation(self, input_str: str) -> str:
        # TODO: add docstring
        translator = str.maketrans(string.punctuation,
                                   " " * len(string.punctuation))
        new_str = input_str.translate(translator)
        return new_str

    def _lower_and_tokenize_words(self, input_str: str) -> List[str]:
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

    def get_all_tokens(self, min_occurancies: int) -> List[str]:
        # TODO: add docstring
        preped_series = []
        for df in [self.train_df, self.val_df]:  # TODO: write check if df is pd.DataFrame
            df = self._rename_cols_and_drop_na(df)
            preped_question1 = df["text_left"].apply(
                                self._lower_and_tokenize_words)
            preped_question2 = df["text_right"].apply(
                                self._lower_and_tokenize_words)
            preped_series.append(preped_question1)
            preped_series.append(preped_question2)

        concat_series = pd.concat(preped_series)
        one_list_of_tokens = list(
            itertools.chain.from_iterable(concat_series.to_list())
        )
        vocab = dict(Counter(one_list_of_tokens))
        vocab = self._filter_rare_words(vocab, min_occurancies)
        return list(vocab.keys())
