import torch
import numpy as np
from faiss.swigfaiss import IndexIDMap
from src.data.text_retriever import TextRetriever
from src.models.knrm_model import KNRM
from typing import Dict, List


class FunnelModel:
    def __init__(self, index: IndexIDMap, reranker: KNRM, documents: Dict[str, str],
                 vocab: Dict[str, str], query_max_len: int = 30, num_cands: int = 100,
                 num_output: int = 10):
        self.index = index
        self.reranker = reranker
        self.emb_layer = self.reranker.embeddings.state_dict()['weight']
        self.documents = documents
        self.vocab = vocab
        self.oov_val = self.vocab['OOV']
        self.query_max_len = query_max_len
        self.tr = TextRetriever()
        self.num_cands = num_cands
        self.num_output = num_output

    def _text_to_token_ids(self, text_list: List[str]) -> torch.LongTensor:
        tokenized = []
        for text in text_list:
            tokenized_text = self.tr.lower_and_tokenize_words(text)[:self.query_max_len]
            token_idxs = [self.vocab.get(i, self.oov_val) for i in tokenized_text]
            tokenized.append(token_idxs)
        tokenized = [elem + [0] * (self.query_max_len - len(elem)) for elem in tokenized]
        tokenized = torch.LongTensor(tokenized)
        return tokenized

    def get_docs(self, query: str) -> List[str]:
        q_vector = [self.vocab.get(token, self.oov_val) for token in self.tr.lower_and_tokenize_words(query)]
        q_emb = self.emb_layer[q_vector].mean(dim=0).reshape(1, -1)
        q_emb = np.array(q_emb).astype(np.float32)
        _, cands_inds = self.index.search(q_emb, k=self.num_cands)
        cands = [self.documents[str(i)] for i in cands_inds[0] if i != -1]
        inputs = dict()
        inputs['query'] = self._text_to_token_ids([query] * len(cands))
        inputs['document'] = self._text_to_token_ids(cands)
        scores = self.reranker.predict(inputs)
        res_ids = scores.reshape(-1).argsort(descending=True)
        res_ids = res_ids[:self.num_output]
        res = [cands[i] for i in res_ids.tolist()]
        return res
