import faiss
import torch
import numpy as np
from src.data.text_retriever import TextRetriever
from typing import Dict


class FaissIndex:
    def __init__(self, documents: Dict[str, str]):
        self.documents = documents
        self.tr = TextRetriever()

    def prepare_index(self, emb_layer: torch.Tensor, vocab: dict):
        idxs, docs = [], []
        for idx in self.documents:
            idxs.append(int(idx))
            docs.append(self.documents[idx])

        embeddings, oov_val = [], vocab['OOV']
        for d in docs:
            tmp_emb = [vocab.get(w, oov_val) for w in self.tr.lower_and_tokenize_words(d)]
            tmp_emb = emb_layer[tmp_emb].mean(dim=0)
            embeddings.append(np.array(tmp_emb))
        embeddings = np.array([embedding for embedding in embeddings]).astype(np.float32)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index = faiss.IndexIDMap(self.index)
        self.index.add_with_ids(embeddings, np.array(idxs))
        return self.index

    def add_docs_to_index(self):
        pass

    def save_index(self):
        pass

    def load_index(self):
        pass
