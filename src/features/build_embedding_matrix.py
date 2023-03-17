import numpy as np
from typing import Dict, List, Tuple


class EmbeddingMatrixBuilder:
    def __init__(self, glove_path, uniform_bound, random_seed):
        self.glove_path = glove_path
        self.uniform_bound = uniform_bound
        np.random.seed(random_seed)

    def _read_glove_embeddings(self) -> Dict[str, List[str]]:
        with open(self.glove_path, encoding='utf-8') as file:
            glove_dict = {}
            for line in file:
                splitted_line = line.split()
                word, embedding = splitted_line[0], splitted_line[1:]
                glove_dict[word] = embedding
        return glove_dict

    def _generate_random_vector(self, emb_dim: int) -> np.array:
        vec = np.random.uniform(low=-self.uniform_bound,
                                high=self.uniform_bound,
                                size=emb_dim)
        return vec

    def _create_vocab_and_emb_matrix(self,
                                     retrieved_words: List[str],
                                     glove_dict: Dict[str, List[str]],
                                     emb_dim: int
                                     ) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
        pad_vec = self._generate_random_vector(emb_dim)
        oov_vec = self._generate_random_vector(emb_dim)
        emb_matrix = []
        emb_matrix.append(pad_vec)
        emb_matrix.append(oov_vec)

        vocab, unk_words = {}, []
        vocab['PAD'], vocab['OOV'] = 0, 1
        for ind, token in enumerate(retrieved_words, 2):
            if token in glove_dict.keys():
                emb_matrix.append(glove_dict[token])
            else:
                unk_words.append(token)
                random_emb = self._generate_random_vector(emb_dim)
                emb_matrix.append(random_emb)
            vocab[token] = ind

        emb_matrix = np.array(emb_matrix).astype(float)
        return (emb_matrix, vocab, unk_words)

    def create_glove_emb_from_file(self,
                                   retrieved_words: List[str]
                                   ) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
        glove_dict = self._read_glove_embeddings()
        emb_dim = len(glove_dict['the'])

        emb_matrix, vocab, unk_words = self._create_vocab_and_emb_matrix(
            retrieved_words, glove_dict, emb_dim
        )
        return (emb_matrix, vocab, unk_words)
