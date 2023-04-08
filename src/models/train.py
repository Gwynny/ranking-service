import mlflow
import torch
from src.data.make_dataset import RankingDataset, TrainTripletsDataset, ValPairsDataset
from src.data.text_retriever import TextRetriever
from src.features.build_embedding_matrix import EmbeddingMatrixBuilder
from src.models.knrm_model import KNRM
from src.models.evaluate import check_ndcg_on_val_set
from typing import List


class TrainKNRM:
    def __init__(self, train_path: str, val_path: str, glove_path: str,
                 freeze_emb: bool, random_vec_bound: float,
                 min_token_occurancies: int, out_layers: List[int],
                 num_kernels: int, seed: int, sigma: float, lr: float,
                 num_epochs: int, change_every_num_ep: int, batch_size: int,
                 num_pos_ex: int, num_rand_pos_ex: int, num_same_rel_ex: int):
        self.train_path = train_path
        self.val_path = val_path
        self.glove_path = glove_path

        self.freeze_emb = freeze_emb
        self.seed = seed
        self.random_vec_bound = random_vec_bound
        self.min_token_occurancies = min_token_occurancies

        self.out_layers = out_layers
        self.num_kernels = num_kernels
        self.sigma = sigma

        self.lr = lr
        self.num_epochs = num_epochs
        self.change_every_num_ep = change_every_num_ep
        self.batch_size = batch_size

        self.num_pos_ex = num_pos_ex
        self.num_rand_pos_ex = num_rand_pos_ex
        self.num_same_rel_ex = num_same_rel_ex

    def _get_ready_for_train(self):
        self.retriever = TextRetriever(self.train_path, self.val_path)
        self.train_df = self.retriever.train_df
        self.val_df = self.retriever.val_df

        emb_builder = EmbeddingMatrixBuilder(self.random_vec_bound, self.seed)
        unique_tokens = self.retriever.get_all_tokens(min_occurancies=self.min_token_occurancies)
        emb_matrix, self.vocab, _ = emb_builder.create_glove_emb_from_file(self.glove_path, unique_tokens)
        torch.manual_seed(self.seed)
        self.knrm = KNRM(emb_matrix,
                         freeze_embeddings=self.freeze_emb,
                         sigma=self.sigma,
                         out_layers=self.out_layers,
                         kernel_num=self.num_kernels)

        self.idx_to_text_mapping_train = self.retriever.get_idx_to_text_mapping('train')
        self.idx_to_text_mapping_val = self.retriever.get_idx_to_text_mapping('val')

        val_pairs = ValPairsDataset.create_val_pairs(self.val_df)
        val_dataset = ValPairsDataset(
            val_pairs,
            self.idx_to_text_mapping_val,
            vocab=self.vocab,
            preproc_func=self.retriever.lower_and_tokenize_words)

        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            collate_fn=RankingDataset.collate_fn,
            shuffle=False
        )

    def _get_train_dataset(self, epoch):
        sampled_train_triplets = \
            TrainTripletsDataset.create_train_triplets(
                self.train_df,
                seed=epoch,
                num_positive_examples=self.num_pos_ex,
                num_random_positive_examples=self.num_rand_pos_ex,
                num_same_rel_examples=self.num_same_rel_ex
            )

        train_dataset = TrainTripletsDataset(
            sampled_train_triplets,
            self.idx_to_text_mapping_train,
            vocab=self.vocab,
            preproc_func=self.retriever.lower_and_tokenize_words
        )

        train_dataloader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=self.batch_size, num_workers=0,
                    collate_fn=RankingDataset.collate_fn, shuffle=True,
                    generator=torch.manual_seed(epoch))
        return train_dataloader

    def fit(self, benchmark_ndcg_score=0.925):
        self._get_ready_for_train()
        opt = torch.optim.SGD(self.knrm.parameters(), lr=self.lr)
        criterion = torch.nn.BCELoss()
        for ep in range(self.num_epochs):
            if ep % self.change_every_num_ep == 0:
                train_dataloader = self._get_train_dataset(ep)

            for j, data in enumerate(train_dataloader):
                opt.zero_grad()
                query_left_docs, query_right_docs, labels = data
                outputs = self.knrm(query_left_docs, query_right_docs)
                loss = criterion(outputs, labels)
                loss.backward()
                opt.step()

            val_ndcg = check_ndcg_on_val_set(self.knrm, self.val_dataloader)
            mlflow.log_metric("val_ndcg_by_epoch", val_ndcg, step=ep)
            if val_ndcg > benchmark_ndcg_score:
                break
