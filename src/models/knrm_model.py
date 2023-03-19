import torch
import numpy as np
from typing import Dict, List


class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        numerator = -torch.pow((x - self.mu), 2)
        denominator = 2 * self.sigma**2
        return torch.exp(numerator / denominator)


class KNRM(torch.nn.Module):
    def __init__(self, embedding_matrix: np.ndarray,
                 freeze_embeddings: bool,
                 kernel_num: int = 21,
                 sigma: float = 0.1,
                 exact_sigma: float = 0.001,
                 out_layers: List[int] = [10, 5]):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze_embeddings,
            padding_idx=0
        )

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()
        self.mlp = self._get_mlp()
        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        kernels = torch.nn.ModuleList()
        shrink_len = 1.0 / (self.kernel_num - 1)
        left, right = -1.0 + shrink_len, 1.0 - shrink_len
        mus = np.append(np.linspace(left, right, self.kernel_num-1), 1.0)
        sigmas = np.array((self.kernel_num-1) * [self.sigma] + [self.exact_sigma])

        for mu, sigma in zip(mus, sigmas):
            kernels.append(GaussianKernel(mu=mu, sigma=sigma))

        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:
        if len(self.out_layers) == 0:
            return torch.nn.Sequential(torch.nn.Linear(self.kernel_num, 1))

        layers = []
        layers.append(torch.nn.Linear(self.kernel_num, self.out_layers[0]))
        layers.append(torch.nn.ReLU())
        for i in range(1, len(self.out_layers)):
            layers.append(torch.nn.Linear(self.out_layers[i-1], self.out_layers[i]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(self.out_layers[-1], 1))
        return torch.nn.Sequential(*layers)

    def forward(self, input_1: Dict[str, torch.Tensor],
                input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor,
                             doc: torch.Tensor
                             ) -> torch.FloatTensor:
        # https://stackoverflow.com/questions/50411191/
        # how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a-matrix-with-re
        eps = 1e-8
        query_m, doc_m = self.embeddings(query), self.embeddings(doc)
        query_norm = doc_m.norm(dim=2)[:, :, None]
        doc_norm = doc_m.norm(dim=2)[:, :, None]
        query_normalised = query_m / torch.clamp(query_norm, min=eps)
        doc_normalised = doc_m / torch.clamp(doc_norm, min=eps)
        similarity_m = torch.bmm(query_normalised, doc_normalised.transpose(1, 2))
        return similarity_m

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        # shape = [Batch, Left], [Batch, Right]
        query, doc = inputs['query'], inputs['document']

        # shape = [Batch, Left, Right]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape = [Batch, Kernels]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape = [Batch]
        out = self.mlp(kernels_out)
        return out
