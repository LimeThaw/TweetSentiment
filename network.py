import torch
import config

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.embed = torch.nn.Embedding(
            num_embeddings=config.DICT_SIZE,
            embedding_dim=config.EMBED_DIM,
            padding_idx=0,
            max_norm=None,
            norm_type=2.0,
            scale_grad_by_freq=False,
            sparse=False
        )
        self.lstm = torch.nn.LSTM(
            input_size=config.EMBED_DIM,
            hidden_size=config.LATENT_DIM,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.linear = torch.nn.Linear(
            in_features=2*config.LATENT_DIM,
            out_features=1
        )
        self.activation = torch.nn.Sigmoid()

        def _init_weights(m):
            if type(m) == torch.nn.Embedding or type(m) == torch.nn.Linear:
                torch.nn.init.xavier_normal_(m.weight)
        self.apply(_init_weights)

    def __call__(self, x):
        embedded = self.embed(x)
        hidden, _ = self.lstm(embedded)
        linearized = self.linear(hidden[:, -1, :])
        return self.activation(linearized)