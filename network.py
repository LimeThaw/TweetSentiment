import torch
import config

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # self.dropout = torch.nn.Dropout(p=0.5)
        self.embed = torch.nn.Embedding(
            num_embeddings=config.DICT_SIZE,
            embedding_dim=config.EMBED_DIM,
            padding_idx=0,
            max_norm=None,
            norm_type=2.0,
            scale_grad_by_freq=False
        )
        self.embed.load_state_dict(torch.load(config.get_embed_dict_name()))
        # self.lstm = torch.nn.LSTM(
        #     input_size=config.EMBED_DIM,
        #     hidden_size=config.LATENT_DIM,
        #     num_layers=1,
        #     bidirectional=False,
        #     batch_first=True
        # )
        self.linear_1 = torch.nn.Linear(
            # in_features=config.LATENT_DIM,
            in_features=config.EMBED_DIM*63,
            # out_features=config.LATENT_DIM
            out_features=1
        )
        torch.nn.init.normal_(self.linear_1.weight)
        # self.linear_2 = torch.nn.Linear(
        #     # in_features=config.LATENT_DIM,
        #     in_features=config.LATENT_DIM,
        #     out_features=1
        # )
        # torch.nn.init.normal_(self.linear_2.weight)
        self.activation = torch.nn.Sigmoid()

        # def _init_weights(m):
        #     if type(m) == torch.nn.Embedding or type(m) == torch.nn.Linear:
        #         torch.nn.init.normal_(m.weight)
        # self.apply(_init_weights)

    def trainable_weights(self):
        return list(self.linear_1.parameters())# + list(self.linear_2.parameters())

    def __call__(self, x):
        # activated = self.get_embedded(x)
        embedded = torch.flatten(self.embed(x), start_dim=1)
        # dropped = self.dropout(embedded)
        # _, f_state = self.lstm(activated)
        # linearized = self.linear(torch.mean(f_state[0], dim=0))
        linearized = self.activation(self.linear_1(embedded))
        return linearized
        # return self.activation(self.linear_2(linearized))