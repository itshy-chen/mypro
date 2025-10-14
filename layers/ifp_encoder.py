from .mamba import MambaTimeEncoder


class IFPencoder(nn.Module):
    """Top-level encoder container combining agent and map encoders."""

    def __init__(
        self,
        agent_input_dim: int,
        *,
        agent_d_model: int = 256,
        agent_n_layers: int = 2,
        agent_dropout: float = 0.0,
        map_feat_channel: int,
        map_encoder_channel: int,
    ) -> None:
        super().__init__()
        self.agent_encoder = MambaTimeEncoder(
            input_dim=agent_input_dim,
            d_model=agent_d_model,
            n_layers=agent_n_layers,
            dropout=agent_dropout,
        )
        self.map_encoder = LaneEmbeddingLayer(
            feat_channel=map_feat_channel,
            encoder_channel=map_encoder_channel,
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "IFPencoder is a container and does not process data directly."
        )
