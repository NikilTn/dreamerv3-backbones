from rssm import RSSM as GRURSSM
from rssm_mamba import RSSM as MambaRSSM
from rssm_s4 import RSSM as S4RSSM
from rssm_s5 import RSSM as S5RSSM
from rssm_transformer import RSSM as TransformerRSSM


BACKBONES = {
    "gru": GRURSSM,
    "rssm": GRURSSM,
    "transformer": TransformerRSSM,
    "storm": TransformerRSSM,
    "mamba": MambaRSSM,
    "mamba2": MambaRSSM,
    "s4": S4RSSM,
    "s3m": S4RSSM,
    "s5": S5RSSM,
}

BACKBONE_CONFIGS = {
    "gru": "rssm",
    "rssm": "rssm",
    "transformer": "transformer",
    "storm": "transformer",
    "mamba": "mamba",
    "mamba2": "mamba",
    "s4": "s4",
    "s3m": "s4",
    "s5": "s5",
}


def build_rssm(model_config, embed_size, act_dim):
    name = str(getattr(model_config, "backbone", "gru")).lower()
    try:
        rssm_cls = BACKBONES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown backbone '{name}'. Available: {sorted(BACKBONES)}") from exc
    backbone_config = getattr(model_config, BACKBONE_CONFIGS[name], None)
    return rssm_cls(model_config.rssm, embed_size, act_dim, backbone_config)
