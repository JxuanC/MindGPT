from modules.vit import ViT
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
import torch

class fMRIViTEncoderConfig(PretrainedConfig):
    model_type = "fMRIViTEncoder"

    def __init__(
        self,
        fmri_dim, rois_len, embed_dim, depth, num_heads, 
        fmri2img = False, **kwargs,
    ):
        super().__init__(**kwargs)
        self.fmri_dim = fmri_dim
        self.rois_len = rois_len
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.hidden_size = embed_dim
        self.fmri2img = fmri2img

class fMRIViTEncoder(PreTrainedModel):
    config_class = fMRIViTEncoderConfig
    def __init__(self, config):
        super(fMRIViTEncoder, self).__init__(config)
        if config.fmri2img:
            self.proj = nn.Linear(config.fmri_dim, 112 * 112 * 3)
            self.patch_embed = PatchEmbed(112, 16, 3, config.embed_dim)
            self.encoder = ViT(config.embed_dim, 49, config.embed_dim, config.depth, config.num_heads)
        else:
            self.encoder = ViT(config.fmri_dim, config.rois_len, config.embed_dim, config.depth, config.num_heads)
        self.config = config

    def forward(self, encoder_inputs, **kwargs):
        # encoder_inputs shape (batch, roi_num, roi_dim)
        if(self.config.fmri2img):
            encoder_inputs = self.proj(encoder_inputs)
            encoder_inputs = torch.reshape(encoder_inputs, (-1, 3, 112, 112))
            encoder_inputs = self.patch_embed(encoder_inputs)
        else:
            encoder_outputs = self.encoder(encoder_inputs)
        return encoder_outputs