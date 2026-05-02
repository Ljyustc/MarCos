import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

# StaticCache / SlidingWindowCache moved between modules across transformers
# versions (qwen2.modeling_qwen2 in some, cache_utils in others). Fall back
# gracefully so this file works on a wider range of installs.
try:
    from transformers.cache_utils import StaticCache
except ImportError:
    try:
        from transformers.models.qwen2.modeling_qwen2 import StaticCache
    except ImportError:
        StaticCache = None
try:
    from transformers.cache_utils import SlidingWindowCache
except ImportError:
    try:
        from transformers.models.qwen2.modeling_qwen2 import SlidingWindowCache
    except ImportError:
        SlidingWindowCache = None


class MyQwen2Model(Qwen2Model):
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values,
        output_attentions: bool = False,
    ):

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = (StaticCache is not None) and isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = (SlidingWindowCache is not None) and isinstance(past_key_values, SlidingWindowCache)

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        return causal_mask

    def _prepare_4d_causal_attention_mask_with_cache_position(
        self,
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config,
        past_key_values=None,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.zeros(
                (batch_size, 1, sequence_length, target_length),
                dtype=dtype,
                device=device
            )
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            if attention_mask.shape[-1] > target_length:
                attention_mask = attention_mask[:, :target_length]
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
        return causal_mask