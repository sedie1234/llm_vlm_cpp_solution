from typing import Mapping, Dict, Any
import torch
from transformers import AutoTokenizer, AutoConfig
from optimum.exporters.onnx import main_export, OnnxConfigWithPast


class DummyNormalizedConfig:
    def __init__(self, config):
        self.config = config


# Gemma3 ONNX export config
class CustomGemma3OnnxConfig(OnnxConfigWithPast):
    NORMALIZED_CONFIG_CLASS = DummyNormalizedConfig

    def __init__(self, config, task: str = None):
        super().__init__(config, task=task)

    @property
    def values_override(self) -> Dict[str, Any]:
        return {"use_cache": True}

    def with_behavior(self, behavior: str, use_past: bool = False):
        self.use_past = use_past
        return self

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        inputs = {
            "input_ids": {0: "batch", 1: "sequence"},
            "position_ids": {0: "batch", 1: "sequence"},
        }
        if getattr(self, "use_past", False):
            inputs["past_key_values"] = {0: "batch"}
        return inputs

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        outputs = {
            "logits": {0: "batch", 1: "sequence"},
        }
        if getattr(self, "use_past", False):
            outputs["past_key_values"] = {0: "batch"}
        return outputs

    def generate_dummy_inputs(
        self,
        model=None,
        tokenizer=None,
        framework: str = "pt",
        batch_size: int = 1,
        sequence_length: int = 1,
        past_sequence_length: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

        input_ids = torch.ones((batch_size, sequence_length), dtype=torch.long)
        position_ids = torch.arange(sequence_length).unsqueeze(0)

        inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
        }

        if getattr(self, "use_past", False):
            num_layers = self._config.num_hidden_layers
            num_kv_heads = self._config.num_key_value_heads
            num_attn_heads = self._config.num_attention_heads
            hidden_size = self._config.hidden_size
            head_dim = hidden_size // num_attn_heads

            for _ in range(num_layers):
                k = torch.zeros((batch_size, num_kv_heads, past_sequence_length, head_dim))
                v = torch.zeros((batch_size, num_kv_heads, past_sequence_length, head_dim))
                past_key_values.append((k, v))

            inputs["past_key_values"] = tuple(past_key_values)

        return inputs


if __name__ == "__main__":
    model_id = "google/gemma-3-1b-it"

    config = AutoConfig.from_pretrained(model_id)
    base_config = CustomGemma3OnnxConfig(config=config, task="causal-lm")

    decoder_config = base_config.with_behavior("decoder", use_past=False)
    decoder_with_past_config = base_config.with_behavior("decoder", use_past=True)

    custom_onnx_configs = {
        "decoder_model": decoder_config,
        "decoder_with_past_model": decoder_with_past_config,
    }

    fn_get_submodels = lambda model: {
        "decoder_model": model,
        "decoder_with_past_model": model,
    }

    main_export(
        model_name_or_path=model_id,
        output="./onnx-gemma3-split",
        task="causal-lm",
        opset=17,
        device="cpu",
        dtype="float32",
        no_post_process=True,
        model_kwargs={"use_cache": True},
        custom_onnx_configs=custom_onnx_configs,
        fn_get_submodels=fn_get_submodels,
    )
