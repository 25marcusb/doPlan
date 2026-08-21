"""Thin wrapper around Qwen3-VL for image-text-to-text generation.

Loads the model once (bf16, device_map="auto") and answers chat-style messages that
interleave text with PIL montage images. Kept separate from the labeling logic so the
same model can serve the smoke test, the batch labeler, and any experiments.
"""

from typing import Dict, List

import torch

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _images_from_messages(messages: List[Dict]) -> List:
    imgs = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "image":
                    imgs.append(c["image"])
    return imgs


class QwenVL:
    def __init__(self, model_name: str, dtype: str = "bfloat16",
                 max_new_tokens: int = 256, do_sample: bool = False):
        self.model_name = model_name
        self.dtype = _DTYPES.get(dtype, torch.bfloat16)
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.model = None
        self.processor = None

    def load(self):
        # Import here so `import model_qwen` is cheap and doesn't require transformers
        # until a model is actually needed.
        from transformers import AutoProcessor
        try:
            from transformers import Qwen3VLForConditionalGeneration as _Model
        except ImportError:
            from transformers import AutoModelForImageTextToText as _Model

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not available; Qwen3-VL-8B needs a GPU here.")

        print(f"[model] loading {self.model_name} ({self.dtype}) ...", flush=True)
        self.model = _Model.from_pretrained(
            self.model_name, torch_dtype=self.dtype, device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        print("[model] loaded.", flush=True)
        return self

    @torch.inference_mode()
    def generate(self, messages: List[Dict]) -> str:
        """Run one chat turn and return the assistant's decoded text."""
        if self.model is None:
            self.load()

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        images = _images_from_messages(messages)
        inputs = self.processor(
            text=[text], images=images if images else None,
            return_tensors="pt", padding=True,
        ).to(self.model.device)

        generated = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=self.do_sample,
        )
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
