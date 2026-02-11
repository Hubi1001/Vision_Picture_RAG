#!/usr/bin/env python3
"""
Qwen-based image verifier for metal parts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from transformers import AutoProcessor
try:
    from transformers import Qwen2VLForConditionalGeneration
except ImportError:
    Qwen2VLForConditionalGeneration = None

DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"


def _select_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _coerce_confidence(val: Any) -> float:
    try:
        num = float(val)
        return max(0.0, min(1.0, num))
    except Exception:
        return 0.0


@dataclass
class VerificationResult:
    verdict: str
    confidence: float
    reason: str
    raw: str


def _parse_output(text: str) -> VerificationResult:
    try:
        parsed = json.loads(text)
    except Exception:
        return VerificationResult(verdict="unsure", confidence=0.0, reason=text, raw=text)

    verdict = str(parsed.get("verdict", "unsure")).lower()
    if verdict not in {"yes", "no", "unsure"}:
        verdict = "unsure"

    confidence = _coerce_confidence(parsed.get("confidence", 0.0))
    reason = str(parsed.get("reason", "")).strip()
    return VerificationResult(verdict=verdict, confidence=confidence, reason=reason, raw=text)


class QwenImageVerifier:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: Optional[str] = None):
        if Qwen2VLForConditionalGeneration is None:
            raise ImportError(
                "Qwen2VLForConditionalGeneration not found. "
                "Install with: pip install git+https://github.com/huggingface/transformers"
            )
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = _select_dtype()
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model_kwargs = dict(
            torch_dtype=dtype,
            device_map="auto" if self.device.startswith("cuda") else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if self.device.startswith("cuda"):
            # Prefer faster attention implementations when available
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id,
                    **model_kwargs,
                )
            except Exception:
                model_kwargs.pop("attn_implementation", None)
                try:
                    model_kwargs["attn_implementation"] = "sdpa"
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_id,
                        **model_kwargs,
                    )
                except Exception:
                    model_kwargs.pop("attn_implementation", None)
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_id,
                        **model_kwargs,
                    )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                **model_kwargs,
            )
        if not self.device.startswith("cuda"):
            self.model = self.model.to(self.device)
        self.model.eval()
        
        # torch.compile dla PyTorch 2.0+ (przyspieszenie inferencji)
        if self.device.startswith("cuda") and hasattr(torch, "compile") and torch.__version__ >= "2.0":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
            except Exception:
                pass  # Ignore compile errors, fallback to eager mode

    def _build_messages(self, image: Image.Image, claim: str):
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You verify whether an image matches a technical claim about a metal part. "
                            "Return JSON: {\"verdict\": \"yes|no|unsure\", \"confidence\": 0-1, \"reason\": \"short\"}. "
                            "Be concise and decisive."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Claim: {claim}"},
                    {"type": "image", "image": image},
                ],
            },
        ]

    def verify(
        self,
        image_path: str | Path,
        claim: str,
        max_new_tokens: int = 128,
        temperature: float = 0.1,
    ) -> VerificationResult:
        image = Image.open(image_path).convert("RGB")
        messages = self._build_messages(image=image, claim=claim)
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        output_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        output_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # Zwolnij pamięć GPU po inferencji
        del inputs, generated_ids, output_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return _parse_output(output_text)


def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="Qwen image verifier")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--claim", required=True, help="Claim to verify against the image")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Qwen VL model id")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    verifier = QwenImageVerifier(model_id=args.model)
    result = verifier.verify(
        image_path=args.image,
        claim=args.claim,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
