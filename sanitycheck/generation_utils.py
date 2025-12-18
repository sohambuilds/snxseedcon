"""
Sanity-check-local generation helpers.

Why this exists:
- We keep sanitycheck logic self-contained (per your request).
- We add `min_new_tokens` to prevent degenerate generations where the model
  immediately emits EOS (producing "" or "\n").
- We do NOT change anything condition-specific here; this is shared plumbing.
"""

from __future__ import annotations

from typing import Optional
import torch


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def generate_text(
    *,
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    min_new_tokens: int,
    do_sample: bool,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
) -> str:
    """
    Generate text from a prompt with consistent, explicit decoding settings.

    Notes:
    - `min_new_tokens` is crucial for avoiding empty outputs.
    - We only pass sampling args when `do_sample=True`.
    """
    if seed is not None:
        _seed_everything(seed)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )

    if do_sample:
        # transformers expects floats; avoid passing None
        if temperature is not None:
            gen_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)

    outputs = model.generate(**gen_kwargs)

    # Decode only the generated part
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


