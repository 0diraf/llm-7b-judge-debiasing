

import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODELS = {
    "qwen": {
        "hf_id":   "Qwen/Qwen2.5-7B-Instruct",
        "display": "Qwen2.5-7B-Instruct",
        "dtype":   torch.float16,
        "trust_remote_code": False,
    },
    "qwen14b": {
        "hf_id":   "Qwen/Qwen2.5-14B-Instruct",
        "display": "Qwen2.5-14B-Instruct",
        "dtype":   torch.float16,
        "trust_remote_code": False,
    },
    "mistral": {
        "hf_id":   "mistralai/Mistral-7B-Instruct-v0.3",
        "display": "Mistral-7B-Instruct-v0.3",
        "dtype":   torch.float16,
        "trust_remote_code": False,
    },
    "olmo": {
        "hf_id":   "allenai/OLMo-2-1124-7B-Instruct",
        "display": "OLMo-2-7B-Instruct",
        "dtype":   torch.float16,
        "trust_remote_code": False,
    },
}

_tokenizer = None
_model = None
_current_model_key = None


def current_model() -> str | None:
    return _current_model_key


def load_model(model_key: str = "qwen", load_in_4bit: bool = False):
    global _tokenizer, _model, _current_model_key

    if model_key not in MODELS:
        raise ValueError(f"Unknown model '{model_key}'. Choose from: {list(MODELS)}")
    if _current_model_key == model_key:
        return
    if _model is not None:
        unload_model()

    cfg  = MODELS[model_key]
    print(f"Loading {cfg['display']} ({cfg['hf_id']})...")

    _tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"], trust_remote_code=cfg["trust_remote_code"])
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    load_kwargs = dict(trust_remote_code=cfg["trust_remote_code"], device_map="auto")
    if cfg.get("attn_implementation"):
        load_kwargs["attn_implementation"] = cfg["attn_implementation"]
    if load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
    else:
        load_kwargs["torch_dtype"] = cfg["dtype"]

    _model = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], **load_kwargs)
    _model.eval()
    _model.generation_config.temperature = None
    _model.generation_config.top_p = None
    _model.generation_config.top_k = None
    _model.generation_config.do_sample = False

    _current_model_key = model_key
    print(f"Loaded: {cfg['display']}")


def unload_model():
    global _tokenizer, _model, _current_model_key
    if _model is None:
        return
    print(f"Unloading {_current_model_key}...")
    del _model, _tokenizer
    _model = _tokenizer = None
    _current_model_key = None
    gc.collect()
    torch.cuda.empty_cache()


def generate(messages: list[dict], max_new_tokens: int = 50, temperature: float | None = None) -> tuple[str, int]:
    assert _model is not None, "Call load_model() first"

    text   = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(text, return_tensors="pt").to(_model.device)
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=_tokenizer.eos_token_id)
    if temperature is None:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        output_ids = _model.generate(**inputs, **gen_kwargs)

    new_ids = output_ids[0][input_len:]
    return _tokenizer.decode(new_ids, skip_special_tokens=True), len(new_ids)


def count_tokens(text: str) -> int:
    assert _tokenizer is not None, "Call load_model() first"
    return len(_tokenizer.encode(text, add_special_tokens=False))
