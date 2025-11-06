from typing import List, Dict, Optional
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

ALLOWED_TOOLS = ["run_python", "inspect_traceback", "set_code"]
TOOLS_ALT = "|".join(re.escape(t) for t in ALLOWED_TOOLS)
PATTERN = rf"(?is)(FINAL_ANSWER:.*?|TOOL:\s*(?:{TOOLS_ALT}).*?)\s*(?=TOOL:|FINAL_ANSWER:|$)"

class LLM:
    def __init__(self, model_name: str = "dummy", **cfg):
        self.model_name = model_name
        self.cfg = cfg
        self.max_new_tokens = int(cfg.get("max_new_tokens", 400))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.verbose = bool(cfg.get("verbose_logs", False))
        self.save_raw = bool(cfg.get("save_raw_files", False))
        self.last_prompt: Optional[str] = None
        self.last_raw: Optional[str] = None

        self._hf = None

        self.default_system = (
            "You are a code-fixing assistant.\n"
            "Respond with EXACTLY ONE directive and nothing else:\n"
            "- TOOL: run_python\n"
            "- TOOL: inspect_traceback\n"
            "- TOOL: set_code\n<code>\n<ENTIRE CORRECTED FILE HERE>\n</code>\n"
            "- FINAL_ANSWER: <short status>\n"
            "Do NOT include <think> or prose outside the directive.\n"
        )

    def chat(self, messages: List[Dict[str, str]]) -> str:
        text = self._generate(messages)  
        text = re.sub(r"(?is)<think>.*?</think>", "", text)

        cands = re.findall(PATTERN, text)
        if not cands:
            return "TOOL: run_python"
        directive = cands[-1].strip().replace("\r\n", "\n")

        return directive


    def _ensure_hf(self):
        if self._hf is not None:
            return
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=dtype 
        )
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        has_chat_template = hasattr(tok, "apply_chat_template")
        self._hf = dict(tokenizer=tok, model=model, has_chat_template=has_chat_template)

    def _format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": self.default_system}] + messages
        return messages

    def _encode_prompt(self, messages) -> str:
        tok = self._hf["tokenizer"]
        if self._hf["has_chat_template"]:
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return prompt, True
        def tag(m): return f"[{m['role'].upper()}]\n{m['content']}\n"
        prompt = "".join(tag(m) for m in messages) + "[ASSISTANT]\n"
        return prompt, False

    def _generate(self, messages: List[Dict[str, str]]) -> str:
        self._ensure_hf()
        tok = self._hf["tokenizer"]
        model = self._hf["model"]

        msgs = self._format_messages(messages)
        prompt, _ = self._encode_prompt(msgs)
        self.last_prompt = prompt

        inputs = tok(prompt, return_tensors="pt")
        for k in inputs:
            inputs[k] = inputs[k].to(model.device)

        do_sample = self.temperature > 0.0
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
            eos_token_id=tok.eos_token_id,
        )
        out_ids = gen_ids[0, inputs["input_ids"].shape[-1]:]
        text = tok.decode(out_ids, skip_special_tokens=True)
        self.last_raw = text

        if self.verbose:
            print("\n----- RAW MODEL OUTPUT START -----")
            print(text)
            print("----- RAW MODEL OUTPUT END -----\n")

        if self.save_raw:
            os.makedirs("logs", exist_ok=True)
            with open("logs/last_prompt.txt", "w", encoding="utf-8") as f:
                f.write(self.last_prompt or "")
            with open("logs/last_raw.txt", "w", encoding="utf-8") as f:
                f.write(self.last_raw or "")

        return text

