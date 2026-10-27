"""Decision Index kit Engine for the Jev-Style v3 decision models (0.8B and 2B), PyTorch runtime as shipped.

Uses the public Hub repo's own runtime (jev_style_decision.py) at a pinned revision, manifest-verified (sha256 of
every file), float32 on CUDA (the parity-tested setting). Fixed protocol for every benchmark: the runtime's one
rendering, ONE global temperature (category=None), no prompt tuning, no option filtering, options in request order,
nothing truncated. Object instructions / option descriptions are passed as compact JSON (the kit's `text`).
0.8B: a choice question whose options exceed the 2,048-token head budget is scored in option chunks (one softmax
over all options). 2B: no separate question/options budget (numbered-option catalogue layout past 2,048 tokens).
Requests the runtime rejects (InputBudgetError / QuestionError: over 25,600 input tokens) raise the kit's
Unsupported (scored as wrong).

Options: model=0.8b|2b, revision=<commit> (default: main), model_dir=<local snapshot>, device=cuda, dtype=float32.
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

from decision_index.engines.base import Engine, Unsupported, text

REPOS = {"0.8b": "chaoliangUNSW/Jev-Style-0.8B-Decision-v3", "2b": "chaoliangUNSW/Jev-Style-2B-Decision-v3"}


def load_runtime(path):
    sys.dont_write_bytecode = True          # never write __pycache__ into the model folder (manifest check)
    spec = importlib.util.spec_from_file_location("jev_style_decision", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["jev_style_decision"] = mod
    spec.loader.exec_module(mod)
    return mod


def to_runtime_question(q):
    ins = q.get("instructions", "")
    ins = ins if isinstance(ins, str) else text(ins)
    if q["type"] == "choice":
        crit = {str(k): (v if (v is None or isinstance(v, str)) else text(v)) for k, v in q["criteria"].items()}
        return {"t": "choice", "ins": ins, "crit": crit}
    if q["type"] == "noul":
        c = q.get("criteria") or {}
        crit = {k: (c[k] if isinstance(c[k], str) else text(c[k])) for k in ("false", "true") if k in c and c[k] not in (None, "")}
        return {"t": "noul", "ins": ins, "crit": crit or None}
    raise Unsupported("unsupported question type " + str(q["type"]))


class JevStyleTorchEngine(Engine):
    latency = "In-process request wall time (CUDA synchronized): rendering + the runtime's decide_many (state computed once per request); excludes model loading."

    def __init__(self, model="2b", revision=None, model_dir=None, device="cuda", dtype="float32", **options):
        super().__init__(**options)
        repo = REPOS[str(model).lower()]
        self.name = repo.split("/")[1].lower()
        if model_dir is None:
            from huggingface_hub import snapshot_download
            model_dir = snapshot_download(repo, revision=revision)
        model_dir = Path(model_dir)
        src = model_dir / "jev_style_decision.py"
        self.rt = load_runtime(src)
        # Every file listed in manifest.json must be present with its sha256. The 0.8B repo's manifest at
        # ec7e427d also lists a stray __pycache__/*.pyc that was never uploaded; only such entries may be absent.
        res = self.rt.verify_manifest(model_dir)
        stray = [f for f in res["missing"] if f.startswith("__pycache__/")]
        if res["bad"] or len(stray) != len(res["missing"]) or not res["checked"]:
            raise RuntimeError(f"integrity check failed: {res}")
        self.m = self.rt.JevStyleDecision(model_dir, device=device, dtype=dtype, verify=False)
        self.torch = self.m.torch
        self.device = device
        temps = getattr(self.m, "temperatures", None)
        temp = temps.get("global") if isinstance(temps, dict) else getattr(self.m, "temperature", None)
        self.provenance = {"model": self.rt.MODEL_NAME, "hf_repo": repo, "revision": revision or "main",
                           "snapshot": model_dir.name, "runtime_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                           "manifest_verified": {"checked": res["checked"], "bad": res["bad"], "missing_ignored": stray}, "device": device, "dtype": dtype,
                           "template": getattr(self.rt, "TEMPLATE_VERSION", None),
                           "context_limit": self.rt.CONTEXT_LIMIT, "head_max": getattr(self.rt, "HARD_HEAD_MAX", None),
                           "temperature": "global (category=None) = %s" % (temp,),
                           "no_truncation": True, "option_filtering": False, "per_benchmark_prompts": False}

    def __call__(self, state, questions):
        keys = list(questions)
        qs = [to_runtime_question(questions[k]) for k in keys]
        try:
            out = self.m.decide_many(state, qs, category=None)
        except self.rt.InputBudgetError as e:
            raise Unsupported(f"InputBudgetError: {e}") from None
        except self.rt.QuestionError as e:
            raise Unsupported(f"QuestionError: {e}") from None
        answers, raw, tokens = {}, {}, 0
        for k, q, r in zip(keys, questions.values(), out):
            if q["type"] == "choice":
                probs = {str(n): float(r["probabilities"][str(n)]) for n in q["criteria"]}
                answers[k] = {"type": "choice", "choice": r["answer"], "probabilities": probs}
            else:
                answers[k] = {"type": "noul", "noul": float(r["probabilities"]["true"])}
            raw[k] = {x: r[x] for x in ("scores", "temperature", "input_tokens", "head_tokens", "option_chunks",
                                        "catalogue_overflow", "blocks") if x in r}
            tokens += r["input_tokens"]
        return {"model": self.rt.MODEL_NAME, "answers": answers, "usage": {"input_tokens": tokens}}, raw

    def runtime(self):
        import transformers
        t = self.torch
        import importlib.metadata as md

        def ver(pkg):
            try:
                return md.version(pkg)
            except md.PackageNotFoundError:
                return None
        return {"torch": t.__version__, "transformers": transformers.__version__, "device": self.device,
                "flash_linear_attention": ver("flash-linear-attention"), "causal_conv1d": ver("causal-conv1d"),
                "gpu": t.cuda.get_device_name(0) if t.cuda.is_available() else None,
                "cuda": t.version.cuda, "tf32_matmul": t.backends.cuda.matmul.allow_tf32}

    def synchronize(self):
        if str(self.device).startswith("cuda"):
            self.torch.cuda.synchronize()
