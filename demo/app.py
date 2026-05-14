"""
ECHO Streaming Demo
====================
Visualises one-step-per-block decoding in real-time.

Each iteration of the generation loop decodes a full block of tokens in a
single forward pass.  The UI colours newly-decoded tokens green so you can
see blocks "appearing" all at once, while the remaining masked positions are
shown as ▒ characters.

Usage:
    cd ECHO_release
    pip install gradio>=4.0
    python demo/app.py
"""
import os
import sys
import time
import html as html_lib

# Force unbuffered stdout so import progress is visible immediately
sys.stdout.reconfigure(line_buffering=True)

# Prevent proxy from intercepting Gradio's localhost health-check (Gradio 5)
os.environ["no_proxy"] = "localhost,127.0.0.1," + os.environ.get("no_proxy", "")
os.environ["NO_PROXY"] = "localhost,127.0.0.1," + os.environ.get("NO_PROXY", "")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

_t0 = time.time()
print(f"[{time.time()-_t0:5.1f}s] importing torch ...")
import torch
from torch.nn import functional as F
print(f"[{time.time()-_t0:5.1f}s] importing transformers ...")
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.cache_utils import DynamicCache
print(f"[{time.time()-_t0:5.1f}s] importing gradio ...")
import gradio as gr
from PIL import Image
print(f"[{time.time()-_t0:5.1f}s] imports done.")

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS    = os.path.join(ROOT, "model_weights")
DEMO_IMGS  = os.path.join(ROOT, "eval", "CXRTest_demo", "CXRTest")

MODELS = {
    "ECHO_block4   (distilled, 1 step/block, block=4)":  os.path.join(WEIGHTS, "ECHO_block4"),
    "ECHO_block8   (distilled, 1 step/block, block=8)":  os.path.join(WEIGHTS, "ECHO_block8"),
    "ECHO_Base_block4  (multi-step teacher, block=4)":   os.path.join(WEIGHTS, "ECHO_Base_block4"),
    "ECHO_Base_block8  (multi-step teacher, block=8)":   os.path.join(WEIGHTS, "ECHO_Base_block8"),
}

PROMPT_EN = (
    "Review this chest X-ray and write a report. "
    "Use this format: Findings: {}, Impression: {}."
)
PROMPT_ZH = (
    "这是一组胸部X光图像，请生成一份医学报告，包括所见和结论。"
    "以以下格式返回报告：所见：{} 结论：{}。"
)


# ── Model cache ────────────────────────────────────────────────────────────────
_cache: dict = {}


def load_model(model_dir: str):
    if model_dir in _cache:
        return _cache[model_dir]

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer  = AutoTokenizer.from_pretrained(model_dir,  trust_remote_code=True)

    if getattr(model.config, "mask_token_id", None):
        mask_id = model.config.mask_token_id
    elif getattr(tokenizer, "mask_token_id", None):
        mask_id = tokenizer.mask_token_id
    else:
        mask_id = tokenizer("<|MASK|>", add_special_tokens=False)["input_ids"][0]

    _cache[model_dir] = (model, processor, tokenizer, mask_id)
    return _cache[model_dir]


# ── Attention mask helper ──────────────────────────────────────────────────────
def make_block_mask(num_blocks: int, block_length: int, device):
    m = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    return (
        m.repeat_interleave(block_length, 0)
         .repeat_interleave(block_length, 1)
    )[None, None]   # (1, 1, T, T)


# ── Sampling ───────────────────────────────────────────────────────────────────
def sample(logits, temperature=0.0, top_k=0, top_p=1.0):
    bs, sl, vs = logits.shape
    v = logits.reshape(-1, vs)

    if temperature == 0:
        tok = torch.argmax(v, -1, keepdim=True)
        p   = F.softmax(v, -1).gather(-1, tok).squeeze(-1)
        return tok.squeeze(-1).view(bs, sl), p.view(bs, sl)

    v = v / temperature
    if top_k > 0:
        kv, _ = torch.topk(v, top_k)
        v = v.masked_fill(v < kv[:, -1:], float("-inf"))
    if top_p < 1.0:
        sv, si = torch.sort(v, descending=True)
        cp  = torch.cumsum(F.softmax(sv, -1), -1)
        sm  = cp > top_p
        sm[:, 1:] = sm[:, :-1].clone()
        sm[:, 0]  = False
        # sv is in sorted space; mask it there, then scatter back to vocab order
        v   = v.scatter(-1, si, sv.masked_fill(sm, float("-inf")))

    p   = F.softmax(v, -1)
    tok = torch.multinomial(p, 1)
    pr  = p.gather(-1, tok).squeeze(-1)
    return tok.squeeze(-1).view(bs, sl), pr.view(bs, sl)


# ── num_transfer_tokens for multi-step remasking ───────────────────────────────
def get_ntt(block_length: int, steps: int):
    base = block_length // steps
    rem  = block_length % steps
    t    = torch.zeros(steps, dtype=torch.int64) + base
    t[:rem] += 1
    return t


# ── HTML renderer ──────────────────────────────────────────────────────────────
MIDEA_BLUE      = "#3487C8"
MIDEA_BLUE_DARK = "#1a4a6e"
MIDEA_BLUE_LIGHT = "#e8f4fd"
MIDEA_BLUE_MID  = "#93c5e8"

CSS = """<style>
.eb{font-family:'Menlo','Consolas',monospace;font-size:14px;line-height:2.1;
    white-space:pre-wrap;word-break:break-word;padding:16px;
    background:#fafafa;border:1px solid #e5e7eb;border-radius:10px;min-height:80px}
.td{color:#1a1a1a}
.tn{background:#dbeafe;color:#1a4a6e;border-radius:4px;padding:1px 1px;font-weight:700;
    box-shadow:0 0 0 1px #3487C8}
.tm{color:#d1d5db;letter-spacing:-1px}
.sb{margin-top:10px;padding:8px 14px;border-radius:8px;font-size:12px;
    font-family:monospace;border:1px solid}
.si{background:#e8f4fd;border-color:#93c5e8;color:#1a4a6e}
.sf{background:#e0eefa;border-color:#3487C8;color:#1a4a6e;font-weight:600}
</style>"""


def render_html(old: str, new: str, n_masks: int, stat: str, final: bool = False) -> str:
    parts = [CSS, '<div class="eb">']

    if old:
        parts.append(f'<span class="td">{html_lib.escape(old)}</span>')
    if new:
        parts.append(f'<span class="tn">{html_lib.escape(new)}</span>')
    if n_masks > 0:
        shown = min(n_masks, 80)
        dots  = "…" if n_masks > 80 else ""
        parts.append(f'<span class="tm">{"▒" * shown}{dots}</span>')

    parts.append('</div>')
    cls = "sf" if final else "si"
    parts.append(f'<div class="sb {cls}">{html_lib.escape(stat)}</div>')
    return "".join(parts)


# ── Streaming generate ─────────────────────────────────────────────────────────
@torch.no_grad()
def echo_stream(
    model, inputs, mask_id, tokenizer,
    gen_length, block_length, denoising_steps,
    temperature, remasking_strategy, confidence_threshold,
    fused_decode=True,
):
    """Yield rendered HTML snapshots after every decoded block."""

    inp = inputs["input_ids"]
    pv  = inputs.get("pixel_values")
    igt = inputs.get("image_grid_thw")
    pvv = inputs.get("pixel_values_videos")
    vgt = inputs.get("video_grid_thw")

    P   = inp.shape[1]
    nb  = (P + gen_length + block_length - 1) // block_length
    TL  = nb * block_length

    x   = torch.full((1, TL), mask_id, dtype=torch.long, device=model.device)
    x[:, :P] = inp

    kv  = DynamicCache()
    gam = make_block_mask(nb, block_length, model.device)

    # ── Prefill ────────────────────────────────────────────────────────────────
    model(
        input_ids=inp, pixel_values=pv, image_grid_thw=igt,
        pixel_values_videos=pvv, video_grid_thw=vgt,
        attention_mask=gam[:, :, :P, :P],
        past_key_values=kv, use_cache=True, store_kv=True,
    )
    pb  = P // block_length
    if pb * block_length < P:
        kv.crop(pb * block_length)

    ntt       = get_ntt(block_length, denoising_steps)
    # Fused decode: ECHO distilled only (not Base), and only when steps=1
    use_fused = fused_decode and (denoising_steps == 1)

    t_total  = 0.0
    n_blocks = 0
    n_fwd    = 0
    prev_x   = None

    for blk in range(pb, nb):
        bs  = blk * block_length
        be  = bs  + block_length
        if bs >= TL:
            break

        cur = x[:, bs:be].clone()
        cp  = torch.arange(bs, be, device=model.device, dtype=torch.long)
        mi  = (cur == mask_id)

        # ── Prompt-only block (no masks) ────────────────────────────────────────
        if mi.sum() == 0:
            model(
                input_ids=cur,
                attention_mask=gam[:, :, bs:be, :be],
                cache_position=cp,
                past_key_values=kv, use_cache=True, store_kv=True,
            )
            n_fwd += 1
            prev_x = cur.clone()
            x[:, bs:be] = cur
            continue

        # ── Timing start ───────────────────────────────────────────────────────
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if use_fused:
            # ── ECHO: single forward decodes the entire block ──────────────────
            if prev_x is not None:
                # Fused: write prev block's KV + denoise cur block together
                cpos = torch.arange(bs - block_length, be, device=model.device, dtype=torch.long)
                out  = model(
                    input_ids=torch.cat([prev_x, cur], 1),
                    attention_mask=gam[:, :, bs - block_length:be, :be],
                    cache_position=cpos,
                    past_key_values=kv, use_cache=True,
                    store_kv=True, store_kv_len=block_length,
                )
                logits = out.logits[:, -block_length:]
            else:
                out = model(
                    input_ids=cur,
                    attention_mask=gam[:, :, bs:be, :be],
                    cache_position=cp,
                    past_key_values=kv, use_cache=True, store_kv=False,
                )
                logits = out.logits
            n_fwd += 1

            x0, _ = sample(logits, temperature)
            cur[mi] = x0[mi]

        else:
            # ── Multi-step: progressively unmask within the block ──────────────
            for step in range(denoising_steps + 1):
                mi = (cur == mask_id)
                if mi.sum() == 0:
                    # All positions resolved — write KV and stop stepping
                    model(
                        input_ids=cur,
                        attention_mask=gam[:, :, bs:be, :be],
                        cache_position=cp,
                        past_key_values=kv, use_cache=True, store_kv=True,
                    )
                    n_fwd += 1
                    break

                out = model(
                    input_ids=cur,
                    attention_mask=gam[:, :, bs:be, :be],
                    cache_position=cp,
                    past_key_values=kv, use_cache=True, store_kv=False,
                )
                n_fwd  += 1
                logits  = out.logits
                x0, x0p = sample(logits, temperature)

                # Remasking: pick which positions to accept this step
                k  = int(ntt[step]) if step < len(ntt) else int(mi.sum())
                ti = torch.zeros_like(cur, dtype=torch.bool)

                if remasking_strategy == "sequential":
                    for j in range(cur.shape[0]):
                        if mi[j].any():
                            mp = mi[j].nonzero(as_tuple=True)[0]
                            ti[j, mp[:min(k, len(mp))]] = True

                elif remasking_strategy == "low_confidence_dynamic":
                    conf = torch.where(mi, x0p, torch.full_like(x0p, float("-inf")))
                    for j in range(cur.shape[0]):
                        hc = conf[j] > confidence_threshold
                        if hc.sum() >= k:
                            ti[j] = hc
                        else:
                            nm = int(mi[j].sum())
                            kk = min(k, nm)
                            if kk > 0:
                                _, idx = torch.topk(conf[j], kk)
                                ti[j, idx] = True

                else:  # low_confidence_static
                    conf = torch.where(mi, x0p, torch.full_like(x0p, float("-inf")))
                    for j in range(cur.shape[0]):
                        nm = int(mi[j].sum())
                        kk = min(k, nm)
                        if kk > 0 and not torch.all(torch.isinf(conf[j])):
                            _, idx = torch.topk(conf[j], kk)
                            ti[j, idx] = True

                cur[ti] = x0[ti]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed  = time.perf_counter() - t0
        t_total += elapsed
        n_blocks += 1

        prev_x = cur.clone()
        x[:, bs:be] = cur

        # ── Build display text ─────────────────────────────────────────────────
        # old_text: everything generated before this block
        old_ids  = [t for t in x[0, P:bs].tolist() if t != mask_id]
        old_text = tokenizer.decode(old_ids, skip_special_tokens=True)

        # new_text: the block we just decoded (non-mask tokens only)
        new_ids  = [t for t in cur[0].tolist() if t != mask_id]
        new_text = tokenizer.decode(new_ids, skip_special_tokens=True)

        # Count remaining masks after this block
        n_masks  = int(x[0, be:].eq(mask_id).sum())

        tpf  = block_length / denoising_steps
        step_label = "1 step" if denoising_steps == 1 else f"{denoising_steps} steps"
        stat = (
            f"⚡ Block {n_blocks}  |  {elapsed * 1000:.0f} ms  "
            f"|  {block_length} tokens / {step_label}  "
            f"|  {tpf:.1f} tok/forward  "
            f"|  {block_length / elapsed:.0f} tok/s  "
            f"|  total {t_total:.2f}s"
        )
        yield render_html(old_text, new_text, n_masks, stat)

        # ── EOS check (skip positions that overlap with prompt) ────────────────
        if tokenizer.eos_token_id:
            valid_start = max(0, P - bs)
            if (cur[:, valid_start:] == tokenizer.eos_token_id).any():
                break

    # ── Final output ───────────────────────────────────────────────────────────
    gen_ids = x[0, P:]
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in gen_ids:
        gen_ids = gen_ids[:(gen_ids == tokenizer.eos_token_id).nonzero()[0][0]]
    if mask_id in gen_ids:
        gen_ids = gen_ids[:(gen_ids == mask_id).nonzero()[0][0]]

    final_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    n_tok      = len(gen_ids)
    ar_fwd     = n_tok   # AR needs one forward per token
    speedup    = ar_fwd / n_fwd if n_fwd else 0

    final_stat = (
        f"✅  Done  |  {n_tok} tokens  |  {n_blocks} blocks  "
        f"|  {t_total:.2f}s  |  {n_tok / t_total:.0f} tok/s  ||  "
        f"ECHO: {n_fwd} forward passes  ·  AR equiv: {ar_fwd}  "
        f"·  ≈ {speedup:.1f}× fewer forwards"
    )
    yield render_html(final_text, "", 0, final_stat, final=True)


# ── Gradio callback ────────────────────────────────────────────────────────────
def generate(image, model_key, prompt, denoising_steps,
             gen_length, temperature, remasking, conf_thr):

    model_dir    = MODELS[model_key]
    block_length  = 8 if "block8" in model_dir else 4
    is_distilled  = "Base" not in model_key   # ECHO_block4/8 vs ECHO_Base_block4/8

    if image is None:
        yield render_html("", "", 0, "⚠️  Please upload a chest X-ray image.", final=True)
        return

    yield render_html("", "", 0, "⏳  Loading model (first run may take ~30 s) …")
    try:
        model, processor, tokenizer, mask_id = load_model(model_dir)
    except Exception as e:
        yield render_html("", "", 0, f"❌  Model load error: {e}", final=True)
        return

    pil = (
        Image.fromarray(image).convert("RGB")
        if not isinstance(image, Image.Image)
        else image
    )
    msgs   = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text   = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=[pil],
        max_pixels=2_250_000, padding=True, return_tensors="pt",
    ).to(model.device)

    yield render_html(
        "", "", gen_length,
        f"⚡  Decoding  (block_length={block_length}, steps/block={denoising_steps}) …",
    )
    try:
        for html in echo_stream(
            model, inputs, mask_id, tokenizer,
            gen_length=int(gen_length),
            block_length=block_length,
            denoising_steps=int(denoising_steps),
            temperature=float(temperature),
            remasking_strategy=remasking,
            confidence_threshold=float(conf_thr),
            fused_decode=is_distilled,   # fused path: ECHO distilled only
        ):
            yield html
    except Exception:
        import traceback
        yield f"<pre style='color:red'>{html_lib.escape(traceback.format_exc())}</pre>"


# ── Demo images ────────────────────────────────────────────────────────────────
def _collect(root, n=10):
    imgs = []
    for dp, _, files in os.walk(root):
        for f in sorted(files):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(dp, f)
            # Exclude lateral views (identified by 'lateral' in path)
            if "lateral" in path.lower():
                continue
            imgs.append(path)
            if len(imgs) >= n:
                return imgs
    return imgs


DEMO_IMAGE_PATHS = _collect(DEMO_IMGS)


# ── UI ─────────────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="ECHO – One-Step Per Block Streaming Demo",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:

    gr.Markdown("""
# ⚡ ECHO — One-Step Per Block Streaming Decoding
**Efficient Chest X-ray Report Generation with Block Diffusion Distillation (DCD)**

Each block of tokens is decoded in a **single forward pass**, achieving 8× fewer model calls than an autoregressive baseline.
<span style="background:#dbeafe;color:#1a4a6e;border-radius:4px;padding:2px 8px;font-weight:700;box-shadow:0 0 0 1px #3487C8">blue tokens</span> = newly decoded this block &nbsp;·&nbsp;
<span style="color:#d1d5db">▒▒▒</span> = positions still masked
""")

    with gr.Row():
        # ── Left: controls ────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=360):
            image_in = gr.Image(label="Chest X-ray Image", type="numpy", height=280)

            model_sel = gr.Dropdown(
                choices=list(MODELS.keys()),
                value=list(MODELS.keys())[0],
                label="Model",
            )

            with gr.Row():
                steps_sl = gr.Slider(
                    1, 16, value=1, step=1,
                    label="Denoising Steps / Block",
                    info="Auto-set on model change. 1 = ECHO distilled; ≥4 = Base multi-step",
                )
                gen_len = gr.Slider(256, 2048, value=1024, step=128, label="Max Gen Tokens")

            prompt_in = gr.Textbox(label="Prompt", value=PROMPT_EN, lines=3)
            with gr.Row():
                gr.Button("🇺🇸 EN preset", size="sm").click(lambda: PROMPT_EN, outputs=prompt_in)
                gr.Button("🇨🇳 ZH preset", size="sm").click(lambda: PROMPT_ZH, outputs=prompt_in)

            with gr.Accordion("Advanced", open=False):
                temp_sl  = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Temperature  (0 = greedy)")
                remask   = gr.Dropdown(
                    ["low_confidence_dynamic", "low_confidence_static", "sequential"],
                    value="low_confidence_dynamic",
                    label="Remasking Strategy",
                )
                conf_thr = gr.Slider(0.5, 1.0, value=0.85, step=0.05, label="Confidence Threshold")

            gen_btn = gr.Button("⚡  Generate", variant="primary", size="lg")

        # ── Right: streaming output ────────────────────────────────────────────
        with gr.Column(scale=1):
            out_html = gr.HTML(
                value=(
                    "<p style='color:#9ca3af;font-style:italic;padding:16px'>"
                    "Select or upload an image, then click ⚡ Generate to watch "
                    "one-step-per-block streaming decoding.</p>"
                ),
                label="Streaming Output",
            )

    # ── Demo images — full-width gallery below the main row ───────────────────
    if DEMO_IMAGE_PATHS:
        gr.Examples(
            examples=[[p] for p in DEMO_IMAGE_PATHS],
            inputs=[image_in],
            label="Demo Images — frontal views only, click to load",
            examples_per_page=10,
        )

    # Auto-update denoising steps when model changes
    def _default_steps(model_key):
        """Distilled ECHO → 1 step; Base (teacher) → block_length steps."""
        is_base = "Base" in model_key
        bl      = 8 if "block8" in model_key else 4
        return gr.update(value=bl if is_base else 1)

    model_sel.change(fn=_default_steps, inputs=[model_sel], outputs=[steps_sl])

    gen_btn.click(
        fn=generate,
        inputs=[image_in, model_sel, prompt_in, steps_sl, gen_len, temp_sl, remask, conf_thr],
        outputs=[out_html],
    )

if __name__ == "__main__":
    # Disable Gradio analytics (avoids slow external requests on servers)
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    print(f"[{time.time()-_t0:5.1f}s] launching gradio on 0.0.0.0:7837 ...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7837,
        share=False,
        ssr_mode=False,          # Fix Gradio 5 503 in proxy/server environments
        allowed_paths=[ROOT],    # Allow serving demo images from ECHO_release tree
    )
