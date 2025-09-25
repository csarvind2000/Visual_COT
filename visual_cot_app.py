#!/usr/bin/env python3
"""
Visual CoT (Chain-of-Thought) — Minimal, backend-agnostic prototype
------------------------------------------------------------------
Goal: given an image and a user question, produce a *visual* CoT consisting of:
  • numbered textual reasoning steps (1..N)
  • per-step *optional* regions (boxes) drawn on the image
  • a storyboard (small panels) that shows the image with overlays for each step
  • a final short answer

This app is designed to be *backend-agnostic*: you can plug in OpenAI, Ollama
(local LLaVA / Qwen2-VL / MiniCPM-V), or HF transformers. By default, the LLM
call is a stub that returns deterministic demo steps if no key/model is set.

Optional perception helpers (auto-detected if installed):
  • ultralytics (YOLO) for quick object proposals
  • groundingdino + sam2 for better region proposals (heavy; optional)

Install (recommended):
  pip install gradio pillow numpy matplotlib pydantic ollama
  # Optional proposals:
  pip install ultralytics

Pull models & run Ollama:
  ollama serve  # in a separate terminal
  ollama pull llava:latest
  # optionally: ollama pull llama3.2-vision:latest

Run:
  python visual_cot_app.py --port 7860

Notes:
  • This is a *prototype* focused on UX/data-flow; swap in your preferred VLM
    inside `call_llm_visual_cot()` and map its output into the schema below.
  • Regions are in [x, y, w, h] (absolute px) in the *original image size*.
"""

from __future__ import annotations
import os, io, json, math, argparse, textwrap
from typing import List, Optional, Dict, Any, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gradio as gr

# ---------------------------- Visual-CoT Schema ----------------------------- #
# Robust model classes with pydantic if available; fall back to dataclasses.
try:
    from pydantic import BaseModel, Field
    PydanticAvailable = True

    class CoTStep(BaseModel):
        idx: int = Field(..., description="1-based step index")
        text: str = Field(..., description="Short, bullet-style reasoning step")
        region: Optional[List[int]] = Field(
            default=None, description="Optional [x,y,w,h] in *original* image px"
        )

    class CoTResult(BaseModel):
        steps: List[CoTStep]
        final_answer: str

except Exception:
    from dataclasses import dataclass, asdict
    PydanticAvailable = False

    @dataclass
    class CoTStep:
        idx: int
        text: str
        region: Optional[List[int]] = None

    @dataclass
    class CoTResult:
        steps: List[CoTStep]
        final_answer: str

        # Provide a pydantic-like method for downstream code
        def model_dump(self) -> Dict[str, Any]:
            return {
                "steps": [asdict(s) for s in self.steps],
                "final_answer": self.final_answer,
            }

# --------------------------- Utility: draw overlays ------------------------- #

def _text_size(draw: ImageDraw.ImageDraw, text: str, font: Optional[ImageFont.ImageFont] = None) -> Tuple[int, int]:
    """Get text size with Pillow, supporting older versions."""
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    # Fallbacks
    try:
        if font is None:
            font = ImageFont.load_default()
        return draw.textlength(text, font=font), font.getbbox(text)[3]  # may fail on very old Pillow
    except Exception:
        # last resort
        return (8 * len(text), 14)

def draw_box(img: Image.Image, box: List[int], label: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    x, y, w, h = box
    x2, y2 = x + w, y + h
    draw.rectangle([x, y, x2, y2], outline=(255, 0, 0), width=3)

    # label background
    font = ImageFont.load_default()
    text = str(label)
    tw, th = _text_size(draw, text, font)
    pad = 4
    bg_x1, bg_y1 = x, max(0, y - (th + 2 * pad))
    bg_x2, bg_y2 = x + tw + 2 * pad, bg_y1 + th + 2 * pad
    draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(255, 0, 0))
    draw.text((bg_x1 + pad, bg_y1 + pad), text, fill=(255, 255, 255), font=font)
    return img

def make_storyboard(img: Image.Image, cot: CoTResult, max_cols: int = 4) -> Image.Image:
    # Create a grid image of overlays per step
    W, H = img.size
    steps = [s for s in cot.steps]
    n = len(steps)
    if n == 0:
        return img.copy()
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)
    panel_w, panel_h = W, H
    board = Image.new("RGB", (cols * panel_w, rows * panel_h), color=(255, 255, 255))
    for i, step in enumerate(steps):
        rimg = img.copy()
        if getattr(step, "region", None):
            rimg = draw_box(rimg, step.region, f"{step.idx}")
        row = i // cols
        col = i % cols
        board.paste(rimg, (col * panel_w, row * panel_h))
    return board

# ----------------------------- Perception (optional) ----------------------- #

def quick_object_proposals(img: Image.Image, max_props: int = 5) -> List[List[int]]:
    """Return coarse proposals [x,y,w,h] using ultralytics if available; else empty."""
    try:
        from ultralytics import YOLO  # type: ignore
        model = YOLO("yolov8n.pt")
        res = model(img, verbose=False)[0]
        props: List[List[int]] = []
        for b in res.boxes.xyxy.cpu().numpy().tolist()[:max_props]:
            x1, y1, x2, y2 = map(int, b)
            props.append([x1, y1, x2 - x1, y2 - y1])
        return props
    except Exception:
        return []

# ----------------------------- LLM / VLM backend --------------------------- #
VISUAL_COT_SYS_PROMPT = """
You are a meticulous visual reasoning assistant. Given a user QUESTION and a JSON list of
OBJECT PROPOSALS (candidate regions in [x,y,w,h] on the original image size), produce a
short CHAIN OF THOUGHT as a *numbered list* of steps (3–7 concise steps). Each step may
OPTIONALLY reference one region (x,y,w,h). Conclude with a short FINAL_ANSWER.

Return STRICT JSON only with fields: {
  "steps": [ {"idx": 1, "text": "...", "region": [x,y,w,h] | null }, ...],
  "final_answer": "..."
}
No preamble, no backticks, no markdown. Keep steps short (<= 20 words).
""".strip()

def call_llm_visual_cot(
    question: str,
    img: Image.Image,
    proposals: List[List[int]],
    backend: str = "ollama",
    ollama_model: str = "llava:latest",
) -> CoTResult:
    """Return a Visual-CoT using Ollama (LLaVA / Llama3.2-Vision). 'demo' backend is offline fallback."""
    # DEMO fallback (no model call)
    if backend == "demo":
        W, H = img.size
        props = proposals[:3] if proposals else [[int(0.1 * W), int(0.1 * H), int(0.3 * W), int(0.3 * H)]]
        steps = [
            {"idx": 1, "text": "Scan the scene and locate salient objects.", "region": props[0]},
            {"idx": 2, "text": "Focus on region likely answering the question.", "region": props[0]},
            {"idx": 3, "text": "Use context to confirm the conclusion.", "region": None},
        ]
        return CoTResult(steps=[CoTStep(**s) for s in steps], final_answer="Demo answer.")

       # OLLAMA backend
    try:
        import base64, tempfile, os
        from io import BytesIO
        import ollama  # type: ignore

        # Encode image once
        buf = BytesIO()
        img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()
        b64 = base64.b64encode(raw_bytes).decode("utf-8")
        data_uri = f"data:image/png;base64,{b64}"

        sys_prompt = VISUAL_COT_SYS_PROMPT
        user_prompt = f"{sys_prompt}\nQUESTION:{question}\nPROPOSALS:{json.dumps(proposals)}"

        def _try_chat(images_payload):
            """Try an ollama.chat call with a given images payload format."""
            return ollama.chat(
                model=ollama_model,
                messages=[{
                    "role": "user",
                    "content": user_prompt,
                    "images": images_payload,  # list[str]
                }],
                options={"temperature": 0.2},
            )

        resp = None
        last_err = None

        # 1) Try data URI (some versions accept this)
        try:
            resp = _try_chat([data_uri])
        except Exception as e:
            last_err = e
            resp = None

        # 2) Try raw base64 (most versions expect this)
        if resp is None:
            try:
                resp = _try_chat([b64])
            except Exception as e:
                last_err = e
                resp = None

        # 3) Try a temp file path (older builds accept a path string)
        temp_path = None
        if resp is None:
            try:
                fd, temp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                with open(temp_path, "wb") as f:
                    f.write(raw_bytes)
                resp = _try_chat([temp_path])
            except Exception as e:
                last_err = e
                resp = None
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

        if resp is None:
            raise RuntimeError(f"Ollama image payload failed: {last_err!r}")

        txt = (resp.get("message") or {}).get("content", "")
        if not txt:
            return CoTResult(
                steps=[CoTStep(idx=1, text="Model returned empty response.", region=None)],
                final_answer="",
            )

        data = json.loads(extract_json(txt))
        steps_raw = data.get("steps", []) or []
        steps = [CoTStep(**s) for s in steps_raw] if steps_raw else [
            CoTStep(idx=1, text="No steps returned by model.", region=None)
        ]
        ans = data.get("final_answer", "")
        return CoTResult(steps=steps, final_answer=ans)

    except Exception as e:
        return CoTResult(
            steps=[CoTStep(idx=1, text=f"Ollama backend error: {e}", region=None)],
            final_answer="",
        )


# ----------------------------- Prompt helpers ------------------------------- #

def extract_json(text: str) -> str:
    """Extract first JSON block from text; fallback to whole text."""
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        return text[s : e + 1]
    return text

# ----------------------------- Gradio UI ----------------------------------- #

def _sample_image_and_question(name: str) -> Tuple[Image.Image, str]:
    W, H = 640, 400
    img = Image.new("RGB", (W, H), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    if name == "Street scene (synthetic)":
        d.rectangle([0, H * 0.6, W, H], fill=(210, 210, 210))  # road
        d.rectangle([80, 220, 140, 320], outline=(0, 0, 255), width=4)  # boy
        d.text((80, 200), "boy", fill=(0, 0, 0))
        d.ellipse([220, 250, 270, 300], outline=(150, 75, 0), width=4)  # dog
        d.text((220, 230), "dog", fill=(0, 0, 0))
        d.rectangle([380, 240, 520, 300], outline=(255, 0, 0), width=4)  # car
        d.text((400, 220), "car", fill=(0, 0, 0))
        q = "Which animal is closest to the boy?"
        return img, q
    elif name == "Fruits on table (synthetic)":
        d.rectangle([0, H * 0.65, W, H], fill=(220, 200, 170))  # table
        d.ellipse([120, 240, 180, 300], outline=(200, 0, 0), width=4)  # apple
        d.text((125, 220), "apple", fill=(0, 0, 0))
        d.arc([240, 240, 320, 320], start=20, end=160, fill=(200, 200, 0), width=6)  # banana
        d.text((245, 220), "banana", fill=(0, 0, 0))
        d.ellipse([380, 250, 430, 300], outline=(255, 140, 0), width=4)  # orange
        d.text((380, 230), "orange", fill=(0, 0, 0))
        q = "How many different fruits are on the table?"
        return img, q
    else:
        return Image.new("RGB", (W, H), color=(245, 245, 245)), ""

def load_sample(name: str):
    img, q = _sample_image_and_question(name)
    return img, q

def pipeline(image: Image.Image, question: str, backend: str, ollama_model: str):
    if image is None or not question:
        return None, "Please provide an image and a question.", None, None

    image = image.convert("RGB")

    # 1) proposals
    proposals = quick_object_proposals(image)

    # 2) LLM/VLM CoT
    cot = call_llm_visual_cot(question, image, proposals, backend=backend, ollama_model=ollama_model)

    # 3) Compose overlays
    img_all = image.copy()
    for step in cot.steps:
        if getattr(step, "region", None):
            img_all = draw_box(img_all, step.region, str(step.idx))

    # 4) Storyboard
    board = make_storyboard(image, cot, max_cols=4)

    # 5) Dump JSON CoT for debugging/export
    if hasattr(cot, "model_dump"):
        cot_json_dict = cot.model_dump()  # pydantic or our fallback
    elif hasattr(cot, "dict"):
        cot_json_dict = cot.dict()
    else:
        cot_json_dict = {
            "steps": [{"idx": s.idx, "text": s.text, "region": getattr(s, "region", None)} for s in cot.steps],
            "final_answer": getattr(cot, "final_answer", ""),
        }

    cot_json = json.dumps(cot_json_dict, indent=2, ensure_ascii=False)
    return img_all, cot_json, board, cot.final_answer

def build_ui():
    with gr.Blocks(title="Visual CoT Prototype", css=".small {font-size: 12px}") as demo:
        gr.Markdown("""
        # Visual Chain-of-Thought
        Upload an image and ask a question. The app will produce a *visual* chain of thought
        (numbered steps with optional regions) and a final short answer.
        """)
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="pil", label="Image")
                q = gr.Textbox(label="Question", placeholder="e.g., Which animal is closest to the boy?")
                sample_dd = gr.Dropdown(["Street scene (synthetic)", "Fruits on table (synthetic)", "None"], value="Street scene (synthetic)", label="Load sample")
                sample_btn = gr.Button("Insert sample above")
                backend = gr.Radio(["ollama", "demo"], value="ollama", label="Backend")
                ollama_model = gr.Dropdown(choices=["llava:latest", "llava:13b", "llama3.2-vision:latest"], value="llava:latest", label="Ollama vision model")
                btn = gr.Button("Run Visual CoT", variant="primary")
                sample_btn.click(load_sample, inputs=[sample_dd], outputs=[img, q])
            with gr.Column():
                out_img = gr.Image(type="pil", label="Overlaid steps (all boxes)")
                out_json = gr.Code(language="json", label="CoT JSON")
                out_board = gr.Image(type="pil", label="Storyboard (per step)")
                out_answer = gr.Textbox(label="Final Answer")
        btn.click(pipeline, inputs=[img, q, backend, ollama_model], outputs=[out_img, out_json, out_board, out_answer])
        gr.Markdown("""
        **Tips**
        - Switch *Backend* between `demo` (offline) and `ollama` (vision model required).
        - Regions are optional; the model uses them only when helpful.
        - Export the JSON to build training/eval datasets for Visual-CoT.
        """, elem_classes=["small"])
    return demo

# ----------------------------- Main ---------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=args.port)
