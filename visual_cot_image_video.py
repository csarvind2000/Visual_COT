#!/usr/bin/env python3
"""
Visual CoT — Image & Video (YOLO grounding + SAM in Chat, Ollama VLM)
- Segmentation happens only when the user asks in chat (e.g., "segment boy").
- Natural language grounding: phrase → YOLO detections (with alias map) → SAM mask.
- CoT provides named regions ("step N") to target boxes/segments/explanations.
"""

from __future__ import annotations
import os, io, re, json, math, argparse, tempfile, hashlib
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageEnhance
import gradio as gr
import cv2

# Optional torch (for SAM Predictor speed & inference_mode)
try:
    import torch
except Exception:
    torch = None

# ======================= Data Models =======================
try:
    from pydantic import BaseModel, Field
    class CoTStep(BaseModel):
        idx: int = Field(...)
        text: str = Field(...)
        region: Optional[List[int]] = Field(default=None)  # [x,y,w,h] in px
    class CoTResult(BaseModel):
        steps: List[CoTStep]
        final_answer: str
except Exception:
    from dataclasses import dataclass, asdict
    @dataclass
    class CoTStep:
        idx: int
        text: str
        region: Optional[List[int]] = None
    @dataclass
    class CoTResult:
        steps: List[CoTStep]
        final_answer: str
        def model_dump(self): return {"steps":[asdict(s) for s in self.steps],"final_answer":self.final_answer}

# ======================= Drawing ===========================
def _text_size(draw: ImageDraw.ImageDraw, text: str, font=None) -> Tuple[int,int]:
    if hasattr(draw, "textbbox"):
        l,t,r,b = draw.textbbox((0,0), text, font=font); return r-l, b-t
    font = font or ImageFont.load_default()
    try: return draw.textlength(text, font=font), font.getbbox(text)[3]
    except Exception: return (8*len(text), 14)

def draw_box(img: Image.Image, box: List[int], label: str) -> Image.Image:
    d = ImageDraw.Draw(img)
    x,y,w,h = box; x2,y2 = x+w, y+h
    d.rectangle([x,y,x2,y2], outline=(255,0,0), width=3)
    font = ImageFont.load_default(); tw,th = _text_size(d, label, font); pad=4
    d.rectangle([x, max(0,y-(th+2*pad)), x+tw+2*pad, max(0,y-(th+2*pad))+th+2*pad], fill=(255,0,0))
    d.text((x+pad, max(0,y-(th+2*pad))+pad), label, fill=(255,255,255), font=font)
    return img

def annotate_text(img: Image.Image, text: str, xy: Tuple[int,int]) -> Image.Image:
    d = ImageDraw.Draw(img); font = ImageFont.load_default(); tw,th = _text_size(d,text,font); pad=4
    x,y = xy; d.rectangle([x,y,x+tw+2*pad,y+th+2*pad], fill=(0,0,0)); d.text((x+pad,y+pad), text, fill=(255,255,255), font=font)
    return img

def overlay_masks(img: Image.Image, masks: List[np.ndarray], alpha: float = 0.45) -> Image.Image:
    if not masks: return img
    base = np.array(img); over = base.copy()
    for m in masks:
        m = m.astype(bool)
        color = np.array([0,255,0], dtype=np.uint8)
        over[m] = (over[m]*(1-alpha) + color*alpha).astype(np.uint8)
    return Image.fromarray(over)

def make_storyboard(img: Image.Image, cot: CoTResult, max_cols=4) -> Image.Image:
    W,H = img.size; steps=list(cot.steps); n=len(steps)
    if n==0: return img.copy()
    cols=min(max_cols,n); rows=(n+cols-1)//cols
    board = Image.new("RGB", (cols*W, rows*H), (255,255,255))
    for i,s in enumerate(steps):
        rimg = img.copy()
        if getattr(s,"region",None): rimg = draw_box(rimg, s.region, f"{s.idx}")
        row,col = divmod(i,cols); board.paste(rimg, (col*W,row*H))
    return board

# ======================= YOLO (Ultralytics) =================
_YOLO_MODEL = None

def _load_yolo():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        from ultralytics import YOLO  # type: ignore
        _YOLO_MODEL = YOLO("yolov8n.pt")
    return _YOLO_MODEL

def proposals_yolo(img: Image.Image, max_props=8) -> List[List[int]]:
    try:
        res = _load_yolo()(img, verbose=False)[0]
        props=[]
        for b in res.boxes.xyxy.cpu().numpy().tolist()[:max_props]:
            x1,y1,x2,y2 = map(int,b); props.append([x1,y1,x2-x1,y2-y1])
        return props
    except Exception:
        return []

def yolo_detect(img: Image.Image, conf=0.25, max_det=20) -> List[Dict[str,Any]]:
    """Return list of {xywh, name, conf} using YOLO cls names."""
    try:
        res = _load_yolo()(img, verbose=False, conf=conf, max_det=max_det)[0]
        names = res.names
        boxes = res.boxes.xyxy.cpu().numpy()
        clsi  = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        out=[]
        for (x1,y1,x2,y2),ci,cf in zip(boxes,clsi,confs):
            x1,y1,x2,y2 = map(int,[x1,y1,x2,y2]); out.append({
                "xywh":[x1,y1,x2-x1,y2-y1],
                "name": names.get(ci, str(ci)),
                "conf": float(cf)
            })
        return out
    except Exception:
        return []

# ======================= SAM (on-demand, chat) ==============
_SAM_MODEL_CACHE: Dict[Tuple[str,str,str], Any] = {}
_SAM_PRED_CACHE: Dict[Tuple[str,str,str,str], Any] = {}

def _device_str() -> str:
    if torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def _get_sam_model(model_type: str, ckpt: str, device: str):
    from segment_anything import sam_model_registry  # type: ignore
    key=(model_type,ckpt,device)
    if key in _SAM_MODEL_CACHE: return _SAM_MODEL_CACHE[key]
    sam = sam_model_registry[model_type](checkpoint=ckpt)
    if device=="cuda": sam.to(device)
    _SAM_MODEL_CACHE[key]=sam; return sam

def _image_signature(img: Image.Image) -> str:
    buf=io.BytesIO(); img.save(buf, format="PNG"); return hashlib.sha1(buf.getvalue()).hexdigest()

def _get_predictor_for_image(img: Image.Image, model_type: str, ckpt: str, device: str):
    from segment_anything import SamPredictor  # type: ignore
    sig=_image_signature(img); key=(model_type,ckpt,device,sig)
    if key in _SAM_PRED_CACHE: return _SAM_PRED_CACHE[key]
    sam=_get_sam_model(model_type, ckpt, device); pred=SamPredictor(sam)
    pred.set_image(np.array(img))
    _SAM_PRED_CACHE[key]=pred; return pred

def sam_segment_box(img: Image.Image, box_xywh: List[int], ckpt: str, model_type="vit_h", device=None) -> Optional[np.ndarray]:
    if not ckpt: return None
    try:
        from segment_anything import SamPredictor  # noqa
    except Exception:
        return None
    dev=device or _device_str()
    pred=_get_predictor_for_image(img, model_type, ckpt, dev)
    np_img=np.array(img); H,W=np_img.shape[:2]
    x,y,w,h=box_xywh; xyxy=np.array([[x,y,x+w,y+h]],dtype=np.float32)
    xyxy_t=torch.from_numpy(xyxy) if torch is not None else xyxy
    if torch is not None and dev=="cuda": xyxy_t=xyxy_t.to(dev)
    if torch is not None:
        with torch.inference_mode():
            masks,scores,_=pred.predict_torch(point_coords=None, point_labels=None, boxes=xyxy_t, multimask_output=True)
        mi=int(torch.argmax(scores[0]).item()); mask=masks[0,mi].detach().cpu().numpy().astype(bool)
    else:
        # very slow path if torch not present; bail out
        return None
    return mask

# ======================= Prompting Helpers ==================
VISUAL_COT_SYS_PROMPT = """You are a meticulous visual reasoning assistant. Given a user QUESTION and a JSON list of
OBJECT PROPOSALS (candidate regions in [x,y,w,h] on the original image size), produce a short CHAIN OF THOUGHT as a numbered list (3–7 concise steps). Each step may OPTIONALLY reference one region (x,y,w,h). Conclude with a short FINAL_ANSWER.
Return STRICT JSON only: {"steps":[{"idx":1,"text":"...","region":[x,y,w,h]|null},...], "final_answer":"..."}""".strip()

VISUAL_COT_SYS_PROMPT_STRICT = """You MUST use the provided PROPOSALS. At least 3 steps must include a non-null region picked EXACTLY from PROPOSALS. Return STRICT JSON only as specified before.""".strip()

def extract_json(text: str) -> str:
    s=text.find("{"); e=text.rfind("}")
    return text[s:e+1] if s!=-1 and e!=-1 and e>s else text

# Accept normalized floats, xyxy, indices, etc.
def _clip_int(v, lo, hi): return int(max(lo, min(hi, round(float(v)))))

def _coerce_region_any(
    r: Any,
    W: int,
    H: int,
    proposals: List[List[int]] | None = None,
) -> Optional[List[int]]:
    """
    Accepts:
      - [x,y,w,h] or [x1,y1,x2,y2] (pixels or normalized 0..1)
      - integer/string index referring to proposals (0- or 1-based)
    Returns:
      - [x,y,w,h] in integer pixels, or None if invalid.
    """
    # proposal index?
    if isinstance(r, (int, float)) and proposals:
        idx = int(r)
        if   1 <= idx <= len(proposals): return proposals[idx - 1][:]
        elif 0 <= idx <  len(proposals): return proposals[idx][:]
    if isinstance(r, str) and proposals:
        m = re.search(r"(\d+)", r)
        if m:
            idx = int(m.group(1))
            if   1 <= idx <= len(proposals): return proposals[idx - 1][:]
            elif 0 <= idx <  len(proposals): return proposals[idx][:]

    # region list?
    if isinstance(r, (list, tuple)) and len(r) == 4:
        vals = [_safe_float(v) for v in r]
        if any(v is None for v in vals):
            return None  # contains null/NaN etc.

        x1, y1, a, b = vals

        # normalized?
        if all(0.0 <= v <= 1.0 for v in vals):
            x = _clip_int(x1 * W, 0, W - 1)
            y = _clip_int(y1 * H, 0, H - 1)
            w = _clip_int(a  * W, 1, W)
            h = _clip_int(b  * H, 1, H)
        else:
            # decide xyxy vs xywh
            if a > x1 and b > y1:  # xyxy
                x = _clip_int(x1, 0, W - 1)
                y = _clip_int(y1, 0, H - 1)
                w = _clip_int(a - x1, 1, W)
                h = _clip_int(b - y1, 1, H)
            else:                  # xywh
                x = _clip_int(x1, 0, W - 1)
                y = _clip_int(y1, 0, H - 1)
                w = _clip_int(a,  1, W)
                h = _clip_int(b,  1, H)

        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))
        return [x, y, w, h]

    # nothing valid
    return None

def _sanitize_list(sr_list):
    out = []
    for i, s in enumerate(sr_list or []):
        idx  = _safe_int(s.get("idx"), i + 1)
        text = str(s.get("text", "") or "").strip()
        r    = _coerce_region_any(s.get("region", None), W, H, proposals=props)
        out.append({"idx": idx, "text": text, "region": r})
    return out


# ======================= Ollama ============================
def ollama_chat_visual(prompt: str, img: Image.Image, model: str) -> str:
    import ollama, base64
    from io import BytesIO
    buf=BytesIO(); img.save(buf, format="PNG"); raw=buf.getvalue()
    b64=base64.b64encode(raw).decode("utf-8")
    data_uri=f"data:image/png;base64,{b64}"
    def _try(payload):
        return ollama.chat(model=model, messages=[{"role":"user","content":prompt,"images":payload}], options={"temperature":0.2})
    for payload in ([data_uri],[b64]):
        try:
            resp=_try(payload); return (resp.get("message") or {}).get("content","")
        except Exception: pass
    # temp file fallback
    path=None
    try:
        fd,path=tempfile.mkstemp(suffix=".png"); os.close(fd); open(path,"wb").write(raw)
        resp=_try([path]); return (resp.get("message") or {}).get("content","")
    finally:
        if path and os.path.exists(path):
            try: os.remove(path)
            except: pass

def _format_step_summary(cot: Dict[str, Any], n: int) -> str:
    """Return 'Step n: <text>  |  Region: [x,y,w,h]/None'"""
    if not cot or "steps" not in cot or n < 1 or n > len(cot["steps"]):
        return f"Step {n} not available."
    st = cot["steps"][n-1]
    txt = (st.get("text") or "").strip()
    reg = st.get("region")
    return f"Step {n}: {txt}\nRegion: {reg if reg else 'None'}"


# ======================= CoT runner ========================
def call_llm_visual_cot(question: str, img: Image.Image, proposals: List[List[int]],
                        backend="ollama", ollama_model="llava:latest", min_non_null_regions=2) -> CoTResult:
    if backend=="demo":
        W,H=img.size; props=proposals[:3] if proposals else [[int(0.1*W),int(0.1*H),int(0.3*W),int(0.3*H)]]
        steps=[{"idx":1,"text":"Scan the scene for salient objects.","region":props[0]},
               {"idx":2,"text":"Focus on the most relevant region.","region":props[0]},
               {"idx":3,"text":"Use context to confirm the answer.","region":None}]
        return CoTResult(steps=[CoTStep(**s) for s in steps], final_answer="Demo answer.")



    def _run(sys_prompt: str, props: List[List[int]]):
        user_prompt = f"{sys_prompt}\nQUESTION:{question}\nPROPOSALS:{json.dumps(props)}"
        data = ollama_chat_visual_json(user_prompt, img, model=ollama_model)
        if not data:
            return None, None
        steps_raw = data.get("steps", []) or []
        ans = data.get("final_answer", "")
        return steps_raw, ans

    
    
    try:
        W,H=img.size; props=proposals[:] if proposals else []
        steps_raw, ans=_run(VISUAL_COT_SYS_PROMPT, props)
        if not props:
            props=fallback_grid_proposals(img,3,3); need_retry=True
        else:
            need_retry=False
        def _sanitize(sr_list):
            out=[]
            for i,s in enumerate(sr_list or []):
                idx=int(float(s.get("idx", i+1))) if isinstance(s.get("idx",1),(int,float,str)) else (i+1)
                text=str(s.get("text","")).strip()
                r=_coerce_region_any(s.get("region",None), W,H, proposals=props)
                out.append({"idx":idx,"text":text,"region":r})
            return out
        steps_s=_sanitize(steps_raw)
        nn=sum(1 for s in steps_s if s.get("region") is not None)
        if need_retry or nn<min_non_null_regions:
            steps_raw, ans=_run(VISUAL_COT_SYS_PROMPT_STRICT, props); steps_s=_sanitize(steps_raw)
        if not steps_s:
            steps_raw, ans=_run(VISUAL_COT_SYS_PROMPT_STRICT, props); steps_s=_sanitize(steps_raw)
        steps=[CoTStep(**s) for s in (steps_s or [])] or [CoTStep(idx=1,text="No steps returned by model.",region=None)]
        return CoTResult(steps=steps, final_answer=ans or "")
    except Exception as e:
        return CoTResult(steps=[CoTStep(idx=1,text=f"Ollama backend error: {e}",region=None)], final_answer="")

# ======================= Phrase grounding via YOLO ========
_ALIAS = {
    "boy":["person","child","man","woman","people","human"],
    "girl":["person","child","woman","people","human"],
    "kid":["person","child","people"],
    "man":["person"], "woman":["person"], "person":["person"], "people":["person"],
    "dog":["dog"], "puppy":["dog"], "cat":["cat"], "kitten":["cat"],
    "car":["car","truck","bus","train"], "vehicle":["car","truck","bus","train"],
    "bike":["bicycle","motorcycle"], "bicycle":["bicycle"], "motorbike":["motorcycle"],
    "bus":["bus"], "truck":["truck"], "motorcycle":["motorcycle"],
    "chair":["chair"], "bench":["bench"], "bottle":["bottle"], "cup":["cup"],
    "bird":["bird"], "cow":["cow"], "horse":["horse"], "sheep":["sheep"],
}

def _norm_phrase(s: str) -> str:
    s=s.lower().strip()
    s=re.sub(r"^(the|a|an)\s+","",s)
    s=re.sub(r"[^\w\s]","",s)
    return s

def find_boxes_for_phrase(img: Image.Image, phrase: str, topk:int=5) -> List[List[int]]:
    """Return candidate boxes for a phrase using YOLO + aliases."""
    dets = yolo_detect(img, conf=0.2, max_det=50)
    if not dets: return []
    p=_norm_phrase(phrase)
    if p in ("objects","everything","all"):
        dets=sorted(dets, key=lambda d: d["conf"], reverse=True)[:topk]
        return [d["xywh"] for d in dets]
    # candidate class names
    cands=set(_ALIAS.get(p, [])) | {p}
    # filter by name containment either side
    scored=[]
    for d in dets:
        name=d["name"].lower()
        if any(c in name or name in c for c in cands):
            scored.append((d["conf"], d["xywh"]))
    if scored:
        scored.sort(reverse=True)
        return [xywh for _,xywh in scored[:topk]]
    # fallback: no direct class match → try CoT step text mentions
    return []

# ======================= CoT helpers ======================
def get_step_region(cot: Dict[str,Any], n:int) -> Optional[List[int]]:
    try:
        if not cot or "steps" not in cot: return None
        return cot["steps"][n-1].get("region")
    except Exception:
        return None

# ======================= Image Pipeline ===================
def pipeline_image(image: Image.Image, question: str, backend: str, ollama_model: str,
                   sam_ckpt: str, sam_model_type: str, state: Dict[str,Any]):
    if image is None or not question:
        return None, "Please provide an image and a question.", None, None, state
    image=image.convert("RGB")
    proposals = proposals_yolo(image, max_props=8) or fallback_grid_proposals(image,3,3)
    cot = call_llm_visual_cot(question, image, proposals, backend=backend, ollama_model=ollama_model)
    img_all=image.copy()
    for s in cot.steps:
        if getattr(s,"region",None): img_all=draw_box(img_all, s.region, str(s.idx))
    board=make_storyboard(image, cot, 4)
    cot_dict = cot.model_dump() if hasattr(cot,"model_dump") else (cot.dict() if hasattr(cot,"dict") else {
        "steps":[{"idx":s.idx,"text":s.text,"region":getattr(s,"region",None)} for s in cot.steps],
        "final_answer":cot.final_answer
    })
    cot_json=json.dumps(cot_dict, indent=2, ensure_ascii=False)
    state=dict(state or {})
    state.update({
        "base_image": image, "current_image": img_all.copy(),
        "history_imgs":[img_all.copy()], "cot": cot_dict, "proposals": proposals,
        "backend": backend, "ollama_model": ollama_model,
        "sam_ckpt": sam_ckpt.strip(), "sam_model_type": sam_model_type
    })
    return img_all, cot_json, board, cot.final_answer, state

# ======================= Video Pipeline ===================
current_video_question = "Analyze the video and explain what the events are."

import math

def _safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        try:
            v = float(str(x).strip())
        except Exception:
            return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v

def _safe_int(x, default: int) -> int:
    v = _safe_float(x)
    return int(round(v)) if v is not None else default


def sample_video_frames(path: str, fps:float=0.5, max_frames:int=12, resize_to:int=720) -> List[Image.Image]:
    cap=cv2.VideoCapture(path); 
    if not cap.isOpened(): return []
    video_fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    step=max(int(round(video_fps/max(fps,0.1))),1)
    frames=[]; idx=0; taken=0
    while True:
        ret=cap.grab(); 
        if not ret: break
        if idx%step==0:
            ok,fr=cap.retrieve(); 
            if not ok: break
            fr=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            img=Image.fromarray(fr); img.thumbnail((resize_to,resize_to), Image.Resampling.LANCZOS)
            frames.append(img); taken+=1
            if taken>=max_frames: break
        idx+=1
    cap.release(); return frames

def write_video(frames: List[Image.Image], out: str, fps:float=2.0):
    if not frames: return None
    h,w=np.array(frames[0]).shape[:2]; vw=cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    for im in frames: vw.write(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
    vw.release(); return out

def _cot_one_frame(i_img, backend, ollama_model):
    i,img=i_img; props=proposals_yolo(img,8) or fallback_grid_proposals(img,3,3)
    cot=call_llm_visual_cot(current_video_question, img, props, backend, ollama_model)
    img_all=img.copy()
    for s in cot.steps:
        if getattr(s,"region",None): img_all=draw_box(img_all, s.region, str(s.idx))
    board=make_storyboard(img, cot, 4)
    jd = cot.model_dump() if hasattr(cot,"model_dump") else {"steps":[{"idx":s.idx,"text":s.text,"region":s.region} for s in cot.steps],"final_answer":cot.final_answer}
    return (i, img_all, jd, board, cot.final_answer)

import json, re, tempfile, os
from typing import Any, Optional, Dict, List
from PIL import Image

def try_parse_json(text: str) -> Optional[dict]:
    """Best-effort JSON parser for model output."""
    if not text:
        return None
    # direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # extract first {...} block
    s = text.find("{"); e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        block = text[s:e+1]
        try:
            return json.loads(block)
        except Exception:
            pass
    return None

def ollama_chat_visual_json(prompt: str, img: Image.Image, model: str) -> Optional[dict]:
    """
    Ask Ollama for JSON directly. Falls back to non-JSON mode and repairs.
    Returns dict or None.
    """
    import ollama, base64
    from io import BytesIO

    # prepare image payloads once
    buf = BytesIO(); img.save(buf, format="PNG"); raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64}"

    def _ask(images_payload, force_json: bool):
        kwargs = dict(model=model,
                      messages=[{"role": "user", "content": prompt, "images": images_payload}],
                      options={"temperature": 0.2})
        if force_json:
            kwargs["format"] = "json"  # <- structured output
        return ollama.chat(**kwargs)

    # 1) try JSON mode (data URI, then raw b64)
    for payload in ([data_uri], [b64]):
        try:
            resp = _ask(payload, force_json=True)
            txt = (resp.get("message") or {}).get("content", "")
            data = try_parse_json(txt)
            if data is not None:
                return data
        except Exception:
            pass

    # 2) fallback: non-JSON mode + repair
    for payload in ([data_uri], [b64]):
        try:
            resp = _ask(payload, force_json=False)
            txt = (resp.get("message") or {}).get("content", "")
            data = try_parse_json(txt)
            if data is not None:
                return data
        except Exception:
            pass

    # 3) last resort: temp file path
    path = None
    try:
        fd, path = tempfile.mkstemp(suffix=".png"); os.close(fd)
        with open(path, "wb") as f: f.write(raw)
        try:
            resp = _ask([path], force_json=True)
            txt = (resp.get("message") or {}).get("content", "")
            data = try_parse_json(txt)
            if data is not None:
                return data
        except Exception:
            pass
        resp = _ask([path], force_json=False)
        txt = (resp.get("message") or {}).get("content", "")
        return try_parse_json(txt)
    finally:
        if path and os.path.exists(path):
            try: os.remove(path)
            except Exception: pass



from concurrent.futures import ThreadPoolExecutor, as_completed
def video_visual_cot(path:str, backend:str, ollama_model:str, fps:float=0.5, max_frames:int=12, workers:int=2):
    frames=sample_video_frames(path,fps,max_frames)
    res=[None]*len(frames)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs=[ex.submit(_cot_one_frame,(i,im),backend,ollama_model) for i,im in enumerate(frames)]
        for f in as_completed(futs): res[f.result()[0]]=f.result()
    over, cjson, boards, ans=[],[],[],[]
    for _,a,j,b,s in res: over.append(a); cjson.append(j); boards.append(b); ans.append(s)
    return over, cjson, boards, ans

def pipeline_video(video_file, question, backend, model, fps, maxf, workers, make_clip):
    if not video_file: return None, "Please upload a video.", None, None, None
    global current_video_question; current_video_question=question or current_video_question
    over,cjson,boards,answers=video_visual_cot(video_file,backend,model,fps,maxf,workers)
    first_img=over[0] if over else None; first_board=boards[0] if boards else None
    out_json=json.dumps({"frames":cjson}, indent=2, ensure_ascii=False)
    summary=" | ".join([a for a in answers if a])
    clip=None
    if make_clip:
        path=os.path.join(tempfile.gettempdir(),"visual_cot_summary.mp4"); write_video(over,path,2.0); clip=path
    return first_img, out_json, first_board, summary, clip

# ======================= Chat (commands + NL) =============
BOX_RE = re.compile(r"\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?")

def parse_box(s:str)->Optional[List[int]]:
    m=BOX_RE.search(s); 
    return [int(m.group(1)),int(m.group(2)),int(m.group(3)),int(m.group(4))] if m else None

def chat_handle(user_msg: str,
                chat_hist: List[Dict[str, str]],
                edited_img: Optional[Image.Image],
                state: Dict[str, Any]):
    """
    Unified chat handler:
      - Direct CoT access:  'step1', 'step 2', 'show step 3', 'steps', 'final'
      - CoT-linked ops:     'box/crop/segment/explain step N'
      - Phrase grounding:   'segment/box/crop/explain <object or phrase>'
      - Explicit coords:    'box/crop [x,y,w,h]'
      - Edits:              grayscale, blur <r>[/region], annotate "t" at x,y, rotate d, brightness f, contrast f, undo
      - Generic VQA fallback using the vision model
    Returns: (chat history, edited image, state)
    """
    # ----- state & context -----
    state = dict(state or {})
    img = state.get("current_image")
    cot = state.get("cot")
    backend = state.get("backend", "ollama")
    ollama_model = state.get("ollama_model", "llava:latest")
    sam_ckpt = state.get("sam_ckpt", "")
    sam_model_type = state.get("sam_model_type", "vit_h")

    history = list(chat_hist or [])
    history.append({"role": "user", "content": user_msg})

    if img is None:
        history.append({"role": "assistant",
                        "content": "Please run Visual CoT on an image first (Image tab)."})
        return history, edited_img, state

    def push_snapshot():
        h = state.get("history_imgs", [])
        h.append(img.copy())
        state["history_imgs"] = h

    def step_summary(cot_dict: Dict[str, Any], n: int) -> str:
        if not cot_dict or "steps" not in cot_dict or n < 1 or n > len(cot_dict["steps"]):
            return f"Step {n} not available."
        st = cot_dict["steps"][n - 1]
        txt = (st.get("text") or "").strip()
        reg = st.get("region")
        return f"Step {n}: {txt}\nRegion: {reg if reg else 'None'}"

    um = user_msg.strip()
    uml = um.lower()
    reply = ""

    # =========================
    # 1) Direct CoT navigation
    # =========================
    m_direct = (re.fullmatch(r"(?:show\s+)?step\s*([0-9]+)", uml)
                or re.fullmatch(r"step([0-9]+)", uml))
    if m_direct and not any(uml.startswith(p) for p in ("box ", "crop ", "segment ", "explain ")):
        if not cot or not cot.get("steps"):
            reply = "No CoT steps yet. Run Visual CoT first."
        else:
            n = int(m_direct.group(1))
            if n < 1 or n > len(cot["steps"]):
                reply = f"Step {n} is out of range (1..{len(cot['steps'])})."
            else:
                st = cot["steps"][n - 1]
                r = st.get("region")
                # draw the step box for visual orientation
                if r:
                    push_snapshot()
                    img = draw_box(img, r, f"step {n}")
                    state["current_image"] = img
                # short explanation from region (if possible)
                explain = ""
                if r and backend == "ollama":
                    x, y, w, h = r
                    sub = img.crop((x, y, x + w, y + h))
                    try:
                        ans = ollama_chat_visual(
                            f"You are given a region for step {n} with text: \"{st.get('text','')}\".\n"
                            "Explain this region in 1–2 sentences, grounded in visible evidence.",
                            sub, model=ollama_model
                        )
                        explain = (ans or "").strip()
                    except Exception as e:
                        explain = f"(Ollama error: {e})"
                reply = step_summary(cot, n) + (f"\n\nAnswer: {explain}" if explain else "")
        history.append({"role": "assistant", "content": reply})
        return history, img, state

    if re.fullmatch(r"(list\s+steps|steps)", uml):
        if not cot or not cot.get("steps"):
            reply = "No CoT steps yet. Run Visual CoT first."
        else:
            lines = [f"{s['idx']}. {s.get('text','').strip()}" for s in cot["steps"]]
            reply = "Steps:\n" + "\n".join(lines)
        history.append({"role": "assistant", "content": reply})
        return history, img, state

    if re.fullmatch(r"(final|final answer|answer)", uml):
        reply = (cot or {}).get("final_answer") or "No final answer yet."
        history.append({"role": "assistant", "content": reply})
        return history, img, state

    # ======================================
    # 2) CoT-linked ops: "... step N"
    # ======================================
    m = re.search(r"^(box|crop|segment|explain)\s+step\s+(\d+)\b", uml)
    if m:
        cmd = m.group(1)
        n = int(m.group(2))
        r = get_step_region(cot, n) if cot else None
        if not r:
            reply = f"No region in step {n}."
        else:
            if cmd == "box":
                push_snapshot()
                img = draw_box(img, r, f"step {n}")
                reply = f"Drew box for step {n}."
            elif cmd == "crop":
                push_snapshot()
                x, y, w, h = r
                img = img.crop((x, y, x + w, y + h))
                reply = f"Cropped to step {n}."
            elif cmd == "segment":
                if not sam_ckpt:
                    reply = "Set SAM checkpoint on the Image tab first."
                else:
                    push_snapshot()
                    mask = sam_segment_box(img, r, ckpt=sam_ckpt, model_type=sam_model_type)
                    if mask is None:
                        reply = "SAM not available or checkpoint invalid."
                    else:
                        img = overlay_masks(img, [mask])
                        reply = f"Segmented step {n}."
            elif cmd == "explain":
                if backend != "ollama":
                    reply = "Q&A requires a vision model (set Backend to 'ollama')."
                else:
                    x, y, w, h = r
                    sub = img.crop((x, y, x + w, y + h))
                    ans = ollama_chat_visual("Explain this region in 2–4 sentences.", sub, model=ollama_model)
                    reply = (ans or "").strip() or "No answer."
        state["current_image"] = img
        history.append({"role": "assistant", "content": reply})
        return history, img, state

    # ======================================================
    # 3) Explicit coords: 'box [x,y,w,h]' / 'crop [x,y,w,h]'
    # ======================================================
    if uml.startswith("box ") or uml == "box":
        bx = parse_box(um)
        if not bx:
            reply = "Usage: box [x,y,w,h]  or  box step N  or  box <object>"
        else:
            push_snapshot()
            img = draw_box(img, bx, "box")
            reply = f"Drew box {bx}."
        state["current_image"] = img
        history.append({"role": "assistant", "content": reply})
        return history, img, state

    if uml.startswith("crop ") or uml == "crop":
        bx = parse_box(um)
        if not bx:
            reply = "Usage: crop [x,y,w,h]  or  crop step N  or  crop <object>"
        else:
            push_snapshot()
            x, y, w, h = bx
            img = img.crop((x, y, x + w, y + h))
            reply = f"Cropped to {bx}."
        state["current_image"] = img
        history.append({"role": "assistant", "content": reply})
        return history, img, state

    # ==========================================================
    # 4) Phrase grounding: 'segment/box/crop/explain <phrase>'
    # ==========================================================
    pm = re.match(r"^(segment|box|crop|explain)\s+(.+)$", um, flags=re.IGNORECASE)
    if pm and "step" not in uml and not parse_box(um):
        action = pm.group(1).lower()
        phrase = pm.group(2).strip()
        boxes = find_boxes_for_phrase(img, phrase, topk=5)

        # fallback: scan CoT step texts for the phrase
        if not boxes and cot and cot.get("steps"):
            for s in cot["steps"]:
                if phrase.lower() in s.get("text", "").lower() and s.get("region"):
                    boxes = [s["region"]]
                    break

        if not boxes:
            reply = (f"Couldn't find '{phrase}'. Try a COCO class like 'person', 'dog', 'car', "
                     f"or use '... step N'.")
        else:
            if action == "box":
                push_snapshot()
                for i, bx in enumerate(boxes[:3]):
                    img = draw_box(img, bx, f"{phrase if i == 0 else phrase + ' ' + str(i+1)}")
                reply = f"Drew box for '{phrase}' ({len(boxes)} candidate(s))."
            elif action == "crop":
                push_snapshot()
                x, y, w, h = boxes[0]
                img = img.crop((x, y, x + w, y + h))
                reply = f"Cropped to '{phrase}'."
            elif action == "segment":
                if not sam_ckpt:
                    reply = "Set SAM checkpoint on the Image tab first."
                else:
                    push_snapshot()
                    masks = []
                    for bx in boxes[:3]:  # limit for speed
                        m = sam_segment_box(img, bx, ckpt=sam_ckpt, model_type=sam_model_type)
                        if m is not None:
                            masks.append(m)
                    if not masks:
                        reply = f"Could not segment '{phrase}'."
                    else:
                        img = overlay_masks(img, masks)
                        reply = f"Segmented '{phrase}' ({len(masks)} region(s))."
            elif action == "explain":
                if backend != "ollama":
                    reply = "Q&A requires a vision model (set Backend to 'ollama')."
                else:
                    x, y, w, h = boxes[0]
                    sub = img.crop((x, y, x + w, y + h))
                    ans = ollama_chat_visual(f"Explain this '{phrase}' region in 2–4 sentences.",
                                             sub, model=ollama_model)
                    reply = (ans or "").strip() or "No answer."
        state["current_image"] = img
        history.append({"role": "assistant", "content": reply})
        return history, img, state

    # =========================
    # 5) Simple edit commands
    # =========================
    if uml.startswith("grayscale"):
        push_snapshot()
        img = ImageOps.grayscale(img).convert("RGB")
        reply = "Applied grayscale."
    elif uml.startswith("blur"):
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", um)
        radius = float(nums[-1]) if nums else 5.0
        bx = parse_box(um)
        push_snapshot()
        if bx:
            x, y, w, h = bx
            crop = img.crop((x, y, x + w, y + h)).filter(ImageFilter.GaussianBlur(radius=radius))
            img.paste(crop, (x, y))
            reply = f"Blurred region {bx} (r={radius})."
        else:
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
            reply = f"Blurred (r={radius})."
    elif uml.startswith("annotate"):
        tm = re.search(r'annotate\s+[\"\'](.+?)[\"\']', um, re.IGNORECASE)
        xm = re.search(r"at\s+(\d+)\s*,\s*(\d+)", um, re.IGNORECASE)
        if tm and xm:
            push_snapshot()
            text = tm.group(1)
            x, y = int(xm.group(1)), int(xm.group(2))
            img = annotate_text(img, text, (x, y))
            reply = "Annotated."
        else:
            reply = 'Usage: annotate "text" at x,y'
    elif uml.startswith("rotate"):
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", um)
        deg = float(nums[0]) if nums else 0.0
        push_snapshot()
        img = img.rotate(-deg, expand=True)
        reply = f"Rotated {deg}°."
    elif uml.startswith("brightness"):
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", um)
        f = float(nums[0]) if nums else 1.0
        push_snapshot()
        img = ImageEnhance.Brightness(img).enhance(f)
        reply = f"Brightness x{f}."
    elif uml.startswith("contrast"):
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", um)
        f = float(nums[0]) if nums else 1.0
        push_snapshot()
        img = ImageEnhance.Contrast(img).enhance(f)
        reply = f"Contrast x{f}."
    elif uml.startswith("undo"):
        hist_imgs = state.get("history_imgs", [])
        if hist_imgs:
            img = hist_imgs.pop()
            state["history_imgs"] = hist_imgs
            reply = "Undid last operation."
        else:
            reply = "Nothing to undo."
    elif uml.startswith("redraw steps"):
        if not cot or not cot.get("steps"):
            reply = "No CoT steps available."
        else:
            push_snapshot()
            img = img.copy()
            for s in cot["steps"]:
                if s.get("region"):
                    img = draw_box(img, s["region"], str(s.get("idx", "?")))
            reply = "Redrew CoT step boxes."
    else:
        # =========================
        # 6) Generic VQA fallback
        # =========================
        if backend != "ollama":
            reply = "Q&A requires a vision model (set Backend to 'ollama')."
        else:
            try:
                ans = ollama_chat_visual("Answer in 2–4 sentences:\nQUESTION: " + um,
                                         img, model=ollama_model)
                reply = (ans or "").strip() or "No answer."
            except Exception as e:
                reply = f"Ollama error: {e}"

    state["current_image"] = img
    history.append({"role": "assistant", "content": reply})
    return history, img, state


# ======================= UI ===============================
def build_ui():
    with gr.Blocks(title="Visual CoT — Image & Video", css=".small{font-size:12px}") as demo:
        gr.Markdown("# Visual Chain-of-Thought — Image & Video")

        css="""
            .small { font-size:12px }
            #final-answer-img textarea, #final-answer-vid textarea {
            width: 100% !important;
            min-height: 140px;
            font-size: 16px;
            line-height: 1.35;
            }
            """

        st = gr.State({})

        with gr.Tabs():
            # ---- Image tab ----
            with gr.Tab("Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Image Inputs")
                        img_in = gr.Image(type="pil", label="Image")
                        backend_i = gr.Radio(["ollama","demo"], value="ollama", label="Backend")
                        model_i = gr.Dropdown(choices=["llava:latest","llava:13b","llama3.2-vision:latest"], value="llava:latest", label="Ollama vision model")
                        img_q = gr.Textbox(label="Image question", value="What is happening in this image? Explain in detail.")
                        gr.Markdown("#### SAM (used only when you ask in chat)")
                        sam_ckpt = gr.Textbox(label="Path to SAM .pth (sam_vit_h/l/b.pth)", placeholder="/path/to/sam_vit_h_4b8939.pth")
                        sam_model_type = gr.Dropdown(choices=["vit_h","vit_l","vit_b"], value="vit_h", label="SAM model type")
                        btn_img = gr.Button("Run Visual CoT (Image)", variant="primary")
                    with gr.Column(scale=1):
                        gr.Markdown("### Visual CoT Outputs")
                        out_img = gr.Image(type="pil", label="Overlaid steps (all boxes)")
                        out_board = gr.Image(type="pil", label="Storyboard (per step)")
                        out_answer = gr.Textbox(label="Final Answer")
                        out_json = gr.Code(language="json", label="CoT JSON")
                    with gr.Column(scale=1):
                        gr.Markdown("### Image Chat & Editor")
                        chat = gr.Chatbot(label="Ask or Command", height=320, type="messages")
                        chat_in = gr.Textbox(placeholder="Try: segment boy | segment objects | box step 2 | explain car | crop dog", label="Your message")
                        send = gr.Button("Send")
                        edited_img = gr.Image(type="pil", label="Edited image")
                        gr.Markdown("""
                                **Natural language commands**
                                - `step1` / `step 2` / `show step 3`  ➜ show that step, draw its box, and explain the region
                                - `steps`  ➜ list all steps
                                - `final`  ➜ show the final answer
                                - `segment boy` / `segment dog` / `segment objects`
                                - `box <object>` / `crop <object>` / `explain <object>`
                                - CoT-linked: `box step N`, `crop step N`, `segment step N`, `explain step N`
                                - Other: `grayscale`, `blur 5`, `annotate "hi" at 40,60`, `undo`
                                """, elem_classes=["small"])

                btn_img.click(
                    pipeline_image,
                    inputs=[img_in, img_q, backend_i, model_i, sam_ckpt, sam_model_type, st],
                    outputs=[out_img, out_json, out_board, out_answer, st],
                )
                send.click(chat_handle, inputs=[chat_in, chat, edited_img, st], outputs=[chat, edited_img, st])
                chat_in.submit(chat_handle, inputs=[chat_in, chat, edited_img, st], outputs=[chat, edited_img, st])

            # ---- Video tab ----
            with gr.Tab("Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Video Inputs")
                        vid_in = gr.Video(label="Video", interactive=True)
                        backend_v = gr.Radio(["ollama","demo"], value="ollama", label="Backend")
                        model_v = gr.Dropdown(choices=["llava:latest","llava:13b","llama3.2-vision:latest"], value="llava:latest", label="Ollama vision model")
                        vid_q = gr.Textbox(label="Video question", value="Analyze the video and explain what the events are.")
                        fps_box = gr.Slider(0.2, 2.0, value=0.5, step=0.1, label="Analysis FPS")
                        maxf_box = gr.Slider(4, 32, value=12, step=1, label="Max frames")
                        workers_box = gr.Slider(1, 4, value=2, step=1, label="Parallel workers")
                        make_clip_box = gr.Checkbox(value=False, label="Generate highlight reel (MP4)")
                        btn_vid = gr.Button("Run Visual CoT (Video)")
                    with gr.Column(scale=1):
                        gr.Markdown("### Video Outputs")
                        out_img_v = gr.Image(type="pil", label="First analyzed frame (overlaid)")
                        out_board_v = gr.Image(type="pil", label="Storyboard (first frame)")
                        out_json_v = gr.Code(language="json", label="Per-frame CoT JSON")
                        out_answer_v = gr.Textbox(label="Summary of answers")
                        out_clip_v = gr.Video(label="Highlight reel (if generated)")
                def _pipe(video_file, question, backend, model, fps, maxf, workers, make_clip):
                    return pipeline_video(video_file, question, backend, model, fps, maxf, workers, make_clip)
                btn_vid.click(_pipe, inputs=[vid_in, vid_q, backend_v, model_v, fps_box, maxf_box, workers_box, make_clip_box],
                              outputs=[out_img_v, out_json_v, out_board_v, out_answer_v, out_clip_v])

        gr.Markdown("**Tip:** If a phrase isn't found, try a COCO class name (e.g., `person`, `dog`, `car`) or use `segment step N`.")

    return demo

# ======================= Main =============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=args.port)
