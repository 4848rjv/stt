# index.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional, List
from io import BytesIO
import os
import re
import hashlib
import threading
import numpy as np
import torch
import soundfile as sf

APP_TITLE = "Silero TTS for FreeSWITCH"
CACHE_DIR = os.environ.get("TTS_CACHE_DIR", "/tmp/tts_cache")
AUTH_USER = os.environ.get("TTS_USER", "")
AUTH_PASS = os.environ.get("TTS_PASS", "")

os.makedirs(CACHE_DIR, exist_ok=True)

app = FastAPI(title=APP_TITLE)

model, example_text = torch.hub.load(
    repo_or_dir="snakers4/silero-models",
    model="silero_tts",
    language="ru",
    speaker="v4_ru",
    trust_repo=True
)

AVAILABLE_SPEAKERS = set(getattr(model, "speakers",
    ["baya_v2","irina_v2","kseniya_v2","natasha_v2","ruslan_v2","aidar_v2"]))
MODEL_RATES = sorted(getattr(model, "sample_rates", [8000, 24000, 48000]))
ACCEPT_RATES = [8000, 16000, 24000, 48000]

_tts_lock = threading.Lock()

def normalize_ws(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.split("\n")]
    out: List[str] = []
    skip_blank = False
    for ln in lines:
        if ln:
            out.append(ln)
            skip_blank = False
        else:
            if not skip_blank:
                out.append("")
            skip_blank = True
    return "\n".join(out).strip()

def text_to_ssml(text: str) -> str:
    text = normalize_ws(text)
    if not text:
        raise ValueError("empty text")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    p_blocks = []
    for p in paragraphs:
        parts = re.findall(r"([^.!?]+)([.!?]+|\Z)", p)
        s_blocks = []
        for body, end in parts:
            sent = body.strip()
            if not sent:
                continue
            sent = re.sub(r",\s*", r', <break time="250ms"/> ', sent)
            sent = re.sub(r";\s*", r'; <break time="250ms"/> ', sent)
            sent = re.sub(r":\s*", r': <break time="250ms"/> ', sent)
            sent = re.sub(r"\s?—\s?", r' <break time="200ms"/> ', sent)
            sent = re.sub(r"\(", r'<break time="150ms"/>(', sent)
            sent = re.sub(r"\)", r')<break time="150ms"/>', sent)

            end_break = ""
            if "!" in end or "?" in end:
                end_break = '<break time="700ms"/>'
            elif "." in end:
                end_break = '<break time="500ms"/>'

            s_blocks.append(f"<s>{sent} {end_break}</s>")
        p_blocks.append(f"<p>{' '.join(s_blocks)}</p>")
    return f"<speak>{' '.join(p_blocks)}</speak>"

def _cache_key(*parts: str) -> str:
    h = hashlib.md5()
    for p in parts:
        h.update((p if p is not None else "").encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()

def _nearest_model_rate(target_sr: int) -> int:
    return min(MODEL_RATES, key=lambda r: abs(r - target_sr))

def _resample_np(wav: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return wav
    if wav.ndim != 1:
        wav = wav.reshape(-1)
    n_src = wav.shape[0]
    n_dst = int(round(n_src * (sr_to / sr_from)))
    if n_dst <= 1:
        return wav[:1]
    x_src = np.linspace(0.0, 1.0, num=n_src, endpoint=True)
    x_dst = np.linspace(0.0, 1.0, num=n_dst, endpoint=True)
    out = np.interp(x_dst, x_src, wav).astype(np.float32)
    out = np.clip(out, -1.0, 1.0)
    return out

def _synthesize_to_file(*, ssml_text: Optional[str], plain_text: Optional[str],
                        speaker: str, sample_rate: int, outfile: str) -> None:
    if speaker not in AVAILABLE_SPEAKERS:
        raise ValueError(f"voice '{speaker}' not in {sorted(AVAILABLE_SPEAKERS)}")
    if sample_rate not in ACCEPT_RATES:
        raise ValueError(f"sample_rate should be in {ACCEPT_RATES}, current value is {sample_rate}")

    if ssml_text is None:
        text = (plain_text or "").strip()
        if not text:
            raise ValueError("empty text")
        ssml_text = text_to_ssml(text)

    model_sr = sample_rate if sample_rate in MODEL_RATES else _nearest_model_rate(sample_rate)

    with _tts_lock:
        wav = model.apply_tts(
            ssml_text=ssml_text,
            speaker=speaker,
            sample_rate=model_sr
        )

    wav = np.asarray(wav, dtype=np.float32)
    if model_sr != sample_rate:
        wav = _resample_np(wav, model_sr, sample_rate)

    sf.write(outfile, wav, sample_rate, format="WAV", subtype="PCM_16")

def _maybe_auth(user: str, pwd: str) -> None:
    if not AUTH_USER and not AUTH_PASS:
        return
    if user != AUTH_USER or pwd != AUTH_PASS:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.get("/ping")
def ping():
    return JSONResponse({
        "status": "ok",
        "voices": sorted(AVAILABLE_SPEAKERS),
        "model_sample_rates": MODEL_RATES,
        "accept_sample_rates": ACCEPT_RATES
    })

@app.get("/api/tts/tts")
def api_tts(
    user: Optional[str] = Query(None),
    pass_: Optional[str] = Query(None, alias="pass"),
    amp_pass: Optional[str] = Query(None, alias="amp;pass"),
    text: Optional[str] = Query(None, description="Plain text"),
    amp_text: Optional[str] = Query(None, alias="amp;text"),
    voice: str = Query("baya_v2"),
    amp_voice: Optional[str] = Query(None, alias="amp;voice"),
    sr: Optional[int] = Query(16000, description="Sample rate"),
    amp_sr: Optional[int] = Query(None, alias="amp;sr"),
    ssml: Optional[str] = Query(None, description="Full SSML (<speak>...</speak>)"),
    amp_ssml: Optional[str] = Query(None, alias="amp;ssml"),
):

    pass_ = pass_ or amp_pass
    text  = text or amp_text
    voice = voice or amp_voice
    ssml  = ssml or amp_ssml
    sr    = (sr or amp_sr or 16000)

    try:
        _maybe_auth(user or "", pass_ or "")
        if not (text or ssml):
            raise HTTPException(status_code=400, detail="either 'text' or 'ssml' must be provided")

        key = _cache_key(voice, str(sr), ssml or "", text or "")
        out_path = os.path.join(CACHE_DIR, f"{key}.wav")

        if not os.path.exists(out_path):
            _synthesize_to_file(
                ssml_text=ssml, plain_text=text,
                speaker=voice, sample_rate=int(sr),
                outfile=out_path
            )

        return FileResponse(out_path, media_type="audio/wav", filename="tts.wav")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/tts")
def tts(
    user: Optional[str] = Query(None),
    password: Optional[str] = Query(None),
    amp_pass: Optional[str] = Query(None, alias="amp;pass"),
    text: Optional[str] = Query(None, description="Plain text"),
    amp_text: Optional[str] = Query(None, alias="amp;text"),
    voice: str = Query("baya_v2"),
    amp_voice: Optional[str] = Query(None, alias="amp;voice"),
    sr: Optional[int] = Query(16000, description="Sample rate"),
    amp_sr: Optional[int] = Query(None, alias="amp;sr"),
    ssml: Optional[str] = Query(None, description="Full SSML"),
    amp_ssml: Optional[str] = Query(None, alias="amp;ssml"),
):
    password = password or amp_pass
    text  = text or amp_text
    voice = voice or amp_voice
    ssml  = ssml or amp_ssml
    sr    = (sr or amp_sr or 16000)

    try:
        _maybe_auth(user or "", password or "")
        if not (text or ssml):
            raise HTTPException(status_code=400, detail="either 'text' or 'ssml' must be provided")

        key = _cache_key(voice, str(sr), ssml or "", text or "")
        out_path = os.path.join(CACHE_DIR, f"{key}.wav")

        if not os.path.exists(out_path):
            _synthesize_to_file(
                ssml_text=ssml, plain_text=text,
                speaker=voice, sample_rate=int(sr),
                outfile=out_path
            )
        return FileResponse(out_path, media_type="audio/wav", filename="tts.wav")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/say")
def say(
    text: str = Query(..., description="Plain text (will be wrapped to SSML with pauses)"),
    voice: str = Query("baya_v2"),
    sr: int = Query(16000, description="Sample rate"),
    user: Optional[str] = Query(None),
    password: Optional[str] = Query(None)
):
    try:
        _maybe_auth(user or "", password or "")
        key = _cache_key(voice, str(sr), "", text)
        out_path = os.path.join(CACHE_DIR, f"{key}.wav")

        if not os.path.exists(out_path):
            _synthesize_to_file(
                ssml_text=None, plain_text=text,
                speaker=voice, sample_rate=int(sr),
                outfile=out_path
            )
        return FileResponse(out_path, media_type="audio/wav", filename="say.wav")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("index:app", host="0.0.0.0", port=5000, reload=False)
