import streamlit as st
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import tempfile
import asyncio
import base64
import socket
import shutil
import requests
import time
import json
import math
import os
import io
import wave
import re
import datetime
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont, ImageOps
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE_TYPE

# 선택적 라이브러리 안전 로드 (PyMuPDF, edge-tts)
try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    try:
        import pymupdf as fitz
        PYMUPDF_AVAILABLE = True
    except ImportError:
        PYMUPDF_AVAILABLE = False

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

# ==========================================
# [1] UI 레이아웃 및 네트워크 방어 설정 (최상단 필수)
# ==========================================
st.set_page_config(page_title="연세유업 AI 위생교육 스튜디오", layout="wide")
socket.setdefaulttimeout(600)

# 고정 AI 모델 상수 (유료 Gemini Developer API 직결)
PLANNING_MODEL = "gemini-3.8-flash"
VISUAL_OMNI_MODEL = "gemini-3.1-flash-image"
MANDATORY_OPENING_LINE = "안녕하십니까, 식품안전팀 곽정혁입니다."

# ==========================================
# [2] 보안 접속 설정 (비밀번호: 2082)
# ==========================================
def check_password():
    def password_entered():
        if st.session_state["password"] == "2082":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    if "password_correct" not in st.session_state:
        st.text_input("시스템 접속 비밀번호 입력", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("비밀번호 오류. 다시 입력하세요.", type="password", on_change=password_entered, key="password")
        return False
    else:
        return True

# ==========================================
# [3] API 키 로드 (유료 Gemini Developer API 직결)
# ==========================================
GEMINI_API_KEY = (
    st.secrets.get("GEMINI_API_KEY")
    or st.secrets.get("GOOGLE_API_KEY")
    or os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GOOGLE_API_KEY")
    or ""
).strip().strip('"').strip("'")

DEFAULT_OPENAI_KEY = (
    st.secrets.get("OPENAI_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or ""
).strip().strip('"').strip("'")

def get_genai_client(api_key):
    if not api_key:
        return None
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1beta")
    )

def get_ffmpeg_binary():
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        return ffmpeg_bin
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError(f"ffmpeg 실행 파일을 찾을 수 없습니다: {e}")

# ==========================================
# [4] 커스텀 CSS
# ==========================================
custom_theme_css = """
<style>
@media screen {
    html, body, [class*="css"] {
        color: #1e293b !important;
        font-family: '연세본문체', 'YonseiBody', '맑은 고딕', 'Malgun Gothic', -apple-system, sans-serif !important;
    }
    h1, h2, h3, h4 {
        font-family: '연세제목체', 'YonseiTitle', '맑은 고딕', 'Malgun Gothic', sans-serif !important;
    }
    [data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 2px solid #e2e8f0 !important;
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border: none !important;
        border-bottom: 3px solid transparent !important;
        border-radius: 0 !important;
        padding: 12px 16px !important;
        color: #64748b !important;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #0f172a !important;
        color: #0f172a !important;
        background-color: transparent !important;
    }
    .stButton>button {
        background-color: #0f172a !important;
        color: white !important;
        font-size: 15px !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 12px 24px !important;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    .stButton>button:hover {
        background-color: #334155 !important;
        transform: translateY(-1px);
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stFileUploader {
        border: 2px solid #cbd5e1 !important;
        border-radius: 6px !important;
        background-color: #ffffff !important;
    }
    hr {
        border-top: 2px solid #cbd5e1 !important;
        margin: 1.5em 0 !important;
    }
}
</style>
"""

# ==========================================
# [5] 깃허브 font/ 폴더 자동 스캔, 특수기호 방어 및 Auto-Fit 텍스트 엔진
# ==========================================
FONT_SEARCH_DIRS = ["font", "fonts", "폰트", "."]

FONT_SEARCH_MAP = {
    "연세제목체": {
        "exact_names": [
            "연세제목체.ttf", "연세제목체B.ttf", "연세제목체OTF.otf", "연세제목체.otf",
            "YonseiTitle.ttf", "YonseiTitleB.ttf", "YonseiTitle.otf"
        ],
        "keywords": ["연세제목", "yonseititle", "연세_제목", "yonsei_title"]
    },
    "연세본문체": {
        "exact_names": [
            "연세본문체.ttf", "연세본문체B.ttf", "연세본문체OTF.otf", "연세본문체.otf",
            "YonseiBody.ttf", "YonseiBodyB.ttf", "YonseiBody.otf"
        ],
        "keywords": ["연세본문", "yonseibody", "연세_본문", "yonsei_body"]
    },
    "맑은고딕": {
        "exact_names": [
            "malgunbd.ttf", "malgun.ttf", "맑은고딕.ttf", "맑은 고딕.ttf", "MalgunGothic.ttf"
        ],
        "keywords": ["malgun", "맑은고딕", "맑은 고딕"]
    }
}

def find_font_file(font_choice, is_bold=False, custom_font_paths=None):
    if custom_font_paths and custom_font_paths.get(font_choice):
        cp = custom_font_paths[font_choice]
        if os.path.exists(cp):
            return cp

    info = FONT_SEARCH_MAP.get(font_choice, FONT_SEARCH_MAP["맑은고딕"])

    if font_choice == "맑은고딕":
        target_files = ["malgunbd.ttf", "MalgunGothicBold.ttf"] if is_bold else ["malgun.ttf", "맑은고딕.ttf", "MalgunGothic.ttf"]
        for d in FONT_SEARCH_DIRS:
            for tf in target_files:
                cand = os.path.join(d, tf) if d != "." else tf
                if os.path.exists(cand):
                    return cand

    for d in FONT_SEARCH_DIRS:
        for exact_name in info["exact_names"]:
            cand = os.path.join(d, exact_name) if d != "." else exact_name
            if os.path.exists(cand):
                return cand

    for d in FONT_SEARCH_DIRS:
        if os.path.exists(d) and os.path.isdir(d):
            try:
                files_in_dir = sorted(os.listdir(d))
            except Exception:
                continue
            matched_files = []
            for fname in files_in_dir:
                lower_f = fname.lower()
                if lower_f.endswith((".ttf", ".otf", ".ttc")):
                    if any(kw in lower_f for kw in info["keywords"]):
                        matched_files.append(os.path.join(d, fname) if d != "." else fname)
            if matched_files:
                if is_bold:
                    for mf in matched_files:
                        if any(b_kw in os.path.basename(mf).lower() for b_kw in ["bold", "b.", "_b", "굵"]):
                            return mf
                return matched_files[0]

    system_fallbacks = [
        "C:/Windows/Fonts/malgunbd.ttf" if is_bold else "C:/Windows/Fonts/malgun.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf" if is_bold else "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if is_bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc"
    ]
    for fb in system_fallbacks:
        if os.path.exists(fb):
            return fb
    return None

def load_pil_font(font_choice, size, is_bold=False, custom_font_paths=None):
    font_path = find_font_file(font_choice, is_bold=is_bold, custom_font_paths=custom_font_paths)
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    for emergency_fb in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "C:/Windows/Fonts/malgun.ttf"]:
        if os.path.exists(emergency_fb):
            try:
                return ImageFont.truetype(emergency_fb, size)
            except Exception:
                pass
    return ImageFont.load_default()

def wrap_text_by_pixel(text, font, max_width, draw):
    if not text:
        return []
    cleaned_text = (
        str(text)
        .replace("•", "-")
        .replace("▸", "-")
        .replace("▪", "-")
        .replace("☑", "[V]")
        .replace("□", "[ ]")
    )
    lines = []
    for paragraph in cleaned_text.split('\n'):
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
            continue
        words = paragraph.split(' ')
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip() if current_line else word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if (bbox[2] - bbox[0]) <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = ""
                for ch in word:
                    test_ch = current_line + ch
                    bbox_c = draw.textbbox((0, 0), test_ch, font=font)
                    if (bbox_c[2] - bbox_c[0]) <= max_width:
                        current_line = test_ch
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = ch
        if current_line:
            lines.append(current_line)
    return lines

def _measure_wrapped_block(lines, font, sz, line_spacing_ratio, draw):
    if not lines:
        return 0, 0, []
    step_gap = max(3, int(round(sz * line_spacing_ratio)))
    empty_h = max(6, int(round(sz * 0.45)))
    line_metrics = []
    total_h = 0
    max_w = 0
    for idx, ln in enumerate(lines):
        if not ln:
            lh = empty_h
            lw = 0
        else:
            bb = draw.textbbox((0, 0), ln, font=font)
            lw = bb[2] - bb[0]
            raw_h = bb[3] - bb[1]
            lh = max(sz + 2, raw_h + 2)
        max_w = max(max_w, lw)
        gap = step_gap if (idx < len(lines) - 1 and ln) else 0
        line_metrics.append((ln, lw, lh, gap))
        total_h += lh + gap
    return max_w, total_h, line_metrics

def draw_autofit_textbox(
    draw,
    text,
    box_coords,
    font_choice,
    max_font_size,
    min_font_size=11,
    fill=(30, 41, 59),
    is_bold=False,
    custom_font_paths=None,
    align="left",
    valign="top",
    line_spacing_ratio=0.32
):
    """
    주어진 사각형 영역(box_coords) 안에 모든 문장이 단 한 글자도 잘리지 않고 들어가도록
    폰트 크기와 줄 간격을 자동으로 축소(Auto-Fit)하여 렌더링합니다.
    """
    if text is None:
        return y_end if (y_end := box_coords[1]) else 0
    raw_str = str(text).strip()
    x1, y1, x2, y2 = box_coords
    if not raw_str:
        return y1

    avail_w = max(24, x2 - x1)
    avail_h = max(18, y2 - y1)

    chosen_font = None
    chosen_metrics = []
    chosen_total_h = 0

    for sz in range(int(max_font_size), int(min_font_size) - 1, -1):
        ratio = line_spacing_ratio if sz > min_font_size + 2 else max(0.16, line_spacing_ratio * 0.65)
        f_cand = load_pil_font(font_choice, sz, is_bold=is_bold, custom_font_paths=custom_font_paths)
        lines = wrap_text_by_pixel(raw_str, f_cand, avail_w, draw)
        mw, th, metrics = _measure_wrapped_block(lines, f_cand, sz, ratio, draw)
        if mw <= avail_w + 2 and th <= avail_h:
            chosen_font = f_cand
            chosen_metrics = metrics
            chosen_total_h = th
            break

    # 최소 크기에서도 세로가 넘칠 경우 초고밀도 간격으로 한 번 더 축소하여 문장 절단 원천 방지
    if chosen_font is None:
        for sz in range(int(min_font_size), 8, -1):
            f_cand = load_pil_font(font_choice, sz, is_bold=is_bold, custom_font_paths=custom_font_paths)
            lines = wrap_text_by_pixel(raw_str, f_cand, avail_w, draw)
            mw, th, metrics = _measure_wrapped_block(lines, f_cand, sz, 0.12, draw)
            chosen_font = f_cand
            chosen_metrics = metrics
            chosen_total_h = th
            if th <= avail_h:
                break

    if valign == "center":
        cur_y = y1 + max(0, (avail_h - chosen_total_h) // 2)
    elif valign == "bottom":
        cur_y = y1 + max(0, avail_h - chosen_total_h)
    else:
        cur_y = y1

    for ln, lw, lh, gap in chosen_metrics:
        if ln:
            if align == "center":
                cur_x = x1 + max(0, (avail_w - lw) // 2)
            elif align == "right":
                cur_x = x1 + max(0, avail_w - lw)
            else:
                cur_x = x1
            bb = draw.textbbox((0, 0), ln, font=chosen_font)
            draw.text((cur_x, cur_y - bb[1]), ln, font=chosen_font, fill=fill)
        cur_y += lh + gap

    return cur_y

# ==========================================
# [6] 문서/동영상 인라인 변환 및 리더 프로필 사진 유틸리티
# ==========================================
def load_default_leader_photos_from_repo():
    photos = {}
    role_map = {
        "생산본부": ["생산본부장", "생산본부", "본부장", "공장장", "leader1", "1_"],
        "유제품생산부문": ["유제품생산부문장", "유제품부문장", "유제품", "leader2", "2_"],
        "상온제품생산부문": ["상온제품생산부문장", "상온부문장", "상온", "leader3", "3_"],
        "품질안전부문": ["품질안전부문장", "품질부문장", "품질안전", "품질", "오재필", "어성현", "leader4", "4_"]
    }

    for root_dir, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for fname in sorted(files):
            lower_f = fname.lower()
            if lower_f.endswith((".png", ".jpg", ".jpeg", ".webp")):
                full_p = os.path.join(root_dir, fname)
                for dept_key, kw_list in role_map.items():
                    if dept_key not in photos and any(kw.lower() in lower_f for kw in kw_list):
                        try:
                            with open(full_p, "rb") as rf:
                                photos[dept_key] = rf.read()
                        except Exception:
                            pass
    return photos

def extract_text_and_Media_from_pptx(file_obj):
    file_bytes = file_obj.getvalue() if hasattr(file_obj, "getvalue") else file_obj.read()
    prs = Presentation(io.BytesIO(file_bytes))
    full_text = []
    extracted_images = []

    for idx, slide in enumerate(prs.slides):
        slide_texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    txt = paragraph.text.strip()
                    if txt:
                        slide_texts.append(txt)
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE or hasattr(shape, "image"):
                try:
                    blob = shape.image.blob
                    im = Image.open(io.BytesIO(blob))
                    if im.width >= 90 and im.height >= 90:
                        extracted_images.append({
                            "slide_idx": idx,
                            "left": int(shape.left or 0),
                            "top": int(shape.top or 0),
                            "width": im.width,
                            "height": im.height,
                            "bytes": blob
                        })
                except Exception:
                    pass
        if slide_texts:
            full_text.append(f"[슬라이드 {idx+1}]\n" + "\n".join(slide_texts))

    return "\n\n".join(full_text), extracted_images

def parse_leader_pptx_text_only(file_obj):
    raw_text, _ = extract_text_and_Media_from_pptx(file_obj)
    texts = {
        "생산본부": "",
        "유제품생산부문": "",
        "상온제품생산부문": "",
        "품질안전부문": ""
    }
    lines = [l.strip() for l in raw_text.split("\n") if l.strip() and not l.startswith("[슬라이드") and "당부말씀" not in l]
    current_key = None
    for line in lines:
        if "생산본부" in line and "부문" not in line:
            current_key = "생산본부"
            continue
        elif "유제품생산부문" in line:
            current_key = "유제품생산부문"
            continue
        elif "상온제품생산부문" in line:
            current_key = "상온제품생산부문"
            continue
        elif "품질안전부문" in line:
            current_key = "품질안전부문"
            continue
        if current_key:
            texts[current_key] = (texts[current_key] + " " + line).strip()

    return texts

def extract_pages_from_pdf(pdf_bytes):
    if not PYMUPDF_AVAILABLE:
        raise RuntimeError("PyMuPDF 라이브러리가 설치되지 않았습니다.")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_info = []
    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.2, 2.2))
        img_bytes = pix.tobytes("png")
        text = page.get_text("text").strip()
        pages_info.append({
            "page_num": page_idx + 1,
            "image_bytes": img_bytes,
            "text": text
        })
    doc.close()
    return pages_info

def build_inline_video_part(video_path):
    raw_size = os.path.getsize(video_path)
    if raw_size <= 15 * 1024 * 1024:
        with open(video_path, "rb") as rf:
            return types.Part.from_bytes(data=rf.read(), mime_type="video/mp4")

    ffmpeg_bin = get_ffmpeg_binary()
    compressed_mp4 = video_path + "_slim.mp4"
    compressed_mp3 = video_path + "_slim.mp3"
    try:
        subprocess.run(
            [
                ffmpeg_bin, "-y", "-i", video_path,
                "-vf", "fps=0.5,scale=640:-2",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "35",
                "-ac", "1", "-ar", "16000", "-b:a", "32k",
                compressed_mp4
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=180
        )
        if os.path.exists(compressed_mp4) and os.path.getsize(compressed_mp4) <= 18 * 1024 * 1024:
            with open(compressed_mp4, "rb") as rf:
                return types.Part.from_bytes(data=rf.read(), mime_type="video/mp4")

        subprocess.run(
            [
                ffmpeg_bin, "-y", "-i", video_path,
                "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k",
                compressed_mp3
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120
        )
        if os.path.exists(compressed_mp3):
            with open(compressed_mp3, "rb") as rf:
                return types.Part.from_bytes(data=rf.read(), mime_type="audio/mpeg")
    finally:
        for p in (compressed_mp4, compressed_mp3):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    raise RuntimeError("동영상 용량이 너무 커서 인라인 변환에 실패했습니다.")

# ==========================================
# [7] 슬라이드 내용 맞춤형 시각 도면 생성 엔진 (gemini-3.1-flash-image 직결)
# ==========================================
def build_tailored_visual_prompt(slide_obj, fallback_topic=""):
    base_vp = str(slide_obj.get("visual_prompt", "")).strip()
    s_title = str(slide_obj.get("title", fallback_topic)).strip()
    cards = slide_obj.get("cards", [])
    card_summary = []
    for c in cards[:3]:
        h_txt = str(c.get("header", "")).strip()
        b_txt = str(c.get("body", "")).replace("\n", " ").strip()[:80]
        if h_txt or b_txt:
            card_summary.append(f"[{h_txt}: {b_txt}]")
    joined_cards = " / ".join(card_summary)

    return (
        f"Slide Topic: {s_title}. "
        f"Key Elements to visually depict together in one coherent scene: {joined_cards}. "
        f"Visual Direction: {base_vp}"
    )

def generate_procedural_infographic(slide_obj, theme_mode, custom_font_paths=None):
    w, h = 840, 760
    is_light = "10월" in theme_mode
    bg_col = (238, 244, 252) if is_light else (13, 25, 43)
    accent = (2, 132, 199) if is_light else (16, 185, 129)
    sub_accent = (30, 64, 175) if is_light else (56, 189, 248)
    ink = (30, 41, 59) if is_light else (226, 232, 240)
    card_fill = (255, 255, 255) if is_light else (19, 36, 62)

    img = Image.new("RGB", (w, h), bg_col)
    draw = ImageDraw.Draw(img)

    for gx in range(0, w, 35):
        draw.line([(gx, 0), (gx, h)], fill=(216, 228, 242) if is_light else (22, 40, 68), width=1)
    for gy in range(0, h, 35):
        draw.line([(0, gy), (w, gy)], fill=(216, 228, 242) if is_light else (22, 40, 68), width=1)

    f_head = load_pil_font("연세제목체", 27, is_bold=True, custom_font_paths=custom_font_paths)
    f_desc = load_pil_font("연세본문체", 20, is_bold=False, custom_font_paths=custom_font_paths)

    s_title = str(slide_obj.get("title", "핵심 위생 관리 도해")).strip()
    cards = slide_obj.get("cards", [])

    draw.rounded_rectangle([(45, 36), (w - 45, 112)], radius=14, fill=sub_accent)
    draw_autofit_textbox(
        draw,
        f"핵심 공정·위생 도해 | {s_title}",
        (65, 44, w - 65, 104),
        "연세제목체",
        max_font_size=27,
        min_font_size=15,
        fill=(255, 255, 255),
        is_bold=True,
        custom_font_paths=custom_font_paths,
        align="center",
        valign="center"
    )

    node_items = cards[:3] if cards else [{"header": s_title, "body": slide_obj.get("content", ""), "metric": "HACCP"}]
    n_cnt = len(node_items)
    start_y = 148
    box_h = 165 if n_cnt == 3 else 220
    gap_y = 26

    for idx_n, item in enumerate(node_items):
        ny1 = start_y + idx_n * (box_h + gap_y)
        ny2 = ny1 + box_h
        nx1, nx2 = 55, w - 55

        draw.rounded_rectangle([(nx1 + 4, ny1 + 4), (nx2 + 4, ny2 + 4)], radius=18, fill=(203, 213, 225) if is_light else (6, 12, 24))
        draw.rounded_rectangle([(nx1, ny1), (nx2, ny2)], radius=18, fill=card_fill, outline=accent, width=3)

        ic_cx, ic_cy, ic_r = nx1 + 72, (ny1 + ny2) // 2, 44
        draw.ellipse([(ic_cx - ic_r, ic_cy - ic_r), (ic_cx + ic_r, ic_cy + ic_r)], fill=accent, outline=sub_accent, width=3)
        num_str = f"0{idx_n + 1}"
        nb = draw.textbbox((0, 0), num_str, font=f_head)
        draw.text((ic_cx - (nb[2] - nb[0]) // 2, ic_cy - 16), num_str, font=f_head, fill=(255, 255, 255))

        h_str = str(item.get("header", f"핵심 요소 {idx_n + 1}")).strip()
        m_str = str(item.get("metric", "")).strip()
        title_line = f"{h_str}  [{m_str}]" if m_str else h_str
        draw_autofit_textbox(
            draw,
            title_line,
            (nx1 + 138, ny1 + 16, nx2 - 22, ny1 + 56),
            "연세제목체",
            max_font_size=24,
            min_font_size=14,
            fill=sub_accent if is_light else (52, 211, 153),
            is_bold=True,
            custom_font_paths=custom_font_paths,
            valign="center"
        )

        raw_b = str(item.get("body", "")).replace("•", "-").strip()
        draw_autofit_textbox(
            draw,
            raw_b,
            (nx1 + 138, ny1 + 62, nx2 - 22, ny2 - 14),
            "연세본문체",
            max_font_size=20,
            min_font_size=12,
            fill=ink,
            is_bold=False,
            custom_font_paths=custom_font_paths,
            valign="center"
        )

        if idx_n < n_cnt - 1:
            mid_x = w // 2
            draw.line([(mid_x, ny2), (mid_x, ny2 + gap_y)], fill=sub_accent, width=4)

    draw.rounded_rectangle([(55, h - 78), (w - 55, h - 28)], radius=10, fill=(224, 242, 254) if is_light else (6, 78, 59), outline=accent, width=2)
    foot_txt = "YONSEI DAIRY FOOD SAFETY VISUAL ARCHITECTURE"
    fb = draw.textbbox((0, 0), foot_txt, font=f_desc)
    draw.text(((w - (fb[2] - fb[0])) // 2, h - 65), foot_txt, font=f_desc, fill=sub_accent if is_light else (255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def generate_ai_illustration(slide_obj, fallback_topic, theme_style, openai_key="", gemini_key="", custom_font_paths=None):
    tailored_prompt = build_tailored_visual_prompt(slide_obj, fallback_topic=fallback_topic)
    style_prefix = (
        "Generate a clean isometric 3D food factory hygiene & equipment technical illustration on a soft light architectural blueprint background, "
        "showing the exact equipment, facility layout, and hygiene actions described below. "
        "Full view centered inside frame with generous margins, strictly NO text, NO letters, NO numbers inside the image. "
        if "10월" in theme_style else
        "Generate a modern dark navy cybernetic isometric 3D food safety engineering illustration with glowing emerald and cyan accents, "
        "showing the exact equipment, facility layout, and hygiene actions described below. "
        "Full view centered inside frame with generous margins, strictly NO text, NO letters, NO numbers inside the image. "
    )
    full_prompt = style_prefix + tailored_prompt

    # 1) 유료 Gemini 정식 이미지 생성 모델 순차 호출
    if gemini_key:
        client = get_genai_client(gemini_key)
        candidate_models = [
            VISUAL_OMNI_MODEL,
            "gemini-3.1-flash-lite-image",
            "gemini-2.5-flash-image"
        ]
        for img_model in candidate_models:
            for retry_idx in range(2):
                try:
                    resp = client.models.generate_content(
                        model=img_model,
                        contents=full_prompt,
                        config=types.GenerateContentConfig(
                            response_modalities=["TEXT", "IMAGE"]
                        )
                    )
                    if resp.candidates and resp.candidates[0].content and resp.candidates[0].content.parts:
                        for part in resp.candidates[0].content.parts:
                            if part.inline_data and part.inline_data.data:
                                data_val = part.inline_data.data
                                return base64.b64decode(data_val) if isinstance(data_val, str) else data_val
                    break
                except Exception as e:
                    if "429" in str(e) and retry_idx == 0:
                        time.sleep(3)
                        continue
                    break

    # 2) OpenAI DALL-E 3 보조 호출
    if openai_key:
        try:
            headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
            payload = {
                "model": "dall-e-3",
                "prompt": full_prompt,
                "n": 1,
                "size": "1024x1024",
                "response_format": "b64_json"
            }
            r = requests.post("https://api.openai.com/v1/images/generations", headers=headers, json=payload, timeout=35)
            if r.status_code == 200:
                b64_str = r.json()["data"][0]["b64_json"]
                return base64.b64decode(b64_str)
        except Exception:
            pass

    return generate_procedural_infographic(slide_obj, theme_style, custom_font_paths=custom_font_paths)

# ==========================================
# [8] 초자연 성우 엔진 및 오프닝·연도 자동 보정기
# ==========================================
def _split_script_chunks(text, max_chars=2600):
    lines = re.split(r'\n+|(?<=[.!?。])\s+', text.strip())
    chunks, cur = [], ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(cur) + len(line) + 1 <= max_chars:
            cur = f"{cur}\n{line}".strip() if cur else line
        else:
            if cur:
                chunks.append(cur)
            cur = line
    if cur:
        chunks.append(cur)
    return chunks if chunks else [text[:max_chars]]

def _pcm_or_wav_to_mp3(raw_audio_bytes, mime_type, output_path, speed_rate=1.0):
    """Gemini Native TTS가 반환하는 24kHz PCM 또는 WAV 오디오를 고음질 MP3로 변환합니다."""
    tmp_wav = output_path + "_raw.wav"
    try:
        if (mime_type and "wav" in mime_type.lower()) or raw_audio_bytes[:4] == b"RIFF":
            with open(tmp_wav, "wb") as wf:
                wf.write(raw_audio_bytes)
        else:
            with wave.open(tmp_wav, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(raw_audio_bytes)

        ffmpeg_bin = get_ffmpeg_binary()
        cmd = [ffmpeg_bin, "-y", "-i", tmp_wav]
        safe_spd = max(0.85, min(1.25, float(speed_rate)))
        if abs(safe_spd - 1.0) >= 0.02:
            cmd.extend(["-filter:a", f"atempo={safe_spd:.2f}"])
        cmd.extend(["-acodec", "libmp3lame", "-q:a", "2", output_path])

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 300
    finally:
        if os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except Exception:
                pass

def _auto_convert_to_two_speaker(script_text):
    """단독 대본일지라도 2인 대화 모드가 켜져 있으면 남성 진행자부터 시작하는 교대 구조로 변환합니다."""
    if "진행자:" in script_text or "파트너:" in script_text:
        return script_text
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', script_text.strip()) if s.strip()]
    if len(sents) <= 1:
        return f"진행자: {script_text}"
    lines = []
    for idx, s in enumerate(sents):
        spk = "진행자" if idx % 2 == 0 else "파트너"
        lines.append(f"{spk}: {s}")
    return "\n".join(lines)

def enforce_opening_greeting(slides_data, dialogue_mode=True):
    """
    모든 생성·확장 완료 직후 첫 번째 슬라이드(slides_data[0])의 첫 문장이
    반드시 '안녕하십니까, 식품안전팀 곽정혁입니다.'(남성 진행자 발화)로 시작하도록 강제 보정합니다.
    """
    if not slides_data:
        return slides_data

    first_sc = str(slides_data[0].get("script", "")).strip()
    compact_head = re.sub(r'[\s,!\.]', '', first_sc[:85])
    has_kwak = ("식품안전팀곽정혁입니다" in compact_head)

    greeting_regex = re.compile(
        r'^(네[,\s]*)?(안녕하십니까|안녕하세요|반갑습니다)[,!\.\s]*'
        r'(연세유업\s*아산공장\s*임직원\s*여러분[,!\.\s]*)?'
        r'(품질안전부문\s*)?식품안전팀\s*곽정혁\s*(주임\s*)?입니다[,!\.\s]*'
    )
    generic_hello_regex = re.compile(r'^(네[,\s]*)?(안녕하십니까|안녕하세요|반갑습니다)[,!\.\s]*')

    if dialogue_mode:
        if first_sc.startswith("파트너:"):
            slides_data[0]["script"] = f"진행자: {MANDATORY_OPENING_LINE}\n{first_sc}"
        elif first_sc.startswith("진행자:"):
            after_tag = first_sc[len("진행자:"):].strip()
            if has_kwak:
                after_tag = greeting_regex.sub("", after_tag).strip()
            else:
                after_tag = generic_hello_regex.sub("", after_tag).strip()
            slides_data[0]["script"] = f"진행자: {MANDATORY_OPENING_LINE} {after_tag}".strip()
        else:
            cleaned = greeting_regex.sub("", first_sc).strip() if has_kwak else generic_hello_regex.sub("", first_sc).strip()
            slides_data[0]["script"] = f"진행자: {MANDATORY_OPENING_LINE} {cleaned}".strip()
    else:
        solo_sc = re.sub(r'^(진행자|파트너)\s*:\s*', '', first_sc).strip()
        if has_kwak:
            solo_sc = greeting_regex.sub("", solo_sc).strip()
        else:
            solo_sc = generic_hello_regex.sub("", solo_sc).strip()
        slides_data[0]["script"] = f"{MANDATORY_OPENING_LINE} {solo_sc}".strip()

    return slides_data

def enforce_edu_year_consistency(slides_data, edu_year_num, edu_month_num):
    """
    AI가 표지(COVER) 제목·부제목이나 대본에서 교육 기준 연도를 2024년·2025년 등 과거 연도로
    잘못 표기한 경우 사용자가 입력한 기준 연도(예: 2026년)로 일괄 교정하고 표지 하단 라벨을 동기화합니다.
    """
    if not slides_data:
        return slides_data

    ym_label = f"{edu_year_num}년 {edu_month_num:02d}월"
    wrong_ym_pattern = re.compile(rf"20\d{{2}}\s*년\s*0?{int(edu_month_num)}\s*월")
    correct_ym_text = f"{edu_year_num}년 {edu_month_num}월"

    for s in slides_data:
        s["edu_ym_label"] = ym_label
        lay = s.get("layout", "")

        # 표지(COVER)의 제목/부제목/본문에 들어간 모든 20XX년 표기를 입력한 교육 연도로 강제 일치
        if lay == "COVER":
            for field in ("title", "subtitle", "content"):
                if s.get(field):
                    s[field] = re.sub(r"20\d{2}\s*년", f"{edu_year_num}년", str(s[field]))

        # 전체 슬라이드 제목·부제목·대본 내 '20XX년 N월' 교육 시기 오기입 교정
        for field in ("title", "subtitle", "script"):
            if s.get(field):
                s[field] = wrong_ym_pattern.sub(correct_ym_text, str(s[field]))

    return slides_data

def create_natural_tts_audio(script_text, output_path, tts_config, return_meta=False):
    """
    1순위: 유료 Gemini Native TTS (남성 진행자 고정 2인 대화 & 남성 단독 성우 지원)
    2순위: OpenAI TTS (남성 중심 보이스)
    3순위: Microsoft Edge Neural TTS (남성 중심 신경망 보이스)
    """
    clean_script = re.sub(r'[\*\#\[\]]', ' ', str(script_text)).strip()
    if not clean_script:
        clean_script = "진행자: 다음 화면의 핵심 위생 수칙을 함께 확인해 보겠습니다."

    engine = tts_config.get("engine", "Gemini TTS")
    gemini_key = tts_config.get("gemini_key", GEMINI_API_KEY).strip()
    gemini_voice = tts_config.get("gemini_voice", "Puck")
    gemini_partner_voice = tts_config.get("gemini_partner_voice", "Kore")
    gemini_dialogue = tts_config.get("gemini_dialogue", True)
    openai_key = tts_config.get("openai_key", "").strip()
    openai_voice = tts_config.get("openai_voice", "echo")
    openai_model = tts_config.get("openai_model", "gpt-4o-mini-tts")
    edge_voice = tts_config.get("edge_voice", "ko-KR-InJoonNeural")
    edge_pitch = tts_config.get("edge_pitch", "+0Hz")
    speed_rate = float(tts_config.get("speed", 1.00))
    err_memo = ""

    # 1) 유료 Gemini Native TTS (NotebookLM 동일 오디오 엔진)
    if "Gemini" in engine and gemini_key:
        client = get_genai_client(gemini_key)
        tts_models = [
            "gemini-2.5-flash-preview-tts",
            "gemini-2.5-pro-preview-tts"
        ]

        if gemini_dialogue:
            dialogue_script = _auto_convert_to_two_speaker(clean_script)
            tts_input_text = (
                "아래 대본을 실제 인기 라디오나 사내 위생교육 방송에서 남성 메인 진행자(식품안전팀 곽정혁)와 보조 파트너가 마주 앉아 생동감 있게 대화하듯 자연스러운 한국어 구어체 호흡으로 읽어주세요:\n\n"
                + dialogue_script
            )
            speech_cfg = types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(
                            speaker="진행자",
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=gemini_voice)
                            )
                        ),
                        types.SpeakerVoiceConfig(
                            speaker="파트너",
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=gemini_partner_voice)
                            )
                        )
                    ]
                )
            )
        else:
            solo_clean = clean_script.replace("진행자:", "").replace("파트너:", "").strip()
            tts_input_text = (
                "실제 사내 위생교육 남성 전문 강사(식품안전팀 곽정혁)가 현장 임직원들과 친근하게 눈을 맞추며 이야기하듯 자연스러운 한국어 구어체 억양과 호흡으로 읽어주세요:\n\n"
                + solo_clean
            )
            speech_cfg = types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=gemini_voice)
                )
            )

        for gm_name in tts_models:
            for retry_i in range(2):
                try:
                    resp = client.models.generate_content(
                        model=gm_name,
                        contents=tts_input_text,
                        config=types.GenerateContentConfig(
                            response_modalities=["AUDIO"],
                            speech_config=speech_cfg
                        )
                    )
                    if resp.candidates and resp.candidates[0].content and resp.candidates[0].content.parts:
                        for part in resp.candidates[0].content.parts:
                            if part.inline_data and part.inline_data.data:
                                raw_b = base64.b64decode(part.inline_data.data) if isinstance(part.inline_data.data, str) else part.inline_data.data
                                m_type = getattr(part.inline_data, "mime_type", "audio/pcm")
                                if _pcm_or_wav_to_mp3(raw_b, m_type, output_path, speed_rate=speed_rate):
                                    mode_tag = f"NotebookLM 2인 대화 (남성진행 {gemini_voice}+파트너 {gemini_partner_voice})" if gemini_dialogue else f"남성 단독 ({gemini_voice})"
                                    meta = {"used_engine": f"Gemini Native TTS ({gm_name} · {mode_tag})", "warning": ""}
                                    return (output_path, meta) if return_meta else output_path
                    break
                except Exception as g_ex:
                    err_memo = f"{gm_name}: {str(g_ex)[:140]}"
                    if "429" in str(g_ex) and retry_i == 0:
                        time.sleep(3)
                        continue
                    break

    # 2) OpenAI TTS 호출
    plain_script = re.sub(r'(진행자|파트너)\s*:\s*', '', clean_script).strip()
    if "OpenAI" in engine and openai_key:
        headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
        chunks = _split_script_chunks(plain_script, max_chars=2600)
        models_to_try = [openai_model, "tts-1-hd"] if openai_model == "gpt-4o-mini-tts" else [openai_model]

        for m_try in models_to_try:
            safe_voice = openai_voice if (m_try == "gpt-4o-mini-tts" or openai_voice not in ("ash", "coral", "sage", "ballad", "verse")) else "echo"
            audio_parts, chunk_ok = [], True
            for ch_txt in chunks:
                payload = {"model": m_try, "input": ch_txt, "voice": safe_voice, "response_format": "mp3", "speed": speed_rate}
                if m_try == "gpt-4o-mini-tts":
                    payload["instructions"] = "실제 사내 위생교육 전문 남성 강사처럼 자연스럽고 생동감 있는 한국어 구어체로 대화하듯 말하세요."
                try:
                    resp = requests.post("https://api.openai.com/v1/audio/speech", headers=headers, json=payload, timeout=90)
                    if resp.status_code == 400 and "speed" in payload:
                        payload.pop("speed", None)
                        resp = requests.post("https://api.openai.com/v1/audio/speech", headers=headers, json=payload, timeout=90)
                    if resp.status_code == 200 and len(resp.content) > 300:
                        audio_parts.append(resp.content)
                    else:
                        err_memo = f"HTTP {resp.status_code}"
                        chunk_ok = False
                        break
                except Exception as ex:
                    err_memo = str(ex)
                    chunk_ok = False
                    break
            if chunk_ok and audio_parts:
                with open(output_path, "wb") as f:
                    for part_bytes in audio_parts:
                        f.write(part_bytes)
                meta = {"used_engine": f"OpenAI TTS ({m_try} / {safe_voice})", "warning": ""}
                return (output_path, meta) if return_meta else output_path

    # 3) Edge Neural TTS 폴백 (남성 보이스 기본)
    if EDGE_TTS_AVAILABLE:
        try:
            pct = int(round((speed_rate - 1.0) * 100))
            rate_str = f"+{pct}%" if pct >= 0 else f"{pct}%"

            async def _save_edge():
                if edge_pitch and edge_pitch != "+0Hz":
                    try:
                        comm = edge_tts.Communicate(plain_script, voice=edge_voice, rate=rate_str, pitch=edge_pitch)
                        await comm.save(output_path)
                        return
                    except TypeError:
                        pass
                comm = edge_tts.Communicate(plain_script, voice=edge_voice, rate=rate_str)
                await comm.save(output_path)

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_save_edge())
            finally:
                loop.close()

            if os.path.exists(output_path) and os.path.getsize(output_path) > 500:
                warn_txt = f"선택 엔진 응답 오류({err_memo})로 남성 신경망 성우({edge_voice})로 전환되었습니다." if "Edge" not in engine else ""
                meta = {"used_engine": f"Edge Neural TTS ({edge_voice})", "warning": warn_txt}
                return (output_path, meta) if return_meta else output_path
        except Exception as edge_ex:
            err_memo = f"{err_memo} | {edge_ex}"

    tts = gTTS(text=plain_script, lang="ko", slow=False)
    tts.save(output_path)
    meta = {"used_engine": "Google 기본 TTS", "warning": err_memo}
    return (output_path, meta) if return_meta else output_path

# ==========================================
# [9] Full HD (1920x1080) 고해상도 슬라이드 렌더링 엔진 (전 구간 Auto-Fit 잘림 0%)
# ==========================================
def paste_image_contain(base_img, img_bytes, box_coords, bg_color=(248, 250, 252)):
    x1, y1, x2, y2 = box_coords
    target_w = max(10, x2 - x1)
    target_h = max(10, y2 - y1)
    try:
        src = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        contained = ImageOps.contain(src, (target_w, target_h), method=Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (target_w, target_h), bg_color)
        offset_x = (target_w - contained.width) // 2
        offset_y = (target_h - contained.height) // 2
        canvas.paste(contained, (offset_x, offset_y))
        base_img.paste(canvas, (x1, y1))
        return True
    except Exception:
        return False

def paste_circular_avatar(base_img, photo_bytes, center_x, center_y, radius, border_color, border_width=6):
    draw = ImageDraw.Draw(base_img)
    diam = radius * 2
    if photo_bytes:
        try:
            avatar = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
            avatar = ImageOps.fit(avatar, (diam, diam), method=Image.Resampling.LANCZOS, centering=(0.5, 0.35))
            mask = Image.new("L", (diam, diam), 0)
            m_draw = ImageDraw.Draw(mask)
            m_draw.ellipse((0, 0, diam, diam), fill=255)
            base_img.paste(avatar, (center_x - radius, center_y - radius), mask)
            draw.ellipse(
                [(center_x - radius, center_y - radius), (center_x + radius, center_y + radius)],
                outline=border_color,
                width=border_width
            )
            return True
        except Exception:
            pass
    return False

def draw_background_grid_hd(draw, width, height, theme_mode):
    if "10월" in theme_mode:
        draw.rectangle([(0, 0), (width, height)], fill=(244, 247, 251))
        for x in range(0, width, 36):
            draw.line([(x, 0), (x, height)], fill=(224, 232, 243), width=1)
        for y in range(0, height, 36):
            draw.line([(0, y), (width, y)], fill=(224, 232, 243), width=1)
        for x in range(0, width, 180):
            draw.line([(x, 0), (x, height)], fill=(203, 216, 235), width=1)
        for y in range(0, height, 180):
            draw.line([(0, y), (width, y)], fill=(203, 216, 235), width=1)
    else:
        draw.rectangle([(0, 0), (width, height)], fill=(9, 17, 30))
        for x in range(0, width, 54):
            draw.line([(x, 0), (x, height)], fill=(16, 30, 52), width=1)
        for y in range(0, height, 54):
            draw.line([(0, y), (width, y)], fill=(16, 30, 52), width=1)
        draw.rectangle([(26, 26), (width - 26, height - 26)], outline=(16, 185, 129), width=3)
        draw.rectangle([(36, 36), (width - 36, height - 36)], outline=(6, 95, 70), width=1)

def create_slide_image_hd(slide_data, slide_idx, total_slides, output_path, font_config, theme_mode, leader_photos=None):
    width, height = 1920, 1080

    if slide_data.get("custom_image_bytes"):
        img = Image.open(io.BytesIO(slide_data["custom_image_bytes"])).convert("RGB")
        img = ImageOps.contain(img, (width, height), method=Image.Resampling.LANCZOS)
        bg_canvas = Image.new("RGB", (width, height), (255, 255, 255))
        bg_canvas.paste(img, ((width - img.width) // 2, (height - img.height) // 2))
        bg_canvas.save(output_path, quality=95)
        return output_path

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    title_font_name = font_config.get("title_font", "연세제목체")
    body_font_name = font_config.get("body_font", "연세본문체")
    custom_paths = font_config.get("custom_paths", {})

    f_hero = load_pil_font(title_font_name, 70, is_bold=True, custom_font_paths=custom_paths)
    f_quote = load_pil_font(title_font_name, 64, is_bold=True, custom_font_paths=custom_paths)
    f_card_title = load_pil_font(title_font_name, 30, is_bold=True, custom_font_paths=custom_paths)
    f_badge_dept = load_pil_font(title_font_name, 21, is_bold=True, custom_font_paths=custom_paths)
    f_badge_role = load_pil_font(title_font_name, 27, is_bold=True, custom_font_paths=custom_paths)
    f_small = load_pil_font(body_font_name, 22, is_bold=False, custom_font_paths=custom_paths)

    layout = slide_data.get("layout", "SPLIT_VISUAL")
    is_light = "10월" in theme_mode
    leader_photos = leader_photos or {}

    # 1) 당부말씀 슬라이드 (긴 말씀도 100% 다 들어가도록 Auto-Fit 적용)
    if layout in ("INTRO_1TOP_3BOT", "INTRO_4GRID"):
        draw.rectangle([(0, 0), (width, height)], fill=(244, 246, 250))
        title_txt = slide_data.get("title", "월간 위생교육 전 당부말씀")
        draw_autofit_textbox(
            draw,
            title_txt,
            (62, 26, 1858, 104),
            title_font_name,
            max_font_size=48,
            min_font_size=26,
            fill=(15, 23, 42),
            is_bold=True,
            custom_font_paths=custom_paths,
            valign="center"
        )

        cards = slide_data.get("cards", [])
        top_card = cards[0] if len(cards) > 0 else {"header": "생산본부", "sub": "생산본부장", "body": ""}

        tx1, ty1, tx2, ty2 = 60, 114, 1860, 400
        red_col = (220, 38, 38)
        draw.rounded_rectangle([(tx1 + 4, ty1 + 5), (tx2 + 4, ty2 + 5)], radius=20, fill=(215, 222, 232))
        draw.rounded_rectangle([(tx1, ty1), (tx2, ty2)], radius=20, fill=(255, 255, 255), outline=(218, 226, 236), width=3)
        draw.rounded_rectangle([(tx1, ty1), (tx1 + 15, ty2)], radius=7, fill=red_col)

        cx_top, cy_top, r_top = tx1 + 115, (ty1 + ty2) // 2, 78
        has_top_photo = paste_circular_avatar(img, leader_photos.get("생산본부"), cx_top, cy_top, r_top, red_col, border_width=6)

        dept_top = top_card.get("header", "생산본부")
        role_top = top_card.get("sub", "생산본부장")
        if has_top_photo:
            draw.text((tx1 + 212, cy_top - 32), dept_top, font=f_badge_dept, fill=(100, 116, 139))
            draw.text((tx1 + 212, cy_top - 2), role_top, font=f_badge_role, fill=(15, 23, 42))
            divider_x = tx1 + 375
        else:
            draw.ellipse([(cx_top - r_top, cy_top - r_top), (cx_top + r_top, cy_top + r_top)], fill=(255, 255, 255), outline=red_col, width=6)
            b1 = draw.textbbox((0, 0), dept_top, font=f_badge_dept)
            b2 = draw.textbbox((0, 0), role_top, font=f_badge_role)
            draw.text((cx_top - (b1[2] - b1[0]) // 2, cy_top - 28), dept_top, font=f_badge_dept, fill=(100, 116, 139))
            draw.text((cx_top - (b2[2] - b2[0]) // 2, cy_top + 4), role_top, font=f_badge_role, fill=(15, 23, 42))
            divider_x = tx1 + 230

        draw.line([(divider_x, ty1 + 32), (divider_x, ty2 - 32)], fill=(226, 232, 240), width=3)
        draw.text((divider_x + 18, ty1 + 16), "“", font=f_quote, fill=(203, 213, 225))

        draw_autofit_textbox(
            draw,
            top_card.get("body", ""),
            (divider_x + 70, ty1 + 24, tx2 - 32, ty2 - 22),
            body_font_name,
            max_font_size=23,
            min_font_size=12,
            fill=(30, 41, 59),
            is_bold=False,
            custom_font_paths=custom_paths,
            valign="center",
            line_spacing_ratio=0.34
        )

        bottom_defaults = [
            ("유제품생산부문", "유제품생산부문장"),
            ("상온제품생산부문", "상온제품생산부문장"),
            ("품질안전부문", "품질안전부문장")
        ]
        navy_col = (15, 62, 124)
        card_w = 576
        gap_x = 36
        by1, by2 = 424, 1044

        for i in range(3):
            bx1 = 60 + i * (card_w + gap_x)
            bx2 = bx1 + card_w
            def_dept, def_role = bottom_defaults[i]
            c_data = cards[i + 1] if (i + 1) < len(cards) else {}
            dept_str = c_data.get("header", def_dept)
            role_str = c_data.get("sub", def_role)
            body_str = c_data.get("body", "")

            draw.rounded_rectangle([(bx1 + 4, by1 + 5), (bx2 + 4, by2 + 5)], radius=20, fill=(215, 222, 232))
            draw.rounded_rectangle([(bx1, by1), (bx2, by2)], radius=20, fill=(255, 255, 255), outline=(218, 226, 236), width=3)
            draw.rounded_rectangle([(bx1, by1), (bx1 + 14, by2)], radius=7, fill=navy_col)

            cx_b, cy_b, r_b = bx1 + 84, by1 + 76, 52
            has_b_photo = paste_circular_avatar(img, leader_photos.get(def_dept), cx_b, cy_b, r_b, navy_col, border_width=5)
            if not has_b_photo:
                draw.ellipse([(cx_b - r_b, cy_b - r_b), (cx_b + r_b, cy_b + r_b)], fill=(255, 255, 255), outline=navy_col, width=5)
                short_tag = "유제품" if "유제품" in dept_str else ("상온" if "상온" in dept_str else "품질")
                bb_tag = draw.textbbox((0, 0), short_tag, font=f_badge_dept)
                draw.text((cx_b - (bb_tag[2] - bb_tag[0]) // 2, cy_b - 12), short_tag, font=f_badge_dept, fill=navy_col)

            draw.text((bx1 + 154, by1 + 40), dept_str, font=f_badge_dept, fill=(100, 116, 139))
            draw.text((bx1 + 154, by1 + 70), role_str, font=f_card_title, fill=(15, 23, 42))
            draw.text((bx2 - 68, by1 + 20), "“", font=f_quote, fill=(203, 213, 225))
            draw.line([(bx1 + 32, by1 + 142), (bx2 - 32, by1 + 142)], fill=(235, 240, 246), width=2)

            draw_autofit_textbox(
                draw,
                body_str,
                (bx1 + 34, by1 + 156, bx2 - 28, by2 - 22),
                body_font_name,
                max_font_size=22,
                min_font_size=11,
                fill=(45, 55, 72),
                is_bold=False,
                custom_font_paths=custom_paths,
                valign="top",
                line_spacing_ratio=0.34
            )

        img.save(output_path, quality=95)
        return output_path

    # 2) 이벤트 안내 고정 양식 슬라이드
    if layout == "OUTRO_EVENT":
        draw_background_grid_hd(draw, width, height, "10월 스타일")

        draw.rectangle([(72, 78), (1848, 1002)], outline=(30, 41, 59), width=3)
        draw.rectangle([(82, 88), (1838, 992)], outline=(100, 116, 139), width=2)

        cx1, cy1, cx2, cy2 = 240, 144, 1680, 936
        draw.rectangle([(cx1 + 12, cy1 + 12), (cx2 + 12, cy2 + 12)], fill=(203, 213, 225))
        draw.rectangle([(cx1, cy1), (cx2, cy2)], fill=(255, 255, 255), outline=(30, 41, 59), width=3)

        ev_month = slide_data.get("event_month", "11월")
        ev_ranks = slide_data.get("event_ranks", "11번째, 41번째, 71번째")
        ev_prize_hi = slide_data.get("event_prize_highlight", "10,000원")
        ev_prize_rest = slide_data.get("event_prize_rest", " 상당의 커피 기프티콘 증정")
        ev_closing = slide_data.get("event_closing", "많은 관심과 적극적인 참여 부탁드립니다.")

        main_t = slide_data.get("title", f"{ev_month} 위생교육 이벤트 안내")
        bb_t = draw.textbbox((0, 0), main_t, font=f_hero)
        draw.text(((width - (bb_t[2] - bb_t[0])) // 2, cy1 + 72), main_t, font=f_hero, fill=(15, 23, 42))

        sub_t = slide_data.get("subtitle", f"마지막으로 {ev_month} 위생교육 이벤트 안내입니다!")
        bb_s = draw.textbbox((0, 0), sub_t, font=f_card_title)
        draw.text(((width - (bb_s[2] - bb_s[0])) // 2, cy1 + 176), sub_t, font=f_card_title, fill=(100, 116, 139))

        bar_x1, bar_x2 = 360, 1560
        bar_h = 110
        bar_ys = [412, 566, 720]
        f_ev_row = load_pil_font(title_font_name, 36, is_bold=True, custom_font_paths=custom_paths)

        for idx_b, by in enumerate(bar_ys):
            draw.rounded_rectangle([(bar_x1, by), (bar_x2, by + bar_h)], radius=14, fill=(248, 250, 252))
            draw.rounded_rectangle([(bar_x1, by), (bar_x1 + 12, by + bar_h)], radius=6, fill=(30, 41, 59))

            ic_x, ic_y = bar_x1 + 62, by + 30
            if idx_b == 0:
                draw.rounded_rectangle([(ic_x, ic_y), (ic_x + 42, ic_y + 50)], radius=6, fill=(71, 85, 105))
                draw.rectangle([(ic_x + 10, ic_y - 6), (ic_x + 32, ic_y + 6)], fill=(30, 41, 59))
                draw.line([(ic_x + 10, ic_y + 26), (ic_x + 18, ic_y + 36), (ic_x + 32, ic_y + 18)], fill=(255, 255, 255), width=4)
            elif idx_b == 1:
                draw.rectangle([(ic_x, ic_y + 15), (ic_x + 44, ic_y + 50)], fill=(71, 85, 105))
                draw.rectangle([(ic_x - 3, ic_y + 5), (ic_x + 47, ic_y + 18)], fill=(51, 65, 85))
                draw.line([(ic_x + 22, ic_y + 5), (ic_x + 22, ic_y + 50)], fill=(255, 255, 255), width=4)
            else:
                draw.ellipse([(ic_x + 12, ic_y), (ic_x + 36, ic_y + 24)], fill=(71, 85, 105))
                draw.rounded_rectangle([(ic_x, ic_y + 30), (ic_x + 48, ic_y + 45)], radius=6, fill=(71, 85, 105))

            tx_start = bar_x1 + 142
            ty_text = by + 33
            if idx_b == 0:
                seg1 = f"{ev_month}은 만점자 중 순번 "
                seg2 = ev_ranks
                seg3 = " 분에게"
                draw.text((tx_start, ty_text), seg1, font=f_ev_row, fill=(30, 41, 59))
                w1 = draw.textbbox((0, 0), seg1, font=f_ev_row)[2]
                draw.text((tx_start + w1, ty_text), seg2, font=f_ev_row, fill=(220, 38, 38))
                w2 = draw.textbbox((0, 0), seg2, font=f_ev_row)[2]
                draw.text((tx_start + w1 + w2, ty_text), seg3, font=f_ev_row, fill=(30, 41, 59))
            elif idx_b == 1:
                seg1 = ev_prize_hi
                seg2 = ev_prize_rest
                draw.text((tx_start, ty_text), seg1, font=f_ev_row, fill=(220, 38, 38))
                w1 = draw.textbbox((0, 0), seg1, font=f_ev_row)[2]
                draw.text((tx_start + w1, ty_text), seg2, font=f_ev_row, fill=(30, 41, 59))
            else:
                draw_autofit_textbox(
                    draw,
                    ev_closing,
                    (tx_start, by + 18, bar_x2 - 24, by + bar_h - 18),
                    title_font_name,
                    max_font_size=36,
                    min_font_size=20,
                    fill=(30, 41, 59),
                    is_bold=True,
                    custom_font_paths=custom_paths,
                    valign="center"
                )

        img.save(output_path, quality=95)
        return output_path

    # 3) 표지 및 본론 슬라이드 (모든 박스·표·하단 배너 Auto-Fit 적용)
    draw_background_grid_hd(draw, width, height, theme_mode)
    text_main_col = (15, 23, 42) if is_light else (248, 250, 252)
    text_sub_col = (51, 65, 85) if is_light else (203, 213, 225)
    card_bg_col = (255, 255, 255) if is_light else (15, 28, 48)
    card_border_col = (30, 64, 175) if is_light else (16, 185, 129)

    if layout == "COVER":
        box_x1, box_y1, box_x2, box_y2 = 110, 150, 1810, 930
        draw.rounded_rectangle([(box_x1 + 8, box_y1 + 8), (box_x2 + 8, box_y2 + 8)], radius=26, fill=(203, 213, 225) if is_light else (5, 10, 20))
        draw.rounded_rectangle([(box_x1, box_y1), (box_x2, box_y2)], radius=26, fill=card_bg_col, outline=card_border_col, width=5)
        draw.rectangle([(box_x1, box_y1), (box_x1 + 32, box_y2)], fill=card_border_col)

        has_cover_vis = bool(slide_data.get("visual_bytes"))
        text_max_w = 960 if has_cover_vis else 1500

        badge_txt = "YONSEI DAIRY | FOOD SAFETY & HACCP EDUCATION"
        draw.text((box_x1 + 85, box_y1 + 76), badge_txt, font=f_card_title, fill=(2, 132, 199) if is_light else (52, 211, 153))

        main_title = slide_data.get("title", "정기 위생교육")
        title_end_y = draw_autofit_textbox(
            draw,
            main_title,
            (box_x1 + 85, box_y1 + 150, box_x1 + 85 + text_max_w, box_y1 + 460),
            title_font_name,
            max_font_size=68,
            min_font_size=32,
            fill=text_main_col,
            is_bold=True,
            custom_font_paths=custom_paths,
            valign="center",
            line_spacing_ratio=0.25
        )

        sub_txt = slide_data.get("subtitle", "") or slide_data.get("content", "")
        if sub_txt:
            draw_autofit_textbox(
                draw,
                sub_txt,
                (box_x1 + 85, max(box_y1 + 470, title_end_y + 20), box_x1 + 85 + text_max_w, box_y2 - 155),
                title_font_name,
                max_font_size=33,
                min_font_size=18,
                fill=text_sub_col,
                is_bold=True,
                custom_font_paths=custom_paths,
                valign="top"
            )

        if has_cover_vis:
            cx1, cy1, cx2, cy2 = box_x2 - 610, box_y1 + 90, box_x2 - 70, box_y1 + 630
            paste_image_contain(img, slide_data["visual_bytes"], (cx1, cy1, cx2, cy2), bg_color=(238, 244, 252) if is_light else (13, 25, 43))
            draw.rectangle([(cx1, cy1), (cx2, cy2)], outline=card_border_col, width=4)

        draw.line([(box_x1 + 85, box_y2 - 135), (box_x2 - 85, box_y2 - 135)], fill=card_border_col, width=3)
        ym_label = slide_data.get("edu_ym_label") or datetime.date.today().strftime('%Y년 %m월')
        footer_info = f"품질안전부문 식품안전팀 곽정혁  |  {ym_label}"
        draw.text((box_x1 + 85, box_y2 - 92), footer_info, font=f_card_title, fill=text_main_col)
        img.save(output_path, quality=95)
        return output_path

    # 공통 상단 타이틀 바 (긴 제목도 잘리지 않도록 Auto-Fit)
    header_bg = (30, 58, 138) if is_light else (15, 28, 48)
    draw.rectangle([(60, 44), (1860, 150)], fill=header_bg, outline=card_border_col, width=3)
    draw.rectangle([(60, 44), (82, 150)], fill=(56, 189, 248) if is_light else (16, 185, 129))
    draw_autofit_textbox(
        draw,
        slide_data.get("title", f"슬라이드 {slide_idx + 1}"),
        (108, 52, 1838, 142),
        title_font_name,
        max_font_size=46,
        min_font_size=24,
        fill=(255, 255, 255),
        is_bold=True,
        custom_font_paths=custom_paths,
        valign="center"
    )

    has_bottom_banner = bool(str(slide_data.get("bottom_summary", "")).strip())
    content_bottom_y = 964 if has_bottom_banner else 994

    if layout == "TABLE_MATRIX" and slide_data.get("table_headers") and slide_data.get("table_rows"):
        headers = slide_data["table_headers"]
        rows = slide_data["table_rows"]
        col_cnt = max(1, len(headers))
        table_x1, table_y1, table_x2, table_y2 = 65, 178, 1855, content_bottom_y
        col_w = (table_x2 - table_x1) // col_cnt

        th_h = 80
        for c_i, h_text in enumerate(headers):
            cx1 = table_x1 + c_i * col_w
            cx2 = table_x2 if c_i == col_cnt - 1 else (cx1 + col_w)
            draw.rectangle([(cx1, table_y1), (cx2, table_y1 + th_h)], fill=(30, 64, 175) if is_light else (6, 78, 59), outline=card_border_col, width=3)
            draw_autofit_textbox(
                draw,
                str(h_text),
                (cx1 + 18, table_y1 + 10, cx2 - 18, table_y1 + th_h - 10),
                title_font_name,
                max_font_size=28,
                min_font_size=15,
                fill=(255, 255, 255),
                is_bold=True,
                custom_font_paths=custom_paths,
                valign="center"
            )

        row_cnt = max(1, len(rows))
        avail_h = table_y2 - (table_y1 + th_h)
        row_h = avail_h // row_cnt
        for r_i, r_data in enumerate(rows):
            ry1 = table_y1 + th_h + r_i * row_h
            ry2 = table_y2 if r_i == row_cnt - 1 else (ry1 + row_h)
            row_bg = (255, 255, 255) if (is_light and r_i % 2 == 0) else ((238, 244, 252) if is_light else (17, 32, 56))
            for c_i in range(col_cnt):
                cx1 = table_x1 + c_i * col_w
                cx2 = table_x2 if c_i == col_cnt - 1 else (cx1 + col_w)
                cell_txt = str(r_data[c_i]) if c_i < len(r_data) else ""
                draw.rectangle([(cx1, ry1), (cx2, ry2)], fill=row_bg, outline=card_border_col, width=2)
                draw_autofit_textbox(
                    draw,
                    cell_txt,
                    (cx1 + 18, ry1 + 10, cx2 - 18, ry2 - 10),
                    title_font_name if c_i == 0 else body_font_name,
                    max_font_size=27 if c_i == 0 else 25,
                    min_font_size=12,
                    fill=text_main_col if c_i == 0 else text_sub_col,
                    is_bold=(c_i == 0),
                    custom_font_paths=custom_paths,
                    valign="center"
                )

    elif layout == "SPLIT_VISUAL" or slide_data.get("visual_bytes"):
        vx1, vy1, vx2, vy2 = 65, 178, 915, content_bottom_y
        draw.rounded_rectangle([(vx1, vy1), (vx2, vy2)], radius=18, fill=card_bg_col, outline=card_border_col, width=4)

        v_bytes = slide_data.get("visual_bytes")
        if not v_bytes:
            v_bytes = generate_procedural_infographic(slide_data, theme_mode, custom_font_paths=custom_paths)

        inner_pad = 16
        paste_image_contain(
            img,
            v_bytes,
            (vx1 + inner_pad, vy1 + inner_pad, vx2 - inner_pad, vy2 - inner_pad),
            bg_color=(238, 244, 252) if is_light else (13, 25, 43)
        )

        cards = slide_data.get("cards", [])
        if not cards:
            cards = [{"header": "핵심 위생 관리 포인트", "body": slide_data.get("content", "")}]
        rx1, rx2 = 950, 1855
        c_cnt = min(len(cards), 3)
        gap = 22
        total_cards_h = content_bottom_y - 178
        card_h = (total_cards_h - gap * (max(1, c_cnt) - 1)) // max(1, c_cnt)
        for i, c_item in enumerate(cards[:c_cnt]):
            cy1 = 178 + i * (card_h + gap)
            cy2 = cy1 + card_h
            draw.rounded_rectangle([(rx1, cy1), (rx2, cy2)], radius=16, fill=card_bg_col, outline=card_border_col, width=3)
            draw.rectangle([(rx1, cy1), (rx1 + 18, cy2)], fill=card_border_col)

            header_str = c_item.get("header", f"핵심 기준 {i+1}")
            metric_str = str(c_item.get("metric", "")).strip()
            header_right_x = rx2 - 24

            if metric_str:
                # 뱃지 폭이 너무 넓어 제목을 침범하지 않도록 최대 290px 내 자동 맞춤
                f_badge_m = load_pil_font(title_font_name, 25, is_bold=True, custom_font_paths=custom_paths)
                m_bb = draw.textbbox((0, 0), metric_str, font=f_badge_m)
                mw = min(290, (m_bb[2] - m_bb[0]))
                bx_left = rx2 - mw - 52
                draw.rounded_rectangle([(bx_left, cy1 + 14), (rx2 - 20, cy1 + 60)], radius=8, fill=(254, 226, 226) if is_light else (127, 29, 29))
                draw_autofit_textbox(
                    draw,
                    metric_str,
                    (bx_left + 12, cy1 + 17, rx2 - 32, cy1 + 57),
                    title_font_name,
                    max_font_size=25,
                    min_font_size=14,
                    fill=(220, 38, 38) if is_light else (254, 202, 202),
                    is_bold=True,
                    custom_font_paths=custom_paths,
                    align="center",
                    valign="center"
                )
                header_right_x = bx_left - 14

            draw_autofit_textbox(
                draw,
                header_str,
                (rx1 + 36, cy1 + 14, max(rx1 + 180, header_right_x), cy1 + 64),
                title_font_name,
                max_font_size=29,
                min_font_size=16,
                fill=card_border_col if is_light else (52, 211, 153),
                is_bold=True,
                custom_font_paths=custom_paths,
                valign="center"
            )

            draw_autofit_textbox(
                draw,
                c_item.get("body", ""),
                (rx1 + 36, cy1 + 70, rx2 - 26, cy2 - 16),
                body_font_name,
                max_font_size=25,
                min_font_size=12,
                fill=text_sub_col,
                is_bold=False,
                custom_font_paths=custom_paths,
                valign="top",
                line_spacing_ratio=0.30
            )

    else:
        cards = slide_data.get("cards", [])
        if not cards:
            raw_lines = [l.strip() for l in slide_data.get("content", "").split("\n") if l.strip()]
            cards = [{"header": f"핵심 포인트 {i+1}", "body": line, "metric": ""} for i, line in enumerate(raw_lines[:4])]

        c_cnt = min(len(cards), 5)
        if c_cnt == 4 and layout == "SELF_AUDIT_2X2":
            mid_y = (178 + content_bottom_y) // 2
            grid_coords = [
                (65, 178, 940, mid_y - 12),
                (980, 178, 1855, mid_y - 12),
                (65, mid_y + 12, 940, content_bottom_y),
                (980, mid_y + 12, 1855, content_bottom_y)
            ]
            for i in range(4):
                gx1, gy1, gx2, gy2 = grid_coords[i]
                c_item = cards[i]
                draw.rounded_rectangle([(gx1, gy1), (gx2, gy2)], radius=18, fill=card_bg_col, outline=card_border_col, width=3)
                draw.rectangle([(gx1, gy1), (gx2, gy1 + 66)], fill=header_bg)
                draw_autofit_textbox(
                    draw,
                    c_item.get("header", ""),
                    (gx1 + 26, gy1 + 10, gx2 - 26, gy1 + 58),
                    title_font_name,
                    max_font_size=29,
                    min_font_size=16,
                    fill=(255, 255, 255),
                    is_bold=True,
                    custom_font_paths=custom_paths,
                    valign="center"
                )
                draw_autofit_textbox(
                    draw,
                    c_item.get("body", ""),
                    (gx1 + 28, gy1 + 78, gx2 - 26, gy2 - 16),
                    body_font_name,
                    max_font_size=25,
                    min_font_size=12,
                    fill=text_sub_col,
                    is_bold=False,
                    custom_font_paths=custom_paths,
                    valign="top",
                    line_spacing_ratio=0.30
                )
        else:
            total_w = 1790
            gap = 28
            card_w = (total_w - gap * (max(1, c_cnt) - 1)) // max(1, c_cnt)
            for i, c_item in enumerate(cards[:c_cnt]):
                cx1 = 65 + i * (card_w + gap)
                cx2 = cx1 + card_w
                cy1, cy2 = 178, content_bottom_y
                draw.rounded_rectangle([(cx1, cy1), (cx2, cy2)], radius=18, fill=card_bg_col, outline=card_border_col, width=4)
                draw.rectangle([(cx1, cy1), (cx2, cy1 + 84)], fill=header_bg)
                draw_autofit_textbox(
                    draw,
                    c_item.get("header", f"STEP {i+1}"),
                    (cx1 + 20, cy1 + 10, cx2 - 20, cy1 + 76),
                    title_font_name,
                    max_font_size=29,
                    min_font_size=15,
                    fill=(255, 255, 255),
                    is_bold=True,
                    custom_font_paths=custom_paths,
                    valign="center"
                )

                body_top_y = cy1 + 104
                metric_str = str(c_item.get("metric", "")).strip()
                if metric_str:
                    draw_autofit_textbox(
                        draw,
                        metric_str,
                        (cx1 + 24, cy1 + 98, cx2 - 24, cy1 + 220),
                        title_font_name,
                        max_font_size=56,
                        min_font_size=22,
                        fill=(239, 68, 68),
                        is_bold=True,
                        custom_font_paths=custom_paths,
                        valign="center"
                    )
                    draw.line([(cx1 + 24, cy1 + 232), (cx2 - 24, cy1 + 232)], fill=card_border_col, width=3)
                    body_top_y = cy1 + 250

                draw_autofit_textbox(
                    draw,
                    c_item.get("body", ""),
                    (cx1 + 24, body_top_y, cx2 - 22, cy2 - 18),
                    body_font_name,
                    max_font_size=25,
                    min_font_size=12,
                    fill=text_sub_col,
                    is_bold=False,
                    custom_font_paths=custom_paths,
                    valign="top",
                    line_spacing_ratio=0.30
                )

    # 하단 핵심 원칙 배너 (긴 문장도 2~3줄 자동 축소·중앙 정렬되어 끝까지 100% 표시)
    bottom_banner = str(slide_data.get("bottom_summary", "")).strip()
    if bottom_banner:
        draw.rounded_rectangle(
            [(65, 978), (1685, 1064)],
            radius=12,
            fill=(254, 242, 242) if is_light else (6, 78, 59),
            outline=(239, 68, 68) if is_light else (16, 185, 129),
            width=3
        )
        draw_autofit_textbox(
            draw,
            f"핵심 원칙: {bottom_banner}",
            (90, 984, 1665, 1058),
            body_font_name,
            max_font_size=25,
            min_font_size=13,
            fill=(185, 28, 28) if is_light else (255, 255, 255),
            is_bold=False,
            custom_font_paths=custom_paths,
            valign="center",
            line_spacing_ratio=0.22
        )

    draw.text((width - 205, height - 50), f"Page {slide_idx + 1} / {total_slides}", font=f_small, fill=text_sub_col)
    img.save(output_path, quality=95)
    return output_path

# ==========================================
# [10] 고해상도 이미지 기반 파워포인트(.pptx) 생성기
# ==========================================
def generate_highres_image_pptx(slides_data, font_config, theme_mode, leader_photos=None):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    total_cnt = len(slides_data)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for idx, slide_data in enumerate(slides_data):
            slide = prs.slides.add_slide(blank_layout)
            img_path = os.path.join(tmp_dir, f"hd_slide_{idx:03d}.png")
            create_slide_image_hd(slide_data, idx, total_cnt, img_path, font_config, theme_mode, leader_photos=leader_photos)

            slide.shapes.add_picture(
                img_path,
                Inches(0),
                Inches(0),
                width=Inches(13.333),
                height=Inches(7.5)
            )
            notes_slide = slide.notes_slide
            notes_slide.notes_text_frame.text = f"[성우 내레이션 대본]\n{slide_data.get('script', '')}"

    pptx_io = io.BytesIO()
    prs.save(pptx_io)
    pptx_io.seek(0)
    return pptx_io.getvalue()

def build_fixed_event_slide(month_int, rank1, rank2, rank3, prize_amount="10,000원", prize_item="상당의 커피 기프티콘 증정", dialogue_mode=True, year_int=None):
    month_str = f"{month_int}월"
    ym_full = f"{year_int}년 {month_str}" if year_int else month_str
    ranks_str = f"{rank1}번째, {rank2}번째, {rank3}번째"
    if dialogue_mode:
        script_text = (
            f"진행자: 자, 오늘 함께 살펴본 {ym_full} 정기 위생교육 내용은 여기까지입니다! 현장에서 늘 애써주시는 아산공장 임직원 여러분 정말 고생 많으셨습니다.\n"
            f"파트너: 맞습니다! 그리고 여러분이 가장 기다리시는 시간이죠, {month_str} 위생교육 퀴즈 이벤트 소식도 안내해 주셔야죠!\n"
            f"진행자: 네, 이번 {month_str}에는 위생교육 퀴즈 만점자분들 중에서 순번 {rank1}번째, {rank2}번째, 그리고 {rank3}번째로 제출해 주신 행운의 주인공 세 분께 {prize_amount} {prize_item} 혜택을 드립니다!\n"
            f"파트너: 오늘 배운 핵심 수칙들 가볍게 복습하시면서 지금 바로 퀴즈 참여하시고, 기분 좋은 커피 기프티콘의 주인공이 되어보세요! 오늘도 안전한 하루 보내세요, 감사합니다!"
        )
    else:
        script_text = (
            f"자, 오늘 준비한 {ym_full} 정기 위생교육 내용은 여기까지입니다! 현장에서 늘 애써주시는 임직원 여러분 정말 고생 많으셨습니다. "
            f"끝으로 기다리시던 {month_str} 위생교육 이벤트 안내드릴게요! "
            f"이번 {month_str}에는 퀴즈 만점자분들 중 순번 {rank1}번째, {rank2}번째, 그리고 {rank3}번째로 제출해 주신 행운의 주인공 세 분께 "
            f"{prize_amount} {prize_item} 혜택을 드립니다. 지금 바로 퀴즈도 푸시고 시원한 커피 기프티콘의 주인공이 되어보세요! 감사합니다."
        )
    return {
        "layout": "OUTRO_EVENT",
        "type": "OUTRO",
        "title": f"{month_str} 위생교육 이벤트 안내",
        "subtitle": f"마지막으로 {month_str} 위생교육 이벤트 안내입니다!",
        "event_month": month_str,
        "event_ranks": ranks_str,
        "event_prize_highlight": prize_amount,
        "event_prize_rest": f" {prize_item}",
        "event_closing": "많은 관심과 적극적인 참여 부탁드립니다.",
        "content": (
            f"{month_str}은 만점자 중 순번 {ranks_str} 분에게\n"
            f"{prize_amount} {prize_item}\n"
            f"많은 관심과 적극적인 참여 부탁드립니다."
        ),
        "script": script_text
    }

def media_worker(idx, slide_data, total_slides, temp_dir, font_config, theme_mode, tts_config, leader_photos):
    img_path = os.path.join(temp_dir, f"slide_{idx:03d}.png")
    audio_path = os.path.join(temp_dir, f"audio_{idx:03d}.mp3")
    create_slide_image_hd(slide_data, idx, total_slides, img_path, font_config, theme_mode, leader_photos=leader_photos)
    create_natural_tts_audio(slide_data.get("script", ""), audio_path, tts_config)
    return idx, img_path, audio_path

# ==========================================
# [11] Gemini 3.8 Flash 기획 및 10~15분 심층/대화형 대본 확장 엔진
# ==========================================
def generate_ai_storyboard(topic, edu_year_num, edu_month_str, body_slide_count, intro_config, uploaded_gemini_files, extra_source_text, dialogue_mode=True):
    client = get_genai_client(GEMINI_API_KEY)
    generation_config = types.GenerateContentConfig(temperature=0.25, max_output_tokens=65536)
    edu_ym_full = f"{edu_year_num}년 {edu_month_str}"

    if intro_config["mode"] != "당부 말씀 슬라이드 생략":
        intro_instruction = f"""
        [슬라이드 1 (고정 오프닝: '월간 위생교육 전 당부말씀' - 상단 생산본부장 1단 와이드 + 하단 3개 부문장 3열 구조)]:
- 반드시 첫 번째 슬라이드는 "layout": "INTRO_1TOP_3BOT", "title": "월간 위생교육 전 당부말씀" 으로 설정하십시오.
- "cards" 배열에 정확히 아래 순서대로 4명의 리더 당부 말씀을 넣으십시오 (입력된 원문을 단 한 글자도 요약·생략하지 말고 그대로 유지):
          1) {{"header": "생산본부", "sub": "생산본부장", "body": "..."}}
          2) {{"header": "유제품생산부문", "sub": "유제품생산부문장", "body": "..."}}
          3) {{"header": "상온제품생산부문", "sub": "상온제품생산부문장", "body": "..."}}
          4) {{"header": "품질안전부문", "sub": "품질안전부문장", "body": "..."}}
- [입력된 당부 말씀 원문]:\n{intro_config.get("custom_text", "")}
        """
    else:
        intro_instruction = "[인트로 생략]: INTRO_1TOP_3BOT 없이 표지(COVER)부터 시작하십시오."

    if dialogue_mode:
        script_style_rule = f"""
        [성우 대본("script") 작성 절대 원칙 - NotebookLM 2인 티키타카 대화형 (남성 메인 진행자 고정)]:
- 화자 역할 고정: `진행자:`는 교육을 주관하는 **'식품안전팀 곽정혁' 본인(남성 메인 진행자)**이고, `파트너:`는 함께 호흡을 맞추며 질문·공감하는 보조 진행자입니다!
- **[첫 슬라이드 도입부 필수 고정 멘트]**: 가장 첫 번째 슬라이드(인덱스 0번)의 첫 발화는 반드시 `진행자: {MANDATORY_OPENING_LINE}` 로 시작해야 합니다! 이어지는 표지(COVER) 슬라이드에서는 이번 `{edu_ym_full}` 교육 핵심 주제('{topic}')를 명확히 소개하십시오.
- 반드시 `진행자:`와 `파트너:` 두 명의 화자가 실제 스튜디오에서 마주 앉아 자연스럽게 묻고 답하며 맞장구치는 대화 형식으로 작성하십시오!
- 각 발화 줄 맨 앞에 정확히 `진행자: ` 또는 `파트너: `를 붙이고 줄바꿈(`\\n`)으로 구분하십시오 (`진행자(곽정혁):`처럼 괄호를 넣지 마십시오).
- 기계적인 낭독투 대신 "어? 그런데 현장에서 이 부분은 왜 특히 조심해야 할까요?", "아, 그거 정말 중요한 포인트예요! 왜냐하면요," 같은 실제 사람의 호흡과 추임새를 살리되, 법적 수치와 팩트는 100% 정확히 전달하십시오.
        """
    else:
        script_style_rule = f"""
        [성우 대본("script") 작성 절대 원칙 - 1인 남성 전문 강사 구어체]:
- 화자는 교육을 주관하는 **'식품안전팀 곽정혁' 본인(남성 강사)**입니다.
- **[첫 슬라이드 도입부 필수 고정 멘트]**: 가장 첫 번째 슬라이드(인덱스 0번)의 첫 문장은 반드시 `"{MANDATORY_OPENING_LINE}"` 로 시작해야 합니다! 이어지는 표지(COVER) 슬라이드에서는 이번 `{edu_ym_full}` 교육 핵심 주제('{topic}')를 명확히 소개하십시오.
- 문어체("~함", "~바랍니다")를 피하고, 실제 인기 강사가 현장 작업자와 눈을 맞추며 대화하듯 생동감 있는 구어체("~거든요", "자, 여기서 꼭 기억할 포인트가 있습니다!")와 호흡용 쉼표(,)를 꼼꼼히 넣어 작성하십시오.
        """

    system_prompt = f"""
    당신은 연세유업 아산공장 식품안전팀(담당자: 곽정혁)의 수석 위생교육 콘텐츠 디렉터이자 HACCP 전문가입니다.
    참고 동영상이나 문서가 주어지면 그 안의 **핵심 교육 내용·법적 기준·수치·원칙만 정확히 추출**하여 재구성하십시오.
    원문에 없는 허위 수치나 규정을 임의로 추정·창작하지 마십시오.
    (※ 마지막 이벤트 안내 슬라이드는 파이썬 코드가 자동 결합하므로 생성하지 마십시오.)

    [교육 기준 연월]: {edu_ym_full} (※ 절대 주의: 참고 문서나 영상에 2023년, 2024년, 2025년 등 과거 연도가 적혀 있더라도 이번 교육의 기준 연도는 반드시 `{edu_year_num}년`입니다! 표지의 `subtitle`이나 대본에 교육 시기를 적을 때 반드시 `{edu_ym_full}`로 표기하십시오.)
    [교육 핵심 주제]: {topic}
    [생성 장수]: 표지(COVER, 1장) + 목차(TOC_STEP, 1장) + 본론 상세({body_slide_count}장)

    {intro_instruction}

    [시각화 및 레이아웃("layout") 배정 핵심 규칙]:
1. "COVER": 표지 슬라이드의 `subtitle`에는 `{edu_ym_full} 연세유업 아산공장 정기 위생교육` 형태로 정확한 연도({edu_year_num}년)를 기입하고, 우측에 들어갈 대표 3D 위생 도면 영문 프롬프트("visual_prompt")를 반드시 작성하십시오.
2. "TOC_STEP": 3~5단계 전체 교육 목차 카드.
3. 본론 슬라이드({body_slide_count}장) 중 **최소 70% 이상을 반드시 "SPLIT_VISUAL" 레이아웃으로 배정**하십시오!
  - 카드 본문("body")과 하단 요약("bottom_summary")은 도중에 끊기지 않도록 **끝맺음이 명확하고 완결된 문장(또는 완결된 개조식 구문)**으로 작성하며, 줄머리 기호는 폰트 깨짐이 없는 하이픈(`- `)만 사용하십시오 (`•`, `□` 사용 금지).
  - "visual_prompt": 해당 슬라이드의 우측 카드("cards")에 적힌 설비·행동·위생 수칙이 하나의 3D 아이소메트릭 도면 안에 직관적으로 묘사되도록 구체적인 영문 프롬프트를 작성하고, `"strictly no text or letters"`를 포함하십시오.
  - "CARDS_3COL": 핵심 수치 비교가 필요한 1~2장에 사용 ("metric" 필드에 짧은 강조 수치 기입).
  - "TABLE_MATRIX": 기준표나 구역 비교 등 표가 필요한 1~2장에 사용 ("table_headers", "table_rows" 필수 작성).
  - "SELF_AUDIT_2X2": 본론 가장 마지막 장은 현장 작업자용 '핵심 자가 진단 체크리스트' 4분할 카드로 구성하십시오 (각 줄 앞에 `[V] ` 표기).

    {script_style_rule}

    [출력 형식 규칙]:
    먼저 <pre_calc> 태그 안에 슬라이드별 카드 구성과 맞춤형 도면 구상을 요약하고, 이어서 <json_output> 태그 안에 순수 JSON 배열만 출력하십시오. (코드 축약 및 '...' 생략 절대 금지)

    <pre_calc>
    사전 설계 요약
    </pre_calc>
    <json_output>
    [
      {{
        "layout": "SPLIT_VISUAL",
        "type": "BODY",
        "title": "설비 배치 및 예방보전 유지관리",
        "subtitle": "",
        "content": "",
        "cards": [
          {{"header": "위생적 배치 및 오염 방지 설비", "sub": "", "metric": "벽·바닥 이격", "body": "- 세척·소독이 용이하도록 설비와 벽면, 바닥 사이에 충분한 작업 공간 및 이격 거리 확보\\n- 개방형 이송 라인 상부에는 이물 낙하 방지용 투명 아크릴 커버 설치"}},
          {{"header": "예방보전 및 유휴설비 관리", "sub": "", "metric": "보호커버 랩핑", "body": "- 정기 점검표 기반 마모 부품, 가스켓, 볼트·너트 조임 상태 주기적 점검\\n- 장기 미사용 유휴설비는 세척·소독 후 보호 랩핑 처리 및 식별표 부착"}}
        ],
        "table_headers": [],
        "table_rows": [],
        "bottom_summary": "설비는 세척하기 편하게 띄워 배치하고, 쉬고 있는 설비는 랩핑 후 이름표를 붙여줍니다.",
        "visual_prompt": "Isometric 3D technical illustration of a hygienic dairy factory floor: stainless steel machine positioned with clear gap from walls and floor, transparent acrylic shield over conveyor line, and a clean plastic-wrapped idle machine, centered full view, strictly no text or letters",
        "script": "진행자: 자, 이번에는 현장의 제조 설비를 어떻게 배치하고 관리해야 하는지 살펴볼까요?\\n파트너: 네! 현장에 가보면 설비가 벽에 딱 붙어 있지 않고 일정 간격 떨어져 있잖아요. 이게 다 청소와 소독을 구석구석 완벽하게 하기 위한 거군요!"
      }}
    ]
    </json_output>
    """

    payload = []
    if uploaded_gemini_files:
        payload.extend(uploaded_gemini_files)
    if extra_source_text.strip():
        payload.append(f"[참고 원문 데이터]:\n{extra_source_text}")
    payload.append(system_prompt)

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=PLANNING_MODEL,
                contents=payload,
                config=generation_config
            )
            raw_text = response.text
            pre_match = re.search(r'<pre_calc>(.*?)</pre_calc>', raw_text, re.DOTALL)
            pre_log = pre_match.group(1).strip() if pre_match else ""

            json_match = re.search(r'<json_output>(.*?)</json_output>', raw_text, re.DOTALL)
            json_str = json_match.group(1).strip() if json_match else raw_text
            json_str = re.sub(r'^```json\s*', '', json_str)
            json_str = re.sub(r'^```\s*', '', json_str)
            json_str = re.sub(r'\s*```$', '', json_str).strip()

            slides_data = json.loads(json_str)
            return slides_data, pre_log, None
        except Exception as e:
            if attempt < 2:
                time.sleep(4)
                continue
            return None, None, f"AI 기획안 생성 오류 ({PLANNING_MODEL}): {e}"

def generate_scripts_for_existing_pdf(pdf_pages_info, edu_year_num, edu_month_str, topic, dialogue_mode=True, has_intro_before=False):
    client = get_genai_client(GEMINI_API_KEY)
    edu_ym_full = f"{edu_year_num}년 {edu_month_str}"
    slides_text_dump = "\n\n".join([f"[슬라이드 {p['page_num']} 텍스트]:\n{p['text']}" for p in pdf_pages_info])
    first_page_rule = (
        ""
        if has_intro_before else
        f"단, 슬라이드 1번의 가장 첫 문장은 반드시 '{'진행자: ' if dialogue_mode else ''}{MANDATORY_OPENING_LINE}' 로 시작하십시오."
    )
    spk_rule = (
        f"반드시 남성 메인 진행자(`진행자:` = 식품안전팀 곽정혁)와 보조 진행자(`파트너:`) 두 명이 자연스럽게 질문하고 맞장구치며 설명하는 NotebookLM 2인 팟캐스트 대화 형식으로 작성하십시오. {first_page_rule}"
        if dialogue_mode else
        f"실제 남성 전문 강사(식품안전팀 곽정혁)가 현장 작업자들과 눈을 맞추며 친근하게 설명하는 1인 구어체 내레이션으로 작성하십시오. {first_page_rule}"
    )
    prompt = f"""
    다음은 연세유업 아산공장 '{edu_ym_full} 정기 위생교육 ({topic})' 슬라이드 총 {len(pdf_pages_info)}장의 텍스트입니다.
    슬라이드 1번부터 {len(pdf_pages_info)}번까지 단 한 장도 빠짐없이 대본(script)을 작성하십시오.
    (※ 교육 기준 연월은 `{edu_ym_full}`이므로 교육 시기를 말할 때 과거 연도로 잘못 말하지 않도록 주의하십시오.)
    {spk_rule}
    원문에 있는 수치는 100% 정확하게 전달하고 없는 사실을 추측하지 마십시오.

    반드시 아래 순수 JSON 배열 포맷으로만 출력하십시오 (개수: 정확히 {len(pdf_pages_info)}개):
    [
      {{"page_num": 1, "title": "슬라이드 1 요약 제목", "script": "1번 슬라이드 생동감 있는 대본"}},
      ...
    ]

    [추출된 슬라이드 텍스트 전체]:
    {slides_text_dump}
    """
    resp = client.models.generate_content(model=PLANNING_MODEL, contents=prompt)
    clean_j = re.sub(r'^```json\s*|\s*```$', '', resp.text.strip(), flags=re.MULTILINE).strip()
    return json.loads(clean_j)

def expand_scripts_for_target_duration(slides_data, topic, edu_year_num, edu_month_str, target_minutes, voice_speed=1.00, extra_source_text="", dialogue_mode=True):
    """
    말하기 속도를 늦추지 않고 목표 시간(10~15분)을 채우기 위해,
    4장 단위 병렬 배치로 슬라이드별 대본을 팩트 기반 심층 대화/강의로 확장합니다.
    """
    if not slides_data or target_minutes <= 4:
        return slides_data

    edu_ym_full = f"{edu_year_num}년 {edu_month_str}"
    chars_per_min = int(420 * float(voice_speed))
    total_target_chars = int(target_minutes * chars_per_min)
    current_total_chars = sum(len(re.sub(r'(진행자|파트너)\s*:\s*', '', str(s.get("script", ""))).strip()) for s in slides_data)

    if current_total_chars >= int(total_target_chars * 0.92):
        return slides_data

    weights = []
    for s in slides_data:
        lay = s.get("layout", "BODY")
        if lay in ("INTRO_1TOP_3BOT", "INTRO_4GRID"):
            weights.append(1.35)
        elif lay in ("COVER", "TOC_STEP", "OUTRO_EVENT"):
            weights.append(0.75)
        else:
            weights.append(1.10)
    weight_sum = sum(weights) if sum(weights) > 0 else len(slides_data)

    slide_targets = [max(250, int(total_target_chars * (w / weight_sum))) for w in weights]
    source_excerpt = extra_source_text[:6000] if extra_source_text else ""

    mode_guide = (
        f"반드시 `진행자: ...\\n파트너: ...` 형태의 2인 대담 형식을 유지하십시오! (`진행자:`는 남성 메인 강사인 식품안전팀 곽정혁 본인, `파트너:`는 보조 진행자입니다. 괄호 없이 `진행자:`와 `파트너:`만 표기) 특히 `slide_idx: 0`번(첫 번째 슬라이드)의 첫 문장은 반드시 `진행자: {MANDATORY_OPENING_LINE}` 로 시작해야 합니다! 한 슬라이드 안에서 진행자와 파트너가 4~6번 이상 자연스럽게 질문·공감·예시 설명을 주고받으며 진짜 사람끼리 대화하는 느낌을 극대화하십시오. (발화자 이름 글자 수는 제외한 순수 대사 분량이 목표 글자 수를 넘어야 합니다.)"
        if dialogue_mode else
        f"1인 남성 전문 위생교육 강사(식품안전팀 곽정혁)가 현장에서 친근하게 풀어서 이야기하는 구어체 형식을 유지하되, `slide_idx: 0`번(첫 번째 슬라이드)의 첫 문장은 반드시 `{MANDATORY_OPENING_LINE}` 로 시작하십시오."
    )

    def _expand_batch(batch_items):
        client = get_genai_client(GEMINI_API_KEY)
        batch_desc = []
        for s_idx, s_obj, min_ch in batch_items:
            cards_txt = json.dumps(s_obj.get("cards", []), ensure_ascii=False)
            tbl_txt = json.dumps(
                {"headers": s_obj.get("table_headers", []), "rows": s_obj.get("table_rows", [])},
                ensure_ascii=False
            )
            batch_desc.append(
                f"--- [슬라이드 인덱스: {s_idx}] ---\n"
                f"- 레이아웃: {s_obj.get('layout', '')}\n"
                f"- 제목: {s_obj.get('title', '')}\n"
                f"- 본문/요약: {s_obj.get('content', '')} / {s_obj.get('bottom_summary', '')}\n"
                f"- 카드 데이터: {cards_txt}\n"
                f"- 표 데이터: {tbl_txt}\n"
                f"- 기존 대본 초안: {s_obj.get('script', '')}\n"
                f"- [필수 목표 순수 대사 글자 수]: 공백 포함 최소 {min_ch}자 ~ {min_ch + 120}자 이상 작성"
            )

        joined_batch = "\n\n".join(batch_desc)
        prompt = f"""
        당신은 연세유업 아산공장 식품안전팀의 위생교육 방송 작가입니다.
        이번 '{edu_ym_full} 정기 위생교육 ({topic})' 영상이 법정 교육 시간 기준인 총 {target_minutes}분 분량을 정상 말속도로 자연스럽게 채울 수 있도록, 아래 슬라이드들의 대본("script")을 심층 확장하십시오.

        [절대 준수 규칙]:
1. 팩트 엄수 및 연도 정확성 (추정·창작 금지): 슬라이드 카드·표·기존 대본 및 참고 원문에 명시된 기준과 수치만 사용하십시오. 원문에 없는 과태료, 법조항 번호, 온도 수치를 절대 지어내지 마십시오. 이번 교육 기준 연월은 `{edu_ym_full}`이므로 2024년·2025년 등 과거 연도로 잘못 언급하지 마십시오.
2. 고정 슬라이드 보존:
  - `slide_idx: 0`이 포함되어 있다면 대본 맨 첫 문장을 반드시 '{'진행자: ' if dialogue_mode else ''}{MANDATORY_OPENING_LINE}' 로 시작하십시오.
  - 'INTRO_1TOP_3BOT'(당부말씀)는 생산본부장, 유제품생산부문장, 상온제품생산부문장, 품질안전부문장 4분의 핵심 당부 말씀을 모두 빠짐없이 소개하고 현장 실천 다짐을 나누십시오.
  - 'OUTRO_EVENT'(이벤트 안내)는 기존 대본의 당첨 순번 숫자와 경품 금액 문구를 100% 동일하게 유지하면서 이번 달 핵심 요약 인사를 덧붙이십시오.
3. 화법 및 구성: {mode_guide}
           같은 문장을 반복하지 말고 ① 이 기준이 왜 식품 안전에 중요한지 배경 설명 → ② 작업 전·중·후 현장에서 눈으로 확인하는 구체적 실천 요령 → ③ 놓치기 쉬운 실수 예방 팁 순서로 풀어내어 각 슬라이드의 [필수 목표 순수 대사 글자 수]를 반드시 달성하십시오.

        [참고 원문 발췌]:
        {source_excerpt}

        [확장 대상 슬라이드 목록]:
        {joined_batch}

        반드시 아래 순수 JSON 배열 형식으로만 출력하십시오 (축약 절대 금지):
        [
          {{"slide_idx": 0, "script": "목표 글자 수를 완벽히 충족하는 생동감 넘치는 대본 전체"}},
          ...
        ]
        """
        cfg = types.GenerateContentConfig(temperature=0.25, max_output_tokens=65536)
        for attempt in range(2):
            try:
                resp = client.models.generate_content(model=PLANNING_MODEL, contents=prompt, config=cfg)
                raw = re.sub(r'^```json\s*|\s*```$', '', resp.text.strip(), flags=re.MULTILINE).strip()
                return json.loads(raw)
            except Exception:
                time.sleep(2)
        return []

    batches = []
    batch_size = 4
    for i in range(0, len(slides_data), batch_size):
        chunk = [
            (idx, slides_data[idx], slide_targets[idx])
            for idx in range(i, min(i + batch_size, len(slides_data)))
        ]
        batches.append(chunk)

    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(_expand_batch, b) for b in batches]
        for fut in as_completed(futures):
            res_list = fut.result()
            if isinstance(res_list, list):
                for item in res_list:
                    s_i = item.get("slide_idx")
                    new_sc = str(item.get("script", "")).strip()
                    if isinstance(s_i, int) and 0 <= s_i < len(slides_data):
                        old_sc = str(slides_data[s_i].get("script", "")).strip()
                        if len(new_sc) > len(old_sc):
                            slides_data[s_i]["script"] = new_sc

    return slides_data

# ==========================================
# [12] 메인 애플리케이션 UI
# ==========================================
def main():
    st.markdown(custom_theme_css, unsafe_allow_html=True)

    for key in [
        "slides_data", "pre_calc_log", "final_video_bytes", "final_pptx_bytes",
        "custom_font_paths", "leader_photos", "intro_full_slide_bytes",
        "intro_extracted_raw_text", "last_intro_file_sig"
    ]:
        if key not in st.session_state:
            if key == "custom_font_paths":
                st.session_state[key] = {}
            elif key == "leader_photos":
                st.session_state[key] = load_default_leader_photos_from_repo()
            else:
                st.session_state[key] = None

    st.markdown("<h1>🎬 연세유업 AI 위생교육 스튜디오 <span style='font-size:0.45em; color:#64748b;'>(연도·월 일치 · 남성 진행 고정 · 잘림 0% Auto-Fit)</span></h1>", unsafe_allow_html=True)
    st.markdown(f"유료 Gemini API 단일 연결로 **`{MANDATORY_OPENING_LINE}` 남성 오프닝 고정**, **문장 잘림 없는 Auto-Fit 슬라이드**, **10~15분 법정 교육 시간 맞춤 대본**을 원클릭으로 제작합니다.")
    st.markdown("<hr>", unsafe_allow_html=True)

    # --- 좌측 사이드바 ---
    with st.sidebar:
        st.header("🧠 유료 AI 엔진 직결 상태")
        key_type_label = "Gemini Paid API (학습 차단 적용)"
        st.info(f"• **연결 방식:** `{key_type_label}`\n• **기획·심층대본:** `{PLANNING_MODEL}`\n• **3D 맞춤 도면:** `{VISUAL_OMNI_MODEL}`")

        st.markdown("---")
        st.header("🎙️ 남성 진행 성우 & 대화 모드 설정")
        tts_engine = st.radio(
            "음성 합성 엔진 선택",
            [
                "Gemini Native TTS (NotebookLM 동일 계열 · 남성 메인 진행)",
                "Edge Neural TTS (보조 남성 신경망 성우)",
                "OpenAI TTS (ChatGPT 남성 성우)"
            ],
            index=0
        )

        gemini_voice = "Puck"
        gemini_partner_voice = "Kore"
        gemini_dialogue = True
        openai_model = "gpt-4o-mini-tts"
        openai_voice = "echo"
        edge_voice = "ko-KR-InJoonNeural"
        edge_pitch = "+0Hz"
        openai_key_input = DEFAULT_OPENAI_KEY

        if "Gemini" in tts_engine:
            gemini_voice_map = {
                "🎙️ NotebookLM 2인 대담 (곽정혁 남성 진행 Puck + 여성 파트너 Kore - 추천)": ("Puck", "Kore", True, "ko-KR-InJoonNeural"),
                "🎙️ NotebookLM 2인 중저음 대담 (곽정혁 중저음 남성 Charon + 여성 파트너 Aoede)": ("Charon", "Aoede", True, "ko-KR-HyunsuMultilingualNeural"),
                "🎙️ NotebookLM 2인 남성 듀오 대담 (남성 진행 Puck + 남성 파트너 Charon)": ("Puck", "Charon", True, "ko-KR-InJoonNeural"),
                "Puck 단독 (생동감 넘치고 활기찬 남성 강사 - 단독 추천)": ("Puck", "Kore", False, "ko-KR-InJoonNeural"),
                "Charon 단독 (신뢰감 있고 차분한 중저음 남성 강사)": ("Charon", "Kore", False, "ko-KR-HyunsuMultilingualNeural"),
                "Fenrir 단독 (힘 있고 명확한 현장 브리핑 남성 강사)": ("Fenrir", "Kore", False, "ko-KR-InJoonNeural"),
                "Orus 단독 (단정하고 안정감 있는 남성 아나운서톤)": ("Orus", "Kore", False, "ko-KR-InJoonNeural")
            }
            chosen_g_label = st.selectbox("NotebookLM / Gemini 남성 중심 보이스 선택", list(gemini_voice_map.keys()), index=0)
            gemini_voice, gemini_partner_voice, gemini_dialogue, edge_voice = gemini_voice_map[chosen_g_label]
        elif "OpenAI" in tts_engine:
            openai_key_input = st.text_input("OpenAI API Key", value=DEFAULT_OPENAI_KEY, type="password").strip().strip('"').strip("'")
            gemini_dialogue = False
            openai_model = st.selectbox("OpenAI 음성 모델", ["gpt-4o-mini-tts", "tts-1-hd"], index=0)
            openai_voice_map = {
                "echo (친근하고 부드러운 남성 강사 - 기본)": ("echo", "ko-KR-InJoonNeural", "+0Hz"),
                "onyx (무게감 있는 중저음 남성 강사)": ("onyx", "ko-KR-HyunsuMultilingualNeural", "-3Hz"),
                "ash (생동감 넘치는 경쾌한 남성 강사)": ("ash", "ko-KR-InJoonNeural", "+5Hz"),
                "verse (차분하고 또렷한 남성 내레이션)": ("verse", "ko-KR-InJoonNeural", "+0Hz")
            }
            chosen_v_label = st.selectbox("ChatGPT 남성 보이스 선택", list(openai_voice_map.keys()), index=0)
            openai_voice, edge_voice, edge_pitch = openai_voice_map[chosen_v_label]
        else:
            gemini_dialogue = False
            edge_voice_map = {
                "인준 (자연스럽고 또렷한 남성 아나운서 - 기본)": ("ko-KR-InJoonNeural", "+0Hz"),
                "현수 (중후하고 차분한 남성 강사)": ("ko-KR-HyunsuMultilingualNeural", "+0Hz"),
                "인준 브라이트 (활기차고 경쾌한 남성 진행자)": ("ko-KR-InJoonNeural", "+5Hz"),
                "현수 딥톤 (신뢰감 있는 중저음 남성 브리핑)": ("ko-KR-HyunsuMultilingualNeural", "-4Hz"),
                "인준 차분톤 (안정감 있는 남성 현장 강사)": ("ko-KR-InJoonNeural", "-4Hz")
            }
            chosen_e_label = st.selectbox("남성 신경망 성우 선택", list(edge_voice_map.keys()), index=0)
            edge_voice, edge_pitch = edge_voice_map[chosen_e_label]

        voice_speed = st.slider("말하기 속도 배율 (자연스러운 정속 1.00 권장)", min_value=0.92, max_value=1.16, value=1.00, step=0.02)

        tts_config = {
            "engine": tts_engine,
            "gemini_key": GEMINI_API_KEY,
            "gemini_voice": gemini_voice,
            "gemini_partner_voice": gemini_partner_voice,
            "gemini_dialogue": gemini_dialogue,
            "openai_key": openai_key_input,
            "openai_model": openai_model,
            "openai_voice": openai_voice,
            "edge_voice": edge_voice,
            "edge_pitch": edge_pitch,
            "speed": voice_speed
        }

        if st.button("🔊 선택한 남성 진행 / 2인 대화 오프닝 미리듣기", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_aud:
                sample_path = tmp_aud.name
            sample_txt = (
                f"진행자: {MANDATORY_OPENING_LINE} 오늘 정기 위생교육도 현장에서 꼭 필요한 핵심만 귀에 쏙쏙 들어오게 함께 살펴보겠습니다.\n"
                "파트너: 네, 안녕하세요! 작업 현장에서 바로 실천할 수 있는 위생 수칙들을 알기 쉽게 짚어드리겠습니다!"
                if gemini_dialogue else
                f"{MANDATORY_OPENING_LINE} 연세유업 아산공장 임직원 여러분, 이번 달 정기 위생교육을 활기차게 시작해 보겠습니다."
            )
            with st.spinner("실제 성우 오디오 샘플을 합성하고 있습니다..."):
                _, tts_meta = create_natural_tts_audio(sample_txt, sample_path, tts_config, return_meta=True)
            with open(sample_path, "rb") as af:
                st.audio(af.read(), format="audio/mp3")
            st.caption(f"🎙️ 출력 확인: `{tts_meta.get('used_engine', '')}`")
            if tts_meta.get("warning"):
                st.warning(tts_meta["warning"])
            if os.path.exists(sample_path):
                try:
                    os.remove(sample_path)
                except Exception:
                    pass

        st.markdown("---")
        st.header("🎨 슬라이드 테마 설정")
        theme_mode = st.radio(
            "슬라이드 디자인 테마",
            ["10월 스타일 (라이트 블루프린트 모눈 테마)", "9월 스타일 (다크 사이버네틱 네온 테마)"],
            index=0
        )

        st.markdown("---")
        st.header("🔤 깃허브 font/ 폴더 서체 연동")
        FONT_OPTIONS = ["연세제목체", "연세본문체", "맑은고딕"]
        selected_title_font = st.selectbox("제목 글씨체", FONT_OPTIONS, index=0)
        selected_body_font = st.selectbox("본문 글씨체", FONT_OPTIONS, index=1)

        t_found = find_font_file(selected_title_font, is_bold=True, custom_font_paths=st.session_state["custom_font_paths"])
        b_found = find_font_file(selected_body_font, is_bold=False, custom_font_paths=st.session_state["custom_font_paths"])
        st.caption(f"✅ 제목 로드됨: `{t_found if t_found else '기본 폰트'}`")
        st.caption(f"✅ 본문 로드됨: `{b_found if b_found else '기본 폰트'}`")

        with st.expander("📁 임시 폰트 파일 수동 업로드 (비상용)"):
            for f_label, k_id in [("연세제목체", "u_yt"), ("연세본문체", "u_yb"), ("맑은고딕", "u_mg")]:
                f_obj = st.file_uploader(f_label, type=["ttf", "otf"], key=k_id)
                if f_obj:
                    ext = os.path.splitext(f_obj.name)[1]
                    p = os.path.join(tempfile.gettempdir(), f"font_{f_label}{ext}")
                    with open(p, "wb") as wf:
                        wf.write(f_obj.getbuffer())
                    st.session_state["custom_font_paths"][f_label] = p

        font_config = {
            "title_font": selected_title_font,
            "body_font": selected_body_font,
            "custom_paths": st.session_state["custom_font_paths"]
        }

    tab1, tab2, tab3 = st.tabs([
        "1. 교육 연도·월·목표 시간(10~15분) & 당부말씀 설정",
        "2. 고해상도 슬라이드 미리보기 & 대화 대본·도면 편집",
        "3. 고해상도 이미지 PPTX & NotebookLM 성우 MP4 렌더링"
    ])

    # ==========================================
    # 탭 1: 입력 및 기획 생성 (교육 기준 연도 입력 추가)
    # ==========================================
    with tab1:
        today_dt = datetime.date.today()
        col_m0, col_m1, col_m2, col_m3 = st.columns([0.9, 0.8, 2.0, 1.3])
        with col_m0:
            edu_year_num = st.number_input("교육 기준 연도:", min_value=2024, max_value=2035, value=int(today_dt.year), step=1)
        with col_m1:
            edu_month_num = st.number_input("교육 기준 월:", min_value=1, max_value=12, value=int(today_dt.month), step=1)
            edu_month_str = f"{edu_month_num}월"
            edu_ym_full_str = f"{edu_year_num}년 {edu_month_str}"
        with col_m2:
            topic = st.text_input("이번 달 위생교육 핵심 주제:", value="작업장 이물혼입 예방 및 영업장 위생관리")
        with col_m3:
            target_video_mins = st.slider(
                "⏱️ 목표 교육 영상 시간 (분):",
                min_value=4,
                max_value=18,
                value=12,
                step=1,
                help="말속도를 늦추지 않고 대본 분량을 심층 확장하여 연간 2시간(월 10~15분) 기준을 충족합니다."
            )

        st.markdown("### 🎥 Step 1. 참고할 원본 교육 영상(MP4) 또는 교육 자료 업로드")
        workflow_mode = st.radio(
            "본론 슬라이드 생성 방식:",
            [
                "표준 모드: 참고 교육 영상(MP4)·문서(PPTX/PDF) 분석 ➡️ 3D 맞춤 도면 & 10~15분 대화형/강의형 대본 자동 제작",
                "PDF 직행 모드: 이미 완성된 슬라이드 PDF 업로드 ➡️ 원본 디자인 유지 + 당부말씀·이벤트 결합 및 목표 시간 맞춤 더빙"
            ],
            index=0
        )

        uploaded_sources = []
        extra_source_text = ""
        existing_pdf_file = None
        body_slide_count = 12

        if "표준 모드" in workflow_mode:
            col_s1, col_s2 = st.columns([1.2, 1])
            with col_s1:
                uploaded_sources = st.file_uploader(
                    "참고할 원본 교육 영상(MP4) 또는 문서(PDF, PPTX) 업로드 (복수 선택 가능):",
                    type=["mp4", "pdf", "pptx"],
                    accept_multiple_files=True
                )
                body_slide_count = st.slider(
                    "본론 슬라이드 생성 장수 (10~15분 영상은 11~15장 권장 / 당부말씀·표지·목차·이벤트 제외):",
                    min_value=4,
                    max_value=18,
                    value=12
                )
            with col_s2:
                extra_source_text = st.text_area(
                    "추가 반영할 사내 수칙 / 한계기준 수치 (선택):",
                    height=135,
                    placeholder="예: 검사구역 조도 540 Lux 이상, 금속 이물 2mm 이상 검출 불가, 자석봉 오전/오후 1회 이상 세척 등"
                )
        else:
            existing_pdf_file = st.file_uploader("완성된 교육 자료 PDF 파일 업로드:", type=["pdf"])

        st.markdown("---")

        # --- Step 2: 당부 말씀 ---
        st.markdown("### 👔 Step 2. 오프닝: 월간 위생교육 전 당부말씀 (Auto-Fit 글씨 자동 조절 & PDF 원본 그대로 사용 지원)")
        intro_mode = st.radio(
            "당부 말씀 적용 방식:",
            [
                "PPTX 업로드 또는 텍스트 입력 (부문장님 얼굴 사진 깃허브 고정 + 글자 잘림 없는 Auto-Fit 표준 레이아웃)",
                "당부 말씀 슬라이드 완성본 이미지/PDF 통째로 올리기 (원본 슬라이드 그림 100% 그대로 표시 + PDF 텍스트 자동 대본 연동)",
                "당부 말씀 슬라이드 생략"
            ]
        )

        intro_custom_text = ""
        ldr_prod = ldr_dairy = ldr_ambient = ldr_qc = ""

        if "PPTX 업로드" in intro_mode:
            st.session_state["intro_full_slide_bytes"] = None
            if not st.session_state.get("leader_photos"):
                st.session_state["leader_photos"] = load_default_leader_photos_from_repo()

            leader_pptx = st.file_uploader("📄 '위생교육 전 말씀.pptx' 업로드 (4분 말씀 텍스트만 자동 추출 / 길어도 글씨 크기 자동 맞춤):", type=["pptx"], key="ldr_pptx")
            parsed_leaders = {}
            if leader_pptx:
                parsed_leaders = parse_leader_pptx_text_only(leader_pptx)
                st.success("PPTX에서 4분의 당부 말씀 텍스트를 추출했습니다! (말씀이 길어도 카드 크기에 맞춰 글씨가 자동 축소됩니다)")

            default_prod = parsed_leaders.get("생산본부") or "일교차가 커지면서 외부 해충이나 이물질이 따뜻한 실내 작업장으로 유입되기 쉬워 각별한 주의가 필요한 시기입니다. 우리 연세유업 아산공장의 흔들림 없는 식품 안전은 결국 현장에 계신 여러분의 빈틈없는 이물 관리와 밀폐 수칙 준수에서 시작됩니다. 현장에 정착된 철저한 HACCP 실행력을 바탕으로 위해요소를 사전에 100% 차단하는 든든한 방어선 역할을 해 주시길 당부 드립니다."
            default_dairy = parsed_leaders.get("유제품생산부문") or "유제품은 미생물과 주변 환경변화에 매우 민감한 특성을 가지고 있습니다. 공장 천정, 덕트, 배관 표면 등에 결로가 발생하면 곰팡이 증식 조건이 되므로 작업장 환경 청소와 철저한 건조 관리에 많은 관심이 필요합니다. 일일 건강 모니터링도 철저히 실천해 주시기 바랍니다."
            default_ambient = parsed_leaders.get("상온제품생산부문") or "상온제품의 안전성을 책임지는 핵심은 멸균 온도 관리와 여과망 점검 등 철저한 CCP 모니터링에 있습니다. 각 공정의 한계기준을 명확히 숙지하고 이탈 시 즉각적인 조치가 이루어질 수 있도록 만전을 기해 주십시오. 완벽한 CCP 점검과 작업장 밀폐를 당부드립니다."
            default_qc = parsed_leaders.get("품질안전부문") or "식품 제조에 있어 '이물 혼입'은 우리가 쌓아온 고객의 신뢰를 한순간에 무너뜨리는 가장 치명적인 사고입니다. 품질과 안전에는 어떠한 타협도 있을 수 없음을 명심하고, 철저한 밀폐와 꼼꼼한 개인위생 등 기본에 충실한 원칙 중심의 현장관리를 실천해 주실 것을 당부드립니다."

            ldr_prod = st.text_area("🔴 [상단 와이드] 생산본부 (생산본부장)", value=default_prod, height=85)
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                ldr_dairy = st.text_area("🔵 [하단 좌측] 유제품생산부문 (유제품생산부문장)", value=default_dairy, height=135)
            with col_b2:
                ldr_ambient = st.text_area("🔵 [하단 중앙] 상온제품생산부문 (상온제품생산부문장)", value=default_ambient, height=135)
            with col_b3:
                ldr_qc = st.text_area("🔵 [하단 우측] 품질안전부문 (품질안전부문장)", value=default_qc, height=135)

            with st.expander("📸 부문장님 프로필 사진 개별 확인 / 수동 교체 (선택)"):
                p_cols = st.columns(4)
                for idx_p, (dept_k, label_k) in enumerate([
                    ("생산본부", "생산본부장 사진"),
                    ("유제품생산부문", "유제품생산부문장 사진"),
                    ("상온제품생산부문", "상온제품생산부문장 사진"),
                    ("품질안전부문", "품질안전부문장 사진")
                ]):
                    with p_cols[idx_p]:
                        up_p = st.file_uploader(label_k, type=["png", "jpg", "jpeg"], key=f"lp_{dept_k}")
                        if up_p:
                            st.session_state["leader_photos"][dept_k] = up_p.getvalue()
                        if st.session_state["leader_photos"].get(dept_k):
                            st.image(st.session_state["leader_photos"][dept_k], width=90, caption=f"{label_k} 적용됨")

            intro_custom_text = (
                f"[생산본부장]: {ldr_prod}\n"
                f"[유제품생산부문장]: {ldr_dairy}\n"
                f"[상온제품생산부문장]: {ldr_ambient}\n"
                f"[품질안전부문장]: {ldr_qc}"
            )

        elif "통째로 올리기" in intro_mode:
            col_fs1, col_fs2 = st.columns([1, 1])
            with col_fs1:
                intro_slide_file = st.file_uploader("당부 말씀 슬라이드 이미지(PNG/JPG) 또는 1페이지 PDF 업로드:", type=["png", "jpg", "jpeg", "pdf"], key="intro_raw_slide")
                if intro_slide_file:
                    ext_i = os.path.splitext(intro_slide_file.name)[1].lower()
                    file_sig = f"{intro_slide_file.name}_{intro_slide_file.size}"
                    if ext_i == ".pdf" and PYMUPDF_AVAILABLE:
                        p_list = extract_pages_from_pdf(intro_slide_file.getvalue())
                        if p_list:
                            st.session_state["intro_full_slide_bytes"] = p_list[0]["image_bytes"]
                            if st.session_state.get("last_intro_file_sig") != file_sig:
                                st.session_state["intro_extracted_raw_text"] = p_list[0]["text"]
                                st.session_state["last_intro_file_sig"] = file_sig
                            st.success("PDF 슬라이드 화면을 100% 원본 그대로 고정했으며, 내부 말씀 텍스트를 우측 대본창에 자동 추출했습니다!")
                    else:
                        st.session_state["intro_full_slide_bytes"] = intro_slide_file.getvalue()
                        st.session_state["last_intro_file_sig"] = file_sig
                        st.success("업로드하신 당부 말씀 이미지를 100% 원본 그대로 사용합니다.")
            with col_fs2:
                default_raw_intro = st.session_state.get("intro_extracted_raw_text") or "생산본부장님 및 유제품, 상온, 품질안전 각 부문장님께서 철저한 이물 관리와 기본 위생 수칙 준수를 당부하셨습니다."
                intro_custom_text = st.text_area(
                    "성우가 읽을 당부 말씀 원문 (PDF 업로드 시 자동 채움):",
                    value=default_raw_intro,
                    height=145
                )

        st.markdown("---")

        # --- Step 3: 고정 양식 이벤트 안내 ---
        st.markdown(f"### 🎁 Step 3. 클로징: {edu_ym_full_str} 위생교육 이벤트 안내 (월별 순번 자동 계산 고정 양식)")
        include_outro = st.checkbox("마지막 슬라이드에 고정 양식 이벤트 페이지 자동 추가", value=True)
        if include_outro:
            default_r1 = int(edu_month_num)
            default_r2 = int(edu_month_num) + 30
            default_r3 = int(edu_month_num) + 60
            col_ev1, col_ev2, col_ev3, col_ev4 = st.columns([1, 1, 1, 1.5])
            with col_ev1:
                ev_rank1 = st.number_input("당첨 순번 1", min_value=1, max_value=500, value=default_r1, step=1)
            with col_ev2:
                ev_rank2 = st.number_input("당첨 순번 2", min_value=1, max_value=500, value=default_r2, step=1)
            with col_ev3:
                ev_rank3 = st.number_input("당첨 순번 3", min_value=1, max_value=500, value=default_r3, step=1)
            with col_ev4:
                ev_prize_amount = st.text_input("경품 강조 금액 (붉은색)", value="10,000원")

        st.markdown("<br>", unsafe_allow_html=True)

        mode_btn_label = "NotebookLM 2인 대화형(남성진행)" if gemini_dialogue else "남성 전문강사 구어체"
        if st.button(f"🚀 {edu_ym_full_str} 기준 · 목표 {target_video_mins}분 [{mode_btn_label}] 대본 & Auto-Fit 3D 슬라이드 생성 시작", type="primary", use_container_width=True):
            if not GEMINI_API_KEY:
                st.error("Streamlit Secrets에 GOOGLE_API_KEY 또는 GEMINI_API_KEY를 설정해주세요.")
            elif "표준 모드" in workflow_mode:
                with st.spinner(f"1/2단계: [{PLANNING_MODEL}] {edu_ym_full_str} 슬라이드 기획 및 [{mode_btn_label}] 초안 설계 중..."):
                    gemini_files = []
                    combined_text = extra_source_text

                    for f in uploaded_sources:
                        ext = os.path.splitext(f.name)[1].lower()
                        if ext == ".pptx":
                            p_txt, _ = extract_text_and_Media_from_pptx(f)
                            combined_text += f"\n\n[참고 PPTX ({f.name}) 내용]:\n{p_txt}"
                        elif ext == ".pdf" and PYMUPDF_AVAILABLE:
                            pdf_pages = extract_pages_from_pdf(f.getvalue())
                            pdf_txt = "\n\n".join([f"[페이지 {p['page_num']}]\n{p['text']}" for p in pdf_pages if p.get("text")])
                            combined_text += f"\n\n[참고 PDF ({f.name}) 내용]:\n{pdf_txt}"
                        elif ext == ".mp4":
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                                tmp.write(f.getbuffer())
                                tmp_path = tmp.name
                            try:
                                gemini_files.append(build_inline_video_part(tmp_path))
                            finally:
                                if os.path.exists(tmp_path):
                                    os.remove(tmp_path)
                        else:
                            gemini_files.append(types.Part.from_bytes(data=f.getvalue(), mime_type="application/pdf"))

                    intro_cfg = {"mode": intro_mode, "custom_text": intro_custom_text}
                    slides, pre_log, err = generate_ai_storyboard(
                        topic, edu_year_num, edu_month_str, body_slide_count, intro_cfg, gemini_files, combined_text, dialogue_mode=gemini_dialogue
                    )

                if err:
                    st.error(err)
                else:
                    slides = [s for s in slides if s.get("layout") != "OUTRO_EVENT"]

                    if slides and intro_mode != "당부 말씀 슬라이드 생략" and slides[0].get("layout") in ("INTRO_1TOP_3BOT", "INTRO_4GRID"):
                        slides[0]["layout"] = "INTRO_1TOP_3BOT"
                        if st.session_state.get("intro_full_slide_bytes") and "통째로" in intro_mode:
                            slides[0]["custom_image_bytes"] = st.session_state["intro_full_slide_bytes"]
                        elif "PPTX 업로드" in intro_mode:
                            slides[0]["cards"] = [
                                {"header": "생산본부", "sub": "생산본부장", "body": ldr_prod},
                                {"header": "유제품생산부문", "sub": "유제품생산부문장", "body": ldr_dairy},
                                {"header": "상온제품생산부문", "sub": "상온제품생산부문장", "body": ldr_ambient},
                                {"header": "품질안전부문", "sub": "품질안전부문장", "body": ldr_qc}
                            ]

                    if include_outro:
                        slides.append(
                            build_fixed_event_slide(
                                edu_month_num,
                                ev_rank1,
                                ev_rank2,
                                ev_rank3,
                                ev_prize_amount,
                                dialogue_mode=gemini_dialogue,
                                year_int=edu_year_num
                            )
                        )

                    with st.spinner(f"2/2단계: 약 {target_video_mins}분 분량 [{mode_btn_label}] 대본 심층 확장 및 [{VISUAL_OMNI_MODEL}] 3D 도면 생성 중..."):
                        slides = expand_scripts_for_target_duration(
                            slides,
                            topic=topic,
                            edu_year_num=edu_year_num,
                            edu_month_str=edu_month_str,
                            target_minutes=target_video_mins,
                            voice_speed=voice_speed,
                            extra_source_text=f"{intro_custom_text}\n\n{combined_text}",
                            dialogue_mode=gemini_dialogue
                        )
                        slides = enforce_opening_greeting(slides, dialogue_mode=gemini_dialogue)
                        slides = enforce_edu_year_consistency(slides, edu_year_num, edu_month_num)

                        local_custom_fonts = dict(st.session_state.get("custom_font_paths", {}))

                        def _fill_visual(idx_s, s_obj):
                            if s_obj.get("layout") in ("SPLIT_VISUAL", "COVER") and not s_obj.get("visual_bytes"):
                                s_obj["visual_bytes"] = generate_ai_illustration(
                                    s_obj,
                                    topic,
                                    theme_mode,
                                    openai_key=openai_key_input,
                                    gemini_key=GEMINI_API_KEY,
                                    custom_font_paths=local_custom_fonts
                                )
                            return idx_s, s_obj

                        with ThreadPoolExecutor(max_workers=3) as vis_exec:
                            vis_futures = [vis_exec.submit(_fill_visual, i, s) for i, s in enumerate(slides)]
                            for vf in as_completed(vis_futures):
                                r_i, r_s = vf.result()
                                slides[r_i] = r_s

                    pure_chars = sum(len(re.sub(r'(진행자|파트너)\s*:\s*', '', str(s.get("script", ""))).strip()) for s in slides)
                    est_min_done = (pure_chars / (420.0 * voice_speed)) + (len(slides) * 0.6 / 60.0)
                    st.session_state["slides_data"] = slides
                    st.session_state["pre_calc_log"] = pre_log
                    st.session_state["final_video_bytes"] = None
                    st.session_state["final_pptx_bytes"] = None
                    st.success(f"총 {len(slides)}장 생성 완료! ({edu_ym_full_str} 기준 · 순수 대사 {pure_chars:,}자 · 예상 시간 약 {est_min_done:.1f}분) [2번 탭]에서 확인하세요.")
            else:
                if not existing_pdf_file:
                    st.warning("PDF 파일을 먼저 업로드해주세요.")
                elif not PYMUPDF_AVAILABLE:
                    st.error("requirements.txt에 PyMuPDF를 추가해주세요.")
                else:
                    with st.spinner(f"PDF 고해상도 추출 및 {edu_ym_full_str} 기준 목표 {target_video_mins}분 [{mode_btn_label}] 대본 생성 중..."):
                        has_intro_flag = (intro_mode != "당부 말씀 슬라이드 생략")
                        pages_info = extract_pages_from_pdf(existing_pdf_file.getvalue())
                        scripts_list = generate_scripts_for_existing_pdf(
                            pages_info,
                            edu_year_num,
                            edu_month_str,
                            topic,
                            dialogue_mode=gemini_dialogue,
                            has_intro_before=has_intro_flag
                        )
                        built_slides = []

                        if has_intro_flag:
                            if "통째로" in intro_mode:
                                intro_sc = (
                                    f"진행자: {MANDATORY_OPENING_LINE} {edu_ym_full_str} 정기 위생교육 시작에 앞서 생산본부장님과 각 부문장님들의 당부 말씀을 함께 살펴보겠습니다. {intro_custom_text}\n"
                                    f"파트너: 네, 현장에서 꼭 마음에 새기고 실천해야 할 핵심 당부 말씀이네요! 그럼 본격적인 교육을 시작해 볼까요?"
                                    if gemini_dialogue else
                                    f"{MANDATORY_OPENING_LINE} {edu_ym_full_str} 정기 위생교육을 시작하기에 앞서, 생산본부장님과 각 부문장님들의 당부 말씀을 먼저 전해드립니다. {intro_custom_text}"
                                )
                            else:
                                intro_sc = (
                                    f"진행자: {MANDATORY_OPENING_LINE} {edu_ym_full_str} 정기 위생교육 시작에 앞서 생산본부장님과 각 부문장님들의 당부 말씀을 함께 짚어보겠습니다. 먼저 생산본부장님께서는 {ldr_prod}\n"
                                    f"파트너: 네, 이어서 유제품생산부문장님께서는 {ldr_dairy} 그리고 상온제품생산부문장님께서는 {ldr_ambient}\n"
                                    f"진행자: 끝으로 품질안전부문장님께서는 {ldr_qc} 라고 강조해 주셨습니다. 현장에서 꼭 함께 실천해 주시길 바랍니다!"
                                    if gemini_dialogue else
                                    f"{MANDATORY_OPENING_LINE} {edu_ym_full_str} 정기 위생교육을 시작하기에 앞서, 생산본부장님과 각 부문장님들의 당부 말씀을 먼저 전해드립니다. {ldr_prod} 이어서 유제품생산부문장님께서는 {ldr_dairy} 상온제품생산부문장님께서는 {ldr_ambient} 끝으로 품질안전부문장님께서는 {ldr_qc}"
                                )

                            intro_slide_obj = {
                                "layout": "INTRO_1TOP_3BOT",
                                "type": "INTRO",
                                "title": "월간 위생교육 전 당부말씀",
                                "cards": [
                                    {"header": "생산본부", "sub": "생산본부장", "body": ldr_prod},
                                    {"header": "유제품생산부문", "sub": "유제품생산부문장", "body": ldr_dairy},
                                    {"header": "상온제품생산부문", "sub": "상온제품생산부문장", "body": ldr_ambient},
                                    {"header": "품질안전부문", "sub": "품질안전부문장", "body": ldr_qc}
                                ],
                                "script": intro_sc
                            }
                            if st.session_state.get("intro_full_slide_bytes") and "통째로" in intro_mode:
                                intro_slide_obj["custom_image_bytes"] = st.session_state["intro_full_slide_bytes"]
                            built_slides.append(intro_slide_obj)

                        pdf_full_txt = "\n\n".join([f"[슬라이드 {p['page_num']}]\n{p['text']}" for p in pages_info])
                        for idx, p_info in enumerate(pages_info):
                            matched_script = scripts_list[idx].get("script", "") if idx < len(scripts_list) else p_info["text"]
                            matched_title = scripts_list[idx].get("title", f"슬라이드 {p_info['page_num']}") if idx < len(scripts_list) else f"슬라이드 {p_info['page_num']}"
                            built_slides.append({
                                "layout": "CUSTOM_PDF_PAGE",
                                "type": "BODY",
                                "title": matched_title,
                                "content": p_info["text"],
                                "custom_image_bytes": p_info["image_bytes"],
                                "script": matched_script
                            })

                        if include_outro:
                            built_slides.append(
                                build_fixed_event_slide(
                                    edu_month_num,
                                    ev_rank1,
                                    ev_rank2,
                                    ev_rank3,
                                    ev_prize_amount,
                                    dialogue_mode=gemini_dialogue,
                                    year_int=edu_year_num
                                )
                            )

                        built_slides = expand_scripts_for_target_duration(
                            built_slides,
                            topic=topic,
                            edu_year_num=edu_year_num,
                            edu_month_str=edu_month_str,
                            target_minutes=target_video_mins,
                            voice_speed=voice_speed,
                            extra_source_text=f"{intro_custom_text}\n\n{pdf_full_txt}",
                            dialogue_mode=gemini_dialogue
                        )
                        built_slides = enforce_opening_greeting(built_slides, dialogue_mode=gemini_dialogue)
                        built_slides = enforce_edu_year_consistency(built_slides, edu_year_num, edu_month_num)

                        pure_chars = sum(len(re.sub(r'(진행자|파트너)\s*:\s*', '', str(s.get("script", ""))).strip()) for s in built_slides)
                        est_min_done = (pure_chars / (420.0 * voice_speed)) + (len(built_slides) * 0.6 / 60.0)
                        st.session_state["slides_data"] = built_slides
                        st.session_state["final_video_bytes"] = None
                        st.session_state["final_pptx_bytes"] = None
                        st.success(f"총 {len(built_slides)}장 준비 완료! ({edu_ym_full_str} 기준 · 순수 대사 {pure_chars:,}자 · 예상 시간 약 {est_min_done:.1f}분) [2번 탭]으로 이동하세요.")

    # ==========================================
    # 탭 2: Full-HD 슬라이드 미리보기 & 대화 대본/도면 편집
    # ==========================================
    with tab2:
        if not st.session_state.get("slides_data"):
            st.info("먼저 [1번 탭]에서 슬라이드 기획을 실행해 주십시오.")
        else:
            total_s = len(st.session_state["slides_data"])
            pure_total_chars = sum(len(re.sub(r'(진행자|파트너)\s*:\s*', '', str(s.get("script", ""))).strip()) for s in st.session_state["slides_data"])
            est_total_sec = int((pure_total_chars / (7.0 * voice_speed)) + (total_s * 0.6))
            est_mm, est_ss = divmod(est_total_sec, 60)

            st.markdown("### 🖥️ Full-HD(1920×1080) 슬라이드 미리보기 및 대화형 대본·3D 도면 편집")
            st.info(
                f"⏱️ **현재 대본 분량 진단 ({edu_ym_full_str} 기준):** 총 **{total_s}장** | 순수 대사 **{pure_total_chars:,}자** "
                f"➡️ 현재 속도(`{voice_speed:.2f}x`) 기준 **예상 영상 길이: 약 {est_mm}분 {est_ss:02d}초** "
                f"({'NotebookLM 2인 대화 모드(남성진행)' if gemini_dialogue else '1인 남성 강사 모드'})"
            )

            updated_slides = []

            with tempfile.TemporaryDirectory() as preview_dir:
                for idx, slide in enumerate(st.session_state["slides_data"]):
                    slide["edu_ym_label"] = f"{edu_year_num}년 {edu_month_num:02d}월"
                    cur_pure_len = len(re.sub(r'(진행자|파트너)\s*:\s*', '', str(slide.get("script", ""))).strip())
                    cur_sec = int(cur_pure_len / (7.0 * voice_speed))
                    is_expanded = (idx in (0, 1, total_s - 1))
                    with st.expander(f"슬라이드 {idx + 1} [{slide.get('layout', 'SLIDE')}] : {slide.get('title', '')} ({cur_pure_len}자 · 약 {cur_sec}초)", expanded=is_expanded):
                        col_p1, col_p2 = st.columns([1.2, 1])
                        with col_p2:
                            new_title = st.text_input("슬라이드 제목", value=slide.get("title", ""), key=f"t_{idx}")
                            slide["title"] = new_title

                            if slide.get("layout") == "COVER":
                                new_sub = st.text_input("표지 부제목 (교육 연도·월 표기)", value=slide.get("subtitle", ""), key=f"sub_{idx}")
                                slide["subtitle"] = new_sub

                            new_script = st.text_area(
                                "성우 대본 (`진행자:`=남성 곽정혁 / `파트너:`=보조 진행자)",
                                value=slide.get("script", ""),
                                height=195,
                                key=f"s_{idx}"
                            )
                            slide["script"] = new_script

                            if slide.get("layout") not in ("INTRO_1TOP_3BOT", "INTRO_4GRID", "OUTRO_EVENT", "CUSTOM_PDF_PAGE", "COVER"):
                                new_bsum = st.text_input(
                                    "하단 핵심 원칙 배너 문구 (길어도 글씨 크기 자동 조절됨)",
                                    value=slide.get("bottom_summary", ""),
                                    key=f"bs_{idx}"
                                )
                                slide["bottom_summary"] = new_bsum

                            if slide.get("layout") in ("SPLIT_VISUAL", "COVER", "CARDS_3COL"):
                                new_v_prompt = st.text_input(
                                    "AI 3D 도면 생성 지시문 (우측 카드 내용과 자동 결합됨)",
                                    value=slide.get("visual_prompt", ""),
                                    key=f"vp_{idx}"
                                )
                                slide["visual_prompt"] = new_v_prompt

                                if st.button(f"🎨 이 슬라이드 3D 도면({VISUAL_OMNI_MODEL}) 다시 그리기", key=f"regen_v_{idx}", use_container_width=True):
                                    with st.spinner(f"슬라이드 {idx + 1}의 3D 맞춤 도면을 새로 생성하고 있습니다..."):
                                        slide["layout"] = "SPLIT_VISUAL" if slide.get("layout") != "COVER" else "COVER"
                                        slide["visual_bytes"] = generate_ai_illustration(
                                            slide,
                                            topic,
                                            theme_mode,
                                            openai_key=openai_key_input,
                                            gemini_key=GEMINI_API_KEY,
                                            custom_font_paths=st.session_state["custom_font_paths"]
                                        )

                            c_u1, c_u2 = st.columns(2)
                            with c_u1:
                                part_vis = st.file_uploader("좌측 그림만 현장사진으로 교체", type=["png", "jpg", "jpeg"], key=f"pv_{idx}")
                                if part_vis:
                                    slide["visual_bytes"] = part_vis.getvalue()
                                    if slide.get("layout") != "COVER":
                                        slide["layout"] = "SPLIT_VISUAL"
                            with c_u2:
                                full_rep = st.file_uploader("슬라이드 전체 이미지 통째 교체", type=["png", "jpg", "jpeg"], key=f"fr_{idx}")
                                if full_rep:
                                    slide["custom_image_bytes"] = full_rep.getvalue()

                        with col_p1:
                            preview_img_path = os.path.join(preview_dir, f"preview_{idx}.png")
                            create_slide_image_hd(slide, idx, total_s, preview_img_path, font_config, theme_mode, leader_photos=st.session_state["leader_photos"])
                            st.image(preview_img_path, caption=f"슬라이드 {idx + 1} (1920x1080 고해상도 Auto-Fit 렌더링)", use_container_width=True)

                        updated_slides.append(slide)

            st.session_state["slides_data"] = updated_slides

    # ==========================================
    # 탭 3: 고해상도 이미지 PPTX 및 NotebookLM 성우 MP4 렌더링
    # ==========================================
    with tab3:
        if not st.session_state.get("slides_data"):
            st.info("먼저 [1번 탭]에서 슬라이드 기획을 실행해 주십시오.")
        else:
            slides_to_render = st.session_state["slides_data"]
            st.markdown(f"### 🎬 최종 결과물 렌더링 및 다운로드 ({edu_ym_full_str} · 총 {len(slides_to_render)}장 · 1920×1080 Full-HD)")

            col_b1, col_b2 = st.columns(2)
            with col_b1:
                if st.button("📊 1단계: 고해상도 이미지 슬라이드 PPTX 생성 (깨짐·잘림 0%)", use_container_width=True):
                    with st.spinner("1920x1080 고해상도 슬라이드 이미지 기반 PPTX 파일을 생성하고 있습니다..."):
                        st.session_state["final_pptx_bytes"] = generate_highres_image_pptx(
                            slides_to_render,
                            font_config,
                            theme_mode,
                            leader_photos=st.session_state["leader_photos"]
                        )
                        st.success("고해상도 이미지 PPTX 생성 완료! (발표자 노트에 대본이 수록되어 있습니다)")

                if st.session_state.get("final_pptx_bytes"):
                    st.download_button(
                        label="📥 파워포인트(.pptx) 파일 다운로드",
                        data=st.session_state["final_pptx_bytes"],
                        file_name=f"연세유업_{edu_year_num}년_{edu_month_str}_위생교육_{datetime.date.today().strftime('%Y%m%d')}.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        use_container_width=True
                    )

            with col_b2:
                if st.button("🎥 2단계: 남성 진행 성우 더빙 + Full-HD MP4 영상 렌더링", type="primary", use_container_width=True):
                    t_start = time.time()
                    with tempfile.TemporaryDirectory() as temp_dir:
                        total_cnt = len(slides_to_render)
                        p_bar = st.progress(0, text="병렬 엔진 가동: Full-HD 슬라이드 및 Gemini Native TTS 오디오 동시 합성 중...")

                        media_results = [None] * total_cnt
                        local_leader_photos = dict(st.session_state.get("leader_photos", {}))
                        with ThreadPoolExecutor(max_workers=3) as executor:
                            futures = [
                                executor.submit(
                                    media_worker,
                                    idx,
                                    s_data,
                                    total_cnt,
                                    temp_dir,
                                    font_config,
                                    theme_mode,
                                    tts_config,
                                    local_leader_photos
                                )
                                for idx, s_data in enumerate(slides_to_render)
                            ]
                            done_cnt = 0
                            for future in as_completed(futures):
                                idx, img_p, aud_p = future.result()
                                media_results[idx] = (img_p, aud_p)
                                done_cnt += 1
                                p_bar.progress(done_cnt / (total_cnt + 1), text=f"({done_cnt}/{total_cnt}) 고해상도 화면 및 성우 오디오 합성 완료...")

                        p_bar.progress(total_cnt / (total_cnt + 1), text="최종 Full-HD MP4 교육 영상 병합 인코딩 중...")
                        v_clips, a_clips = [], []
                        try:
                            for img_path, audio_path in media_results:
                                a_clip = AudioFileClip(audio_path)
                                i_clip = ImageClip(img_path).set_duration(a_clip.duration + 0.6)
                                v_clips.append(i_clip.set_audio(a_clip))
                                a_clips.append(a_clip)

                            final_video = concatenate_videoclips(v_clips, method="compose")
                            out_mp4 = os.path.join(temp_dir, "yonsei_edu_hd.mp4")
                            final_video.write_videofile(
                                out_mp4,
                                fps=5,
                                codec="libx264",
                                audio_codec="aac",
                                preset="ultrafast",
                                threads=4,
                                ffmpeg_params=["-pix_fmt", "yuv420p"],
                                logger=None
                            )
                            with open(out_mp4, "rb") as f:
                                st.session_state["final_video_bytes"] = f.read()
                        finally:
                            for c in v_clips:
                                try:
                                    c.close()
                                except Exception:
                                    pass
                            for a in a_clips:
                                try:
                                    a.close()
                                except Exception:
                                    pass

                        p_bar.empty()
                        st.success(f"🎉 Full-HD 교육 영상 렌더링 완료! (총 소요시간: {time.time() - t_start:.1f}초)")

            if st.session_state.get("final_video_bytes"):
                st.markdown("---")
                st.video(st.session_state["final_video_bytes"])
                st.download_button(
                    label="📥 완성된 MP4 교육 영상 다운로드",
                    data=st.session_state["final_video_bytes"],
                    file_name=f"연세유업_{edu_year_num}년_{edu_month_str}_위생교육영상_{datetime.date.today().strftime('%Y%m%d')}.mp4",
                    mime="video/mp4",
                    type="primary",
                    use_container_width=True
                )

if __name__ == "__main__":
    if check_password():
        main()
