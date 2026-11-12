import streamlit as st
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import asyncio
import base64
import socket
import requests
import time
import json
import math
import os
import io
import re
import datetime
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont, ImageOps
from moviepy.editor import ImageClip, AudioFileClip, VideoFileClip, concatenate_videoclips
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
# [3] API 키 로드
# ==========================================
GEMINI_API_KEY = (
    st.secrets.get("GEMINI_API_KEY")
    or st.secrets.get("GOOGLE_API_KEY")
    or os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GOOGLE_API_KEY")
)
DEFAULT_OPENAI_KEY = (
    st.secrets.get("OPENAI_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or ""
)

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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
# [5] 깃허브 font/ 폴더 자동 스캔 및 한글 폰트 로더
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
    """
    1순위: 사이드바 임시 업로드 폰트
    2순위: 깃허브 저장소 내 font/ , fonts/ , 폰트/ , 루트(.) 폴더 자동 스캔
    3순위: 로컬 Windows 폰트 폴더 또는 리눅스 서버 나눔고딕(fonts-nanum) 안전 폴백
    """
    if custom_font_paths and custom_font_paths.get(font_choice):
        cp = custom_font_paths[font_choice]
        if os.path.exists(cp):
            return cp

    info = FONT_SEARCH_MAP.get(font_choice, FONT_SEARCH_MAP["맑은고딕"])

    # 맑은고딕 굵기 우선 분기
    if font_choice == "맑은고딕":
        target_files = ["malgunbd.ttf", "MalgunGothicBold.ttf"] if is_bold else ["malgun.ttf", "맑은고딕.ttf", "MalgunGothic.ttf"]
        for d in FONT_SEARCH_DIRS:
            for tf in target_files:
                cand = os.path.join(d, tf) if d != "." else tf
                if os.path.exists(cand):
                    return cand

    # 깃허브 font/ 폴더 내 정확한 파일명 매칭
    for d in FONT_SEARCH_DIRS:
        for exact_name in info["exact_names"]:
            cand = os.path.join(d, exact_name) if d != "." else exact_name
            if os.path.exists(cand):
                return cand

    # 깃허브 font/ 폴더 내 부분 키워드 스캔 (파일명이 조금 달라도 자동 인식)
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

    # 시스템 기본 한글 폰트 폴백 (네모 깨짐 원천 차단)
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
    lines = []
    for paragraph in str(text).split('\n'):
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
                        lines.append(current_line)
                        current_line = ch
        if current_line:
            lines.append(current_line)
    return lines

# ==========================================
# [6] 문서/이미지/리더 프로필 사진 추출 유틸리티
# ==========================================
def load_default_leader_photos_from_repo():
    """
    깃허브 저장소 내 모든 폴더(루트 및 하위 폴더 전체)를 스캔하여
    4분 리더의 프로필 사진을 고정 로드합니다.
    """
    photos = {}
    role_map = {
        "생산본부": ["생산본부장", "생산본부", "본부장", "공장장", "leader1", "1_"],
        "유제품생산부문": ["유제품생산부문장", "유제품부문장", "유제품", "leader2", "2_"],
        "상온제품생산부문": ["상온제품생산부문장", "상온부문장", "상온", "leader3", "3_"],
        "품질안전부문": ["품질안전부문장", "품질부문장", "품질안전", "품질", "오재필", "어성현", "leader4", "4_"]
    }

    # 깃허브 저장소 내 모든 폴더(.git 제외) 자동 탐색
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
    """참고용 본론 PPTX에서 슬라이드 텍스트와 삽입된 이미지들을 추출합니다."""
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
    """
    '위생교육 전 말씀.pptx' 파일에서 이미지는 절대 건드리지 않고,
    오직 4분의 당부 말씀 텍스트만 추출합니다. (깃허브 고정 사진 보호)
    """
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

def capture_frame_from_video(video_path, sec):
    try:
        clip = VideoFileClip(video_path)
        safe_sec = min(max(0.0, float(sec)), max(0.0, clip.duration - 0.3))
        frame = clip.get_frame(safe_sec)
        clip.close()
        pil_img = Image.fromarray(frame)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None

# ==========================================
# [7] 3단계 시각 자료(일러스트/도면) 생성 엔진 (학습자 몰입도 극대화)
# ==========================================
def generate_procedural_infographic(slide_title, visual_prompt, theme_mode, custom_font_paths=None):
    """
    외부 이미지 생성 API가 실패하거나 한도 초과일 때에도 절대 빈 화면이 나오지 않도록,
    슬라이드 주제를 분석해 고해상도(840x760) 맞춤형 벡터 인포그래픽 도면을 직접 그립니다.
    """
    w, h = 840, 760
    is_light = "10월" in theme_mode
    bg_col = (238, 244, 252) if is_light else (13, 25, 43)
    accent = (2, 132, 199) if is_light else (16, 185, 129)
    sub_accent = (30, 64, 175) if is_light else (56, 189, 248)
    ink = (30, 41, 59) if is_light else (226, 232, 240)

    img = Image.new("RGB", (w, h), bg_col)
    draw = ImageDraw.Draw(img)

    # 모눈 그리드 배경
    for gx in range(0, w, 35):
        draw.line([(gx, 0), (gx, h)], fill=(216, 228, 242) if is_light else (22, 40, 68), width=1)
    for gy in range(0, h, 35):
        draw.line([(0, gy), (w, gy)], fill=(216, 228, 242) if is_light else (22, 40, 68), width=1)

    f_label = load_pil_font("연세제목체", 28, is_bold=True, custom_font_paths=custom_font_paths)
    f_sub = load_pil_font("연세본문체", 22, is_bold=False, custom_font_paths=custom_font_paths)

    combined_kw = f"{slide_title} {visual_prompt}".lower()
    cx, cy = w // 2, h // 2 - 20

    # 외곽 원형 기술 링
    draw.ellipse([(cx - 240, cy - 240), (cx + 240, cy + 240)], outline=accent, width=4)
    draw.ellipse([(cx - 210, cy - 210), (cx + 210, cy + 210)], fill=(255, 255, 255) if is_light else (19, 36, 62), outline=sub_accent, width=3)

    if any(k in combined_kw for k in ["온도", "냉장", "냉동", "살균", "멸균", "센서", "열", "temp", "therm"]):
        # 온도계 & 센서 파형 다이어그램
        draw.rounded_rectangle([(cx - 32, cy - 130), (cx + 32, cy + 70)], radius=30, fill=bg_col, outline=sub_accent, width=6)
        draw.ellipse([(cx - 58, cy + 35), (cx + 58, cy + 150)], fill=(239, 68, 68), outline=sub_accent, width=6)
        draw.rounded_rectangle([(cx - 14, cy - 65), (cx + 14, cy + 60)], radius=10, fill=(239, 68, 68))
        for sy in range(cy - 105, cy + 25, 28):
            draw.line([(cx + 42, sy), (cx + 72, sy)], fill=sub_accent, width=4)
        tag = "REAL-TIME TEMP & CCP MONITORING"
    elif any(k in combined_kw for k in ["구역", "동선", "청결", "밀폐", "출입", "문", "벽", "배수", "zone", "flow", "door"]):
        # 청결구역 vs 일반구역 차단 아키텍처 도면
        draw.rounded_rectangle([(cx - 155, cy - 110), (cx - 15, cy + 110)], radius=12, fill=(219, 234, 254) if is_light else (12, 74, 110), outline=accent, width=4)
        draw.rounded_rectangle([(cx + 15, cy - 110), (cx + 155, cy + 110)], radius=12, fill=(241, 245, 249) if is_light else (30, 41, 59), outline=(239, 68, 68), width=4)
        draw.line([(cx, cy - 145), (cx, cy + 145)], fill=(239, 68, 68), width=7)
        draw.text((cx - 132, cy - 20), "청결구역", font=f_label, fill=ink)
        draw.text((cx + 38, cy - 20), "일반구역", font=f_label, fill=ink)
        tag = "CLEAN ZONE & ACCESS BARRIER"
    elif any(k in combined_kw for k in ["이물", "금속", "해충", "벌레", "자석", "여과", "필터", "유리", "metal", "pest", "foreign"]):
        # 다중 여과 & 금속 검출 실드 도면
        pts = [(cx, cy - 145), (cx + 135, cy - 85), (cx + 110, cy + 75), (cx, cy + 150), (cx - 110, cy + 75), (cx - 135, cy - 85)]
        draw.polygon(pts, fill=(236, 253, 245) if is_light else (6, 78, 59), outline=accent)
        draw.line([(cx - 55, cy + 5), (cx - 12, cy + 48), (cx + 68, cy - 42)], fill=accent, width=14)
        tag = "FOREIGN MATTER DEFENSE SHIELD"
    else:
        # 표준 HACCP 위생 관리 체크 인포그래픽
        draw.rounded_rectangle([(cx - 105, cy - 130), (cx + 105, cy + 130)], radius=18, fill=bg_col, outline=sub_accent, width=5)
        draw.rounded_rectangle([(cx - 50, cy - 152), (cx + 50, cy - 110)], radius=10, fill=accent)
        for idx_r, ry in enumerate([cy - 55, cy + 5, cy + 65]):
            draw.ellipse([(cx - 72, ry - 14), (cx - 44, ry + 14)], fill=accent)
            draw.line([(cx - 25, ry), (cx + 72, ry)], fill=ink, width=6)
        tag = "HACCP HYGIENE STANDARD PROTOCOL"

    # 하단 캡션 배너
    draw.rounded_rectangle([(60, h - 110), (w - 60, h - 38)], radius=12, fill=sub_accent)
    bb = draw.textbbox((0, 0), tag, font=f_sub)
    draw.text(((w - (bb[2] - bb[0])) // 2, h - 88), tag, font=f_sub, fill=(255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def generate_ai_illustration(visual_prompt, slide_title, theme_style, openai_key="", gemini_key="", custom_font_paths=None):
    """
    1차: OpenAI DALL-E 3 (키 보유 및 선택 시) 또는 Gemini Imagen 3 / Multimodal Flash 이미지 생성 호출
    2차: 실패 시 고해상도 자체 벡터 인포그래픽으로 100% 폴백하여 절대 빈칸이 발생하지 않음
    """
    style_prefix = (
        "Clean isometric 3D food factory hygiene technical illustration on a soft light architectural grid background, modern infographic vector art, strictly NO text or letters inside the image: "
        if "10월" in theme_style else
        "Modern dark navy cybernetic isometric 3D food safety engineering illustration with glowing emerald and cyan accents, clean vector art, strictly NO text or letters inside the image: "
    )
    full_prompt = style_prefix + (visual_prompt or slide_title)

    # 1) Gemini Imagen 3 REST API 시도
    if gemini_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict?key={gemini_key}"
            payload = {
                "instances": [{"prompt": full_prompt}],
                "parameters": {"sampleCount": 1, "aspectRatio": "1:1"}
            }
            r = requests.post(url, json=payload, timeout=25)
            if r.status_code == 200:
                preds = r.json().get("predictions", [])
                if preds and preds[0].get("bytesBase64Encoded"):
                    return base64.b64decode(preds[0]["bytesBase64Encoded"])
        except Exception:
            pass

        # Gemini 멀티모달 이미지 출력 모델 시도
        for m_name in ["gemini-2.0-flash-exp-image-generation", "gemini-omni-1.2-flash"]:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={gemini_key}"
                payload = {
                    "contents": [{"parts": [{"text": full_prompt}]}],
                    "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}
                }
                r = requests.post(url, json=payload, timeout=25)
                if r.status_code == 200:
                    parts = r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
                    for p in parts:
                        if "inlineData" in p and p["inlineData"].get("data"):
                            return base64.b64decode(p["inlineData"]["data"])
            except Exception:
                continue

    # 2) OpenAI DALL-E 3 폴백 시도 (OpenAI 키가 입력된 경우)
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

    # 3) 100% 보장 프로시저럴 벡터 인포그래픽 생성
    return generate_procedural_infographic(slide_title, visual_prompt, theme_style, custom_font_paths=custom_font_paths)

# ==========================================
# [8] 초자연스러운 성우 음성 엔진 (OpenAI ChatGPT TTS + Edge Neural TTS)
# ==========================================
def create_natural_tts_audio(script_text, output_path, tts_config):
    """
    1순위: OpenAI TTS (gpt-4o-mini-tts 또는 tts-1-hd) - ChatGPT 음성 대화와 동일한 구어체 목소리
    2순위: Microsoft Edge Neural TTS (ko-KR-InJoonNeural 남성 / ko-KR-SunHiNeural 여성) - 무료 고품질 성우
    3순위: gTTS 비상 폴백
    """
    clean_script = re.sub(r'[\*\#\[\]]', ' ', str(script_text)).strip()
    if not clean_script:
        clean_script = "다음 화면의 핵심 위생 수칙을 함께 확인해 보겠습니다."

    engine = tts_config.get("engine", "OpenAI TTS")
    openai_key = tts_config.get("openai_key", "").strip()
    openai_voice = tts_config.get("openai_voice", "echo")
    openai_model = tts_config.get("openai_model", "gpt-4o-mini-tts")
    edge_voice = tts_config.get("edge_voice", "ko-KR-InJoonNeural")
    speed_rate = float(tts_config.get("speed", 1.03))

    # 1) OpenAI TTS 호출
    if "OpenAI" in engine and openai_key:
        headers = {
            "Authorization": f"Bearer {openai_key}",
            "Content-Type": "application/json"
        }
        models_to_try = [openai_model, "tts-1-hd"] if openai_model == "gpt-4o-mini-tts" else [openai_model]
        for m_try in models_to_try:
            try:
                safe_voice = openai_voice
                if m_try != "gpt-4o-mini-tts" and safe_voice in ("ash", "coral", "sage", "ballad", "verse"):
                    safe_voice = "echo"
                payload = {
                    "model": m_try,
                    "input": clean_script,
                    "voice": safe_voice,
                    "response_format": "mp3",
                    "speed": speed_rate
                }
                if m_try == "gpt-4o-mini-tts":
                    payload["instructions"] = (
                        "실제 사내 위생교육 전문 강사나 인기 팟캐스트 진행자처럼 아주 자연스럽고 친근하며 생동감 있는 한국어 구어체로 말하세요. "
                        "기계적인 낭독을 피하고 문맥에 맞춰 호흡과 억양을 활기차고 명확하게 살려주세요."
                    )
                resp = requests.post("https://api.openai.com/v1/audio/speech", headers=headers, json=payload, timeout=60)
                if resp.status_code == 200 and len(resp.content) > 500:
                    with open(output_path, "wb") as f:
                        f.write(resp.content)
                    return output_path
            except Exception:
                continue

    # 2) Edge Neural TTS 호출 (무료 남성/여성 아나운서)
    if EDGE_TTS_AVAILABLE:
        try:
            pct = int((speed_rate - 1.0) * 100)
            rate_str = f"+{pct}%" if pct >= 0 else f"{pct}%"

            async def _save_edge():
                communicate = edge_tts.Communicate(clean_script, voice=edge_voice, rate=rate_str)
                await communicate.save(output_path)

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_save_edge())
            finally:
                loop.close()

            if os.path.exists(output_path) and os.path.getsize(output_path) > 500:
                return output_path
        except Exception:
            pass

    # 3) 최후 비상 폴백
    tts = gTTS(text=clean_script, lang="ko", slow=False)
    tts.save(output_path)
    return output_path

# ==========================================
# [9] Full HD (1920x1080) 고해상도 슬라이드 렌더링 엔진
# ==========================================
def paste_circular_avatar(base_img, photo_bytes, center_x, center_y, radius, border_color, border_width=6):
    """부문장님 얼굴 사진을 원형으로 깔끔하게 크롭하여 배지 안에 합성합니다."""
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
    """
    모든 슬라이드를 Full HD (1920x1080) 고해상도 PNG로 렌더링합니다.
    이 이미지가 미리보기, PPTX 슬라이드, MP4 영상에 동일하게 사용되어 깨짐이나 어긋남이 0%입니다.
    """
    width, height = 1920, 1080

    if slide_data.get("custom_image_bytes"):
        img = Image.open(io.BytesIO(slide_data["custom_image_bytes"])).convert("RGB")
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        img.save(output_path, quality=95)
        return output_path

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    title_font_name = font_config.get("title_font", "연세제목체")
    body_font_name = font_config.get("body_font", "연세본문체")
    custom_paths = font_config.get("custom_paths", {})

    f_hero = load_pil_font(title_font_name, 70, is_bold=True, custom_font_paths=custom_paths)
    f_quote = load_pil_font(title_font_name, 64, is_bold=True, custom_font_paths=custom_paths)
    f_title = load_pil_font(title_font_name, 48, is_bold=True, custom_font_paths=custom_paths)
    f_sub = load_pil_font(title_font_name, 33, is_bold=True, custom_font_paths=custom_paths)
    f_card_title = load_pil_font(title_font_name, 30, is_bold=True, custom_font_paths=custom_paths)
    f_badge_dept = load_pil_font(title_font_name, 21, is_bold=True, custom_font_paths=custom_paths)
    f_badge_role = load_pil_font(title_font_name, 27, is_bold=True, custom_font_paths=custom_paths)
    f_body = load_pil_font(body_font_name, 26, is_bold=False, custom_font_paths=custom_paths)
    f_small = load_pil_font(body_font_name, 22, is_bold=False, custom_font_paths=custom_paths)
    f_metric = load_pil_font(title_font_name, 64, is_bold=True, custom_font_paths=custom_paths)

    layout = slide_data.get("layout", "SPLIT_VISUAL")
    is_light = "10월" in theme_mode
    leader_photos = leader_photos or {}

    # ---------------------------------------------------------
    # [1] INTRO_1TOP_3BOT: 상단 생산본부장 와이드 + 하단 3개 부문장 3열 (얼굴 사진 원형 합성 지원)
    # ---------------------------------------------------------
    if layout in ("INTRO_1TOP_3BOT", "INTRO_4GRID"):
        draw.rectangle([(0, 0), (width, height)], fill=(244, 246, 250))
        title_txt = slide_data.get("title", "월간 위생교육 전 당부말씀")
        draw.text((62, 34), title_txt, font=f_title, fill=(15, 23, 42))

        cards = slide_data.get("cards", [])
        top_card = cards[0] if len(cards) > 0 else {"header": "생산본부", "sub": "생산본부장", "body": ""}

        # 상단 와이드 카드 (생산본부장)
        tx1, ty1, tx2, ty2 = 60, 118, 1860, 396
        red_col = (220, 38, 38)
        draw.rounded_rectangle([(tx1 + 4, ty1 + 5), (tx2 + 4, ty2 + 5)], radius=20, fill=(215, 222, 232))
        draw.rounded_rectangle([(tx1, ty1), (tx2, ty2)], radius=20, fill=(255, 255, 255), outline=(218, 226, 236), width=3)
        draw.rounded_rectangle([(tx1, ty1), (tx1 + 15, ty2)], radius=7, fill=red_col)

        cx_top, cy_top, r_top = tx1 + 115, (ty1 + ty2) // 2, 78
        has_top_photo = paste_circular_avatar(img, leader_photos.get("생산본부"), cx_top, cy_top, r_top, red_col, border_width=6)

        dept_top = top_card.get("header", "생산본부")
        role_top = top_card.get("sub", "생산본부장")
        if has_top_photo:
            # 사진이 있을 때는 사진 우측에 직책 표기 (사진 가림 0%)
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

        draw.line([(divider_x, ty1 + 38), (divider_x, ty2 - 38)], fill=(226, 232, 240), width=3)
        draw.text((divider_x + 22, ty1 + 20), "“", font=f_quote, fill=(203, 213, 225))

        top_body_lines = wrap_text_by_pixel(top_card.get("body", ""), f_small, (tx2 - divider_x) - 105, draw)
        t_y = ty1 + 42
        for line in top_body_lines[:5]:
            draw.text((divider_x + 76, t_y), line, font=f_small, fill=(30, 41, 59))
            t_y += 38

        # 하단 3열 세로 카드
        bottom_defaults = [
            ("유제품생산부문", "유제품생산부문장"),
            ("상온제품생산부문", "상온제품생산부문장"),
            ("품질안전부문", "품질안전부문장")
        ]
        navy_col = (15, 62, 124)
        card_w = 576
        gap_x = 36
        by1, by2 = 426, 1036

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

            cx_b, cy_b, r_b = bx1 + 84, by1 + 78, 52
            has_b_photo = paste_circular_avatar(img, leader_photos.get(def_dept), cx_b, cy_b, r_b, navy_col, border_width=5)
            if not has_b_photo:
                draw.ellipse([(cx_b - r_b, cy_b - r_b), (cx_b + r_b, cy_b + r_b)], fill=(255, 255, 255), outline=navy_col, width=5)
                short_tag = "유제품" if "유제품" in dept_str else ("상온" if "상온" in dept_str else "품질")
                bb_tag = draw.textbbox((0, 0), short_tag, font=f_badge_dept)
                draw.text((cx_b - (bb_tag[2] - bb_tag[0]) // 2, cy_b - 12), short_tag, font=f_badge_dept, fill=navy_col)

            draw.text((bx1 + 154, by1 + 42), dept_str, font=f_badge_dept, fill=(100, 116, 139))
            draw.text((bx1 + 154, by1 + 72), role_str, font=f_card_title, fill=(15, 23, 42))
            draw.text((bx2 - 68, by1 + 22), "“", font=f_quote, fill=(203, 213, 225))
            draw.line([(bx1 + 32, by1 + 146), (bx2 - 32, by1 + 146)], fill=(235, 240, 246), width=2)

            b_lines = wrap_text_by_pixel(body_str, f_small, card_w - 68, draw)
            cy_text = by1 + 168
            for line in b_lines[:11]:
                if cy_text > by2 - 34:
                    break
                draw.text((bx1 + 36, cy_text), line, font=f_small, fill=(45, 55, 72))
                cy_text += 37

        img.save(output_path, quality=95)
        return output_path

    # ---------------------------------------------------------
    # [2] OUTRO_EVENT: '이벤트 페이지.pptx' 고정 양식 100% 재현
    # ---------------------------------------------------------
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
                draw.text((tx_start, ty_text), ev_closing, font=f_ev_row, fill=(30, 41, 59))

        img.save(output_path, quality=95)
        return output_path

    # 공통 배경 그리드
    draw_background_grid_hd(draw, width, height, theme_mode)
    text_main_col = (15, 23, 42) if is_light else (248, 250, 252)
    text_sub_col = (51, 65, 85) if is_light else (203, 213, 225)
    card_bg_col = (255, 255, 255) if is_light else (15, 28, 48)
    card_border_col = (30, 64, 175) if is_light else (16, 185, 129)

    # ---------------------------------------------------------
    # [3] COVER: 고해상도 표지 (우측에 시각 도면 카드 함께 배치하여 심심함 해소)
    # ---------------------------------------------------------
    if layout == "COVER":
        box_x1, box_y1, box_x2, box_y2 = 110, 150, 1810, 930
        draw.rounded_rectangle([(box_x1 + 8, box_y1 + 8), (box_x2 + 8, box_y2 + 8)], radius=26, fill=(203, 213, 225) if is_light else (5, 10, 20))
        draw.rounded_rectangle([(box_x1, box_y1), (box_x2, box_y2)], radius=26, fill=card_bg_col, outline=card_border_col, width=5)
        draw.rectangle([(box_x1, box_y1), (box_x1 + 32, box_y2)], fill=card_border_col)

        has_cover_vis = bool(slide_data.get("visual_bytes"))
        text_max_w = 960 if has_cover_vis else 1500

        badge_txt = "YONSEI DAIRY | FOOD SAFETY & HACCP EDUCATION"
        draw.text((box_x1 + 85, box_y1 + 80), badge_txt, font=f_card_title, fill=(2, 132, 199) if is_light else (52, 211, 153))

        main_title = slide_data.get("title", "정기 위생교육")
        t_lines = wrap_text_by_pixel(main_title, f_hero, text_max_w, draw)
        ty = box_y1 + 165
        for tl in t_lines[:3]:
            draw.text((box_x1 + 85, ty), tl, font=f_hero, fill=text_main_col)
            ty += 96

        sub_txt = slide_data.get("subtitle", "") or slide_data.get("content", "")
        s_lines = wrap_text_by_pixel(sub_txt, f_sub, text_max_w, draw)
        sy = ty + 30
        for sl in s_lines[:3]:
            draw.text((box_x1 + 85, sy), sl, font=f_sub, fill=text_sub_col)
            sy += 52

        if has_cover_vis:
            try:
                c_img = Image.open(io.BytesIO(slide_data["visual_bytes"])).convert("RGB")
                c_img = ImageOps.fit(c_img, (540, 540), method=Image.Resampling.LANCZOS)
                img.paste(c_img, (box_x2 - 610, box_y1 + 90))
                draw.rectangle([(box_x2 - 610, box_y1 + 90), (box_x2 - 70, box_y1 + 630)], outline=card_border_col, width=4)
            except Exception:
                pass

        draw.line([(box_x1 + 85, box_y2 - 135), (box_x2 - 85, box_y2 - 135)], fill=card_border_col, width=3)
        footer_info = f"품질안전부문 식품안전팀 곽정혁  |  {datetime.date.today().strftime('%Y년 %m월')}"
        draw.text((box_x1 + 85, box_y2 - 92), footer_info, font=f_card_title, fill=text_main_col)
        img.save(output_path, quality=95)
        return output_path

    # 공통 상단 타이틀 헤더 바
    header_bg = (30, 58, 138) if is_light else (15, 28, 48)
    draw.rectangle([(60, 46), (1860, 152)], fill=header_bg, outline=card_border_col, width=3)
    draw.rectangle([(60, 46), (82, 152)], fill=(56, 189, 248) if is_light else (16, 185, 129))
    draw.text((108, 70), slide_data.get("title", f"슬라이드 {slide_idx + 1}"), font=f_title, fill=(255, 255, 255))

    # ---------------------------------------------------------
    # [4] TABLE_MATRIX: 어긋남 없는 정밀 표 렌더링
    # ---------------------------------------------------------
    if layout == "TABLE_MATRIX" and slide_data.get("table_headers") and slide_data.get("table_rows"):
        headers = slide_data["table_headers"]
        rows = slide_data["table_rows"]
        col_cnt = max(1, len(headers))
        table_x1, table_y1, table_x2, table_y2 = 65, 184, 1855, 972
        col_w = (table_x2 - table_x1) // col_cnt

        th_h = 84
        for c_i, h_text in enumerate(headers):
            cx1 = table_x1 + c_i * col_w
            cx2 = table_x2 if c_i == col_cnt - 1 else (cx1 + col_w)
            draw.rectangle([(cx1, table_y1), (cx2, table_y1 + th_h)], fill=(30, 64, 175) if is_light else (6, 78, 59), outline=card_border_col, width=3)
            draw.text((cx1 + 24, table_y1 + 24), str(h_text), font=f_card_title, fill=(255, 255, 255))

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
                c_font = f_card_title if c_i == 0 else f_body
                c_lines = wrap_text_by_pixel(cell_txt, c_font, col_w - 42, draw)
                cy = ry1 + 20
                for cl in c_lines:
                    if cy > ry2 - 36:
                        break
                    draw.text((cx1 + 22, cy), cl, font=c_font, fill=text_main_col if c_i == 0 else text_sub_col)
                    cy += 39

    # ---------------------------------------------------------
    # [5] SPLIT_VISUAL: 좌측 고해상도 일러스트/도면/현장사진 + 우측 핵심 카드
    # ---------------------------------------------------------
    elif layout == "SPLIT_VISUAL" or slide_data.get("visual_bytes"):
        vx1, vy1, vx2, vy2 = 65, 184, 915, 970
        draw.rounded_rectangle([(vx1, vy1), (vx2, vy2)], radius=18, fill=card_bg_col, outline=card_border_col, width=4)

        v_bytes = slide_data.get("visual_bytes")
        if not v_bytes:
            v_bytes = generate_procedural_infographic(slide_data.get("title", ""), slide_data.get("visual_prompt", ""), theme_mode, custom_font_paths=custom_paths)

        try:
            v_img = Image.open(io.BytesIO(v_bytes)).convert("RGB")
            v_img = ImageOps.fit(v_img, (vx2 - vx1 - 28, vy2 - vy1 - 28), method=Image.Resampling.LANCZOS)
            img.paste(v_img, (vx1 + 14, vy1 + 14))
        except Exception:
            pass

        cards = slide_data.get("cards", [])
        if not cards:
            cards = [{"header": "핵심 위생 관리 포인트", "body": slide_data.get("content", "")}]
        rx1, rx2 = 950, 1855
        c_cnt = min(len(cards), 3)
        gap = 24
        card_h = (786 - gap * (max(1, c_cnt) - 1)) // max(1, c_cnt)
        for i, c_item in enumerate(cards[:c_cnt]):
            cy1 = 184 + i * (card_h + gap)
            cy2 = cy1 + card_h
            draw.rounded_rectangle([(rx1, cy1), (rx2, cy2)], radius=16, fill=card_bg_col, outline=card_border_col, width=3)
            draw.rectangle([(rx1, cy1), (rx1 + 18, cy2)], fill=card_border_col)

            header_str = c_item.get("header", f"핵심 기준 {i+1}")
            metric_str = str(c_item.get("metric", "")).strip()
            draw.text((rx1 + 38, cy1 + 22), header_str, font=f_card_title, fill=card_border_col if is_light else (52, 211, 153))
            if metric_str:
                m_bb = draw.textbbox((0, 0), metric_str, font=f_card_title)
                mw = m_bb[2] - m_bb[0]
                draw.rounded_rectangle([(rx2 - mw - 56, cy1 + 18), (rx2 - 24, cy1 + 60)], radius=8, fill=(254, 226, 226) if is_light else (127, 29, 29))
                draw.text((rx2 - mw - 40, cy1 + 22), metric_str, font=f_card_title, fill=(220, 38, 38) if is_light else (254, 202, 202))

            b_lines = wrap_text_by_pixel(c_item.get("body", ""), f_body, (rx2 - rx1) - 68, draw)
            by = cy1 + 74
            for bl in b_lines:
                if by > cy2 - 36:
                    break
                draw.text((rx1 + 38, by), bl, font=f_body, fill=text_sub_col)
                by += 40

    # ---------------------------------------------------------
    # [6] CARDS_3COL / TOC_STEP / SELF_AUDIT_2X2
    # ---------------------------------------------------------
    else:
        cards = slide_data.get("cards", [])
        if not cards:
            raw_lines = [l.strip() for l in slide_data.get("content", "").split("\n") if l.strip()]
            cards = [{"header": f"핵심 포인트 {i+1}", "body": line, "metric": ""} for i, line in enumerate(raw_lines[:4])]

        c_cnt = min(len(cards), 5)
        if c_cnt == 4 and layout == "SELF_AUDIT_2X2":
            grid_coords = [
                (65, 184, 940, 564),
                (980, 184, 1855, 564),
                (65, 590, 940, 970),
                (980, 590, 1855, 970)
            ]
            for i in range(4):
                gx1, gy1, gx2, gy2 = grid_coords[i]
                c_item = cards[i]
                draw.rounded_rectangle([(gx1, gy1), (gx2, gy2)], radius=18, fill=card_bg_col, outline=card_border_col, width=3)
                draw.rectangle([(gx1, gy1), (gx2, gy1 + 68)], fill=header_bg)
                draw.text((gx1 + 28, gy1 + 17), c_item.get("header", ""), font=f_card_title, fill=(255, 255, 255))
                b_lines = wrap_text_by_pixel(c_item.get("body", ""), f_body, (gx2 - gx1) - 56, draw)
                by = gy1 + 90
                for bl in b_lines:
                    if by > gy2 - 38:
                        break
                    draw.text((gx1 + 28, by), bl, font=f_body, fill=text_sub_col)
                    by += 42
        else:
            total_w = 1790
            gap = 28
            card_w = (total_w - gap * (max(1, c_cnt) - 1)) // max(1, c_cnt)
            for i, c_item in enumerate(cards[:c_cnt]):
                cx1 = 65 + i * (card_w + gap)
                cx2 = cx1 + card_w
                cy1, cy2 = 184, 970
                draw.rounded_rectangle([(cx1, cy1), (cx2, cy2)], radius=18, fill=card_bg_col, outline=card_border_col, width=4)
                draw.rectangle([(cx1, cy1), (cx2, cy1 + 84)], fill=header_bg)
                h_lines = wrap_text_by_pixel(c_item.get("header", f"STEP {i+1}"), f_card_title, card_w - 36, draw)
                if h_lines:
                    draw.text((cx1 + 22, cy1 + 24), h_lines[0], font=f_card_title, fill=(255, 255, 255))

                by = cy1 + 114
                metric_str = str(c_item.get("metric", "")).strip()
                if metric_str:
                    m_lines = wrap_text_by_pixel(metric_str, f_metric, card_w - 36, draw)
                    for ml in m_lines[:2]:
                        draw.text((cx1 + 26, by), ml, font=f_metric, fill=(239, 68, 68))
                        by += 80
                    draw.line([(cx1 + 26, by + 10), (cx2 - 26, by + 10)], fill=card_border_col, width=3)
                    by += 32

                b_lines = wrap_text_by_pixel(c_item.get("body", ""), f_body, card_w - 48, draw)
                for bl in b_lines:
                    if by > cy2 - 42:
                        break
                    draw.text((cx1 + 24, by), bl, font=f_body, fill=text_sub_col)
                    by += 43

    # 하단 요약 원칙 배너 & 페이지 번호
    bottom_banner = slide_data.get("bottom_summary", "")
    if bottom_banner:
        draw.rounded_rectangle([(65, 990), (1670, 1054)], radius=12, fill=(254, 242, 242) if is_light else (6, 78, 59), outline=(239, 68, 68) if is_light else (16, 185, 129), width=3)
        b_sum_lines = wrap_text_by_pixel(f"핵심 원칙: {bottom_banner}", f_body, 1560, draw)
        if b_sum_lines:
            draw.text((92, 1006), b_sum_lines[0], font=f_body, fill=(185, 28, 28) if is_light else (255, 255, 255))

    draw.text((width - 210, height - 52), f"Page {slide_idx + 1} / {total_slides}", font=f_small, fill=text_sub_col)
    img.save(output_path, quality=95)
    return output_path

# ==========================================
# [10] 고해상도 이미지 기반 파워포인트(.pptx) 생성기 (노트북 스타일 100% 일치)
# ==========================================
def generate_highres_image_pptx(slides_data, font_config, theme_mode, leader_photos=None):
    """
    미리보기와 100% 똑같은 Full HD(1920x1080) 렌더링 이미지를 각 슬라이드 전체 화면(16:9)에 얹어
    표 깨짐, 도형 밀림, PC별 폰트 누락을 완벽하게 차단합니다. (발표자 노트에 성우 대본 자동 수록)
    """
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

def build_fixed_event_slide(month_int, rank1, rank2, rank3, prize_amount="10,000원", prize_item="상당의 커피 기프티콘 증정"):
    month_str = f"{month_int}월"
    ranks_str = f"{rank1}번째, {rank2}번째, {rank3}번째"
    script_text = (
        f"자, 오늘 준비한 {month_str} 정기 위생교육 내용은 여기까지입니다! 현장에서 늘 애써주시는 임직원 여러분 정말 고생 많으셨습니다. "
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
# [11] Gemini AI 기획 엔진 (시각 도면 강화 + 구어체/생동감 대본 설계)
# ==========================================
def generate_ai_storyboard(model_name, topic, edu_month_str, body_slide_count, intro_config, uploaded_gemini_files, extra_source_text):
    model = genai.GenerativeModel(model_name)
    generation_config = genai.types.GenerationConfig(temperature=0.25, max_output_tokens=65536)

    if intro_config["mode"] != "당부 말씀 슬라이드 생략":
        intro_instruction = f"""
        [슬라이드 1 (고정 오프닝: '월간 위생교육 전 당부말씀' - 상단 생산본부장 1단 와이드 + 하단 3개 부문장 3열 구조)]:
        - 반드시 첫 번째 슬라이드는 "layout": "INTRO_1TOP_3BOT", "title": "월간 위생교육 전 당부말씀" 으로 설정하십시오.
        - "cards" 배열에 정확히 아래 순서대로 4명의 리더 당부 말씀을 넣으십시오 (입력된 원문 유지):
          1) {{"header": "생산본부", "sub": "생산본부장", "body": "..."}}
          2) {{"header": "유제품생산부문", "sub": "유제품생산부문장", "body": "..."}}
          3) {{"header": "상온제품생산부문", "sub": "상온제품생산부문장", "body": "..."}}
          4) {{"header": "품질안전부문", "sub": "품질안전부문장", "body": "..."}}
        - "script"에는 딱딱한 기계식 낭독이 아니라, 활기차고 신뢰감 있는 사내 방송 진행자 톤으로 생산본부장님과 각 부문장님의 핵심 당부 포인트를 귀에 쏙쏙 들어오게 브리핑하십시오.
        - [입력된 당부 말씀 원문]:\n{intro_config.get("custom_text", "")}
        """
    else:
        intro_instruction = "[인트로 생략]: INTRO_1TOP_3BOT 없이 표지(COVER)부터 시작하십시오."

    system_prompt = f"""
    당신은 연세유업 아산공장 식품안전팀(담당자: 곽정혁)의 수석 위생교육 콘텐츠 디렉터이자 HACCP 전문가입니다.
    학습자가 절대 졸거나 지루해하지 않도록 **시각적인 도면·일러스트가 풍부한 슬라이드 구조**와, **실제 인기 강사나 라디오 DJ처럼 귀에 쏙쏙 꽂히는 생동감 넘치는 구어체 내레이션 대본**을 설계해야 합니다.
    단, 법적 기준·온도·수치·규정 등 모든 교육 내용은 추측 없이 100% 명확한 팩트와 업로드된 자료에만 기반하십시오.
    (※ 마지막 이벤트 안내 슬라이드는 파이썬 코드가 자동 결합하므로 생성하지 마십시오.)

    [교육 월]: {edu_month_str}
    [교육 핵심 주제]: {topic}
    [생성 장수]: 표지(COVER, 1장) + 목차(TOC_STEP, 1장) + 본론 상세({body_slide_count}장)

    {intro_instruction}

    [시각화 및 레이아웃("layout") 배정 핵심 규칙]:
    1. "COVER": 표지 슬라이드에도 우측에 들어갈 멋진 대표 일러스트 영문 프롬프트("visual_prompt")를 반드시 작성하십시오.
    2. "TOC_STEP": 3~5단계 전체 교육 목차 카드.
    3. 본론 슬라이드({body_slide_count}장) 중 **최소 절반 이상(60% 이상)을 반드시 "SPLIT_VISUAL" 레이아웃으로 배정**하여 매 페이지마다 풍부한 그림·도면·현장 시각자료가 함께 보이도록 하십시오!
       - "SPLIT_VISUAL": 좌측에는 큼직한 시각 도면/일러스트/영상캡처가 들어가고 우측에는 2~3개의 핵심 요약 카드가 배치됩니다.
         * "visual_prompt": 슬라이드 핵심 내용을 직관적으로 보여주는 3D 아이소메트릭 공정도, 위생 설비 단면도, 작업자 위생 수칙 일러스트 등을 묘사하는 구체적인 영문 프롬프트를 반드시 작성하십시오 (글자 없는 순수 그림 묘사).
         * 만약 업로드된 참고 동영상(MP4)의 특정 장면을 캡처해 쓰는 것이 더 효과적이라면 해당 시간(초 단위 숫자)을 "video_timestamp_sec"에 적으십시오.
       - "CARDS_3COL": 핵심 수치(예: "3mm 이상", "540 Lux", "15℃ 이하") 강조 비교가 필요할 때 사용 ("metric" 필드에 짧은 강조 수치 기입).
       - "TABLE_MATRIX": 기준표, 행정처분, 4M1E, 구역 비교 등 표가 꼭 필요한 1~2장에 사용 ("table_headers", "table_rows" 필수 작성).
       - "SELF_AUDIT_2X2": 본론 가장 마지막 장은 현장 작업자용 '핵심 자가 진단 체크리스트' 4분할 카드로 구성하십시오 (각 줄 앞에 '☑ ' 또는 '□ ' 표기).

    [성우 내레이션 대본("script") 작성 절대 원칙 - 매우 중요!]:
    - 문어체("~함", "~임", "~바랍니다"의 기계적 반복)를 피하고, **실제 사람이 눈을 맞추며 대화하듯 자연스럽고 생동감 넘치는 구어체("~거든요", "~해보면 어떨까요?", "자, 여기서 정말 중요한 포인트가 하나 있습니다!", "~해주시는 센스, 잊지 마세요!")**를 적절히 섞어 유쾌하고 몰입감 있게 작성하십시오.
    - 오디오 성우가 자연스럽게 숨을 쉬며 읽을 수 있도록 쉼표(,)와 마침표(.)를 호흡 단위로 꼼꼼히 찍어주십시오.
    - 영문 약어(HACCP, CCP 등)는 성우가 자연스럽게 발음하도록 문맥을 매끄럽게 이어주십시오.

    [출력 형식 규칙]:
    먼저 <pre_calc> 태그 안에 슬라이드별 시각 도면 구상과 팩트 검증 요약을 적고, 이어서 <json_output> 태그 안에 순수 JSON 배열만 출력하십시오. (코드 축약 및 '...' 생략 절대 금지)

    <pre_calc>
    사전 설계 요약
    </pre_calc>
    <json_output>
    [
      {{
        "layout": "SPLIT_VISUAL",
        "type": "BODY",
        "title": "슬라이드 제목",
        "subtitle": "",
        "content": "",
        "cards": [
          {{"header": "카드 소제목 1", "sub": "", "metric": "540 Lux", "body": "• 핵심 설명 첫째 줄\\n• 핵심 설명 둘째 줄"}},
          {{"header": "카드 소제목 2", "sub": "", "metric": "", "body": "• 핵심 설명 첫째 줄\\n• 핵심 설명 둘째 줄"}}
        ],
        "table_headers": [],
        "table_rows": [],
        "bottom_summary": "현장 작업자가 기억해야 할 한 줄 핵심 원칙",
        "visual_prompt": "Isometric 3D illustration of a clean dairy factory inspection room with bright LED protective lighting over stainless steel conveyor belt, clean vector style, no text",
        "video_timestamp_sec": null,
        "script": "자, 다음으로 우리 현장의 조명 기준을 한번 살펴볼까요? 선별이나 검사가 이루어지는 구역은 최소 540 룩스 이상의 밝기가 확보되어야 하는데요, 왜 이렇게 밝아야 할까요? 바로 미세한 이물질 하나까지도 우리 눈으로 놓치지 않고 잡아내기 위해서입니다..."
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
            response = model.generate_content(payload, generation_config=generation_config)
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
            return None, None, f"AI 기획안 생성 오류: {e}"

def generate_scripts_for_existing_pdf(model_name, pdf_pages_info, edu_month_str, topic):
    model = genai.GenerativeModel(model_name)
    slides_text_dump = "\n\n".join([f"[슬라이드 {p['page_num']} 텍스트]:\n{p['text']}" for p in pdf_pages_info])
    prompt = f"""
    다음은 연세유업 아산공장 '{edu_month_str} 정기 위생교육 ({topic})' 슬라이드 총 {len(pdf_pages_info)}장의 텍스트입니다.
    슬라이드 1번부터 {len(pdf_pages_info)}번까지 단 한 장도 빠짐없이, 실제 전문 강사가 현장 작업자들과 눈을 맞추며 친근하고 생동감 있게 설명하는 구어체 내레이션 대본(script)을 작성하십시오.
    딱딱한 낭독투 대신 "~거든요", "자, 화면을 함께 보실까요?" 같은 자연스러운 연결어와 호흡용 쉼표(,)를 적극 활용하고, 수치는 100% 정확하게 전달하십시오.

    반드시 아래 순수 JSON 배열 포맷으로만 출력하십시오 (개수: 정확히 {len(pdf_pages_info)}개):
    [
      {{"page_num": 1, "title": "슬라이드 1 요약 제목", "script": "1번 슬라이드 생동감 있는 구어체 대본"}},
      ...
    ]

    [추출된 슬라이드 텍스트 전체]:
    {slides_text_dump}
    """
    resp = model.generate_content(prompt)
    clean_j = re.sub(r'^```json\s*|\s*```$', '', resp.text.strip(), flags=re.MULTILINE).strip()
    return json.loads(clean_j)

# ==========================================
# [12] 메인 애플리케이션 UI
# ==========================================
def main():
    st.markdown(custom_theme_css, unsafe_allow_html=True)

    for key in ["slides_data", "pre_calc_log", "final_video_bytes", "final_pptx_bytes", "custom_font_paths", "uploaded_video_local_path", "leader_photos", "intro_full_slide_bytes"]:
        if key not in st.session_state:
            if key == "custom_font_paths":
                st.session_state[key] = {}
            elif key == "leader_photos":
                st.session_state[key] = load_default_leader_photos_from_repo()
            else:
                st.session_state[key] = None

    st.markdown("<h1>🎬 연세유업 AI 위생교육 스튜디오 <span style='font-size:0.45em; color:#64748b;'>(Full-HD 이미지 PPTX · AI 도면 일러스트 · 초자연 성우 통합판)</span></h1>", unsafe_allow_html=True)
    st.markdown("원본 교육 영상(MP4)·문서 업로드부터 **부문장님 사진 포함 당부말씀 ➡️ 시각 도면 본론 슬라이드 ➡️ 월별 고정 이벤트 ➡️ 깨짐 없는 이미지 PPTX & 자연스러운 성우 MP4**까지 원스톱으로 완성합니다.")
    st.markdown("<hr>", unsafe_allow_html=True)

    # --- 좌측 사이드바 ---
    with st.sidebar:
        st.header("🎙️ 성우 목소리 엔진 설정")
        tts_engine = st.radio(
            "음성 합성 엔진 선택",
            ["OpenAI TTS (ChatGPT 초자연 구어체 성우)", "Edge Neural TTS (무료 남성/여성 아나운서)"],
            index=0 if DEFAULT_OPENAI_KEY else 1
        )
        openai_key_input = st.text_input(
            "OpenAI API Key (OpenAI TTS / DALL-E 도면용)",
            value=DEFAULT_OPENAI_KEY,
            type="password",
            help="Streamlit Secrets에 OPENAI_API_KEY를 등록해 두면 매번 입력하지 않아도 자동 연결됩니다."
        )

        if "OpenAI" in tts_engine:
            openai_model = st.selectbox("OpenAI 음성 모델", ["gpt-4o-mini-tts", "tts-1-hd"], index=0, help="gpt-4o-mini-tts는 강사/팟캐스트 억양 지시가 반영되어 가장 사람처럼 자연스럽습니다.")
            voice_labels = {
                "echo (친근하고 부드러운 남성 - 추천)": "echo",
                "onyx (신뢰감 있는 중저음 남성)": "onyx",
                "alloy (발랄하고 명쾌한 챗GPT 기본톤)": "alloy",
                "ash (생동감 넘치는 구어체 남성)": "ash",
                "nova (밝고 활기찬 여성)": "nova",
                "shimmer (차분한 여성 내레이션)": "shimmer"
            }
            chosen_v_label = st.selectbox("ChatGPT 보이스 선택", list(voice_labels.keys()), index=0)
            openai_voice = voice_labels[chosen_v_label]
            edge_voice = "ko-KR-InJoonNeural"
        else:
            openai_model = "tts-1-hd"
            openai_voice = "echo"
            edge_labels = {
                "인준 (자연스러운 남성 아나운서 - 추천)": "ko-KR-InJoonNeural",
                "현수 (차분한 남성 소통형 보이스)": "ko-KR-HyunsuNeural",
                "선희 (밝고 또렷한 여성 아나운서)": "ko-KR-SunHiNeural"
            }
            chosen_e_label = st.selectbox("무료 신경망 성우 선택", list(edge_labels.keys()), index=0)
            edge_voice = edge_labels[chosen_e_label]

        voice_speed = st.slider("말하기 속도 배율", min_value=0.90, max_value=1.20, value=1.04, step=0.02)

        tts_config = {
            "engine": tts_engine,
            "openai_key": openai_key_input,
            "openai_model": openai_model,
            "openai_voice": openai_voice,
            "edge_voice": edge_voice,
            "speed": voice_speed
        }

        if st.button("🔊 현재 선택한 목소리 3초 미리듣기", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_aud:
                sample_path = tmp_aud.name
            create_natural_tts_audio("안녕하세요! 연세유업 아산공장 임직원 여러분, 오늘도 활기차고 유익한 위생교육을 함께 시작해 볼까요?", sample_path, tts_config)
            with open(sample_path, "rb") as af:
                st.audio(af.read(), format="audio/mp3")

        st.markdown("---")
        st.header("🎨 AI 두뇌 & 시각 테마 설정")
        selected_model = st.selectbox(
            "Gemini 기획 모델",
            ["gemini-3.8-flash", "gemini-omni-1.2-flash", "gemini-2.5-flash", "gemini-4-argon"],
            index=0
        )
        theme_mode = st.radio(
            "슬라이드 디자인 테마",
            ["10월 스타일 (라이트 블루프린트 모눈 테마)", "9월 스타일 (다크 사이버네틱 네온 테마)"],
            index=0
        )
        enable_ai_drawing = st.checkbox("슬라이드에 AI 시각 도면/일러스트 자동 생성", value=True)

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
        "1. 교육 영상/자료 입력 & 당부말씀·이벤트 설정",
        "2. 고해상도 슬라이드 미리보기 & 그림·대본 편집",
        "3. 고해상도 이미지 PPTX & 성우 MP4 렌더링"
    ])

    # ==========================================
    # 탭 1: 입력 및 기획 생성
    # ==========================================
    with tab1:
        col_m1, col_m2 = st.columns([1, 2.5])
        with col_m1:
            edu_month_num = st.number_input("교육 기준 월 (이벤트 순번 자동 연동):", min_value=1, max_value=12, value=11, step=1)
            edu_month_str = f"{edu_month_num}월"
        with col_m2:
            topic = st.text_input("이번 달 위생교육 핵심 주제:", value="작업장 이물혼입 예방 및 영업장 위생관리")

        st.markdown("### 🎥 Step 1. 참고할 원본 교육 영상(MP4) 또는 교육 자료 업로드")
        workflow_mode = st.radio(
            "본론 슬라이드 생성 방식:",
            [
                "표준 모드: 참고 교육 영상(MP4)·문서(PPTX/PDF) 또는 주제 입력 ➡️ AI가 시각 도면 포함 고해상도 슬라이드 & 구어체 대본 새로 제작",
                "PDF 직행 모드: 이미 완성된 슬라이드 PDF 업로드 ➡️ 원본 디자인 그대로 유지하고 앞뒤에 당부말씀·이벤트 결합 및 성우 더빙"
            ],
            index=0
        )

        uploaded_sources = []
        extra_source_text = ""
        existing_pdf_file = None
        body_slide_count = 8

        if "표준 모드" in workflow_mode:
            col_s1, col_s2 = st.columns([1.2, 1])
            with col_s1:
                uploaded_sources = st.file_uploader(
                    "참고할 원본 교육 영상(MP4) 또는 문서(PDF, PPTX) 업로드 (복수 선택 가능):",
                    type=["mp4", "pdf", "pptx"],
                    accept_multiple_files=True
                )
                body_slide_count = st.slider("본론 슬라이드 생성 장수 (당부말씀/표지/목차/이벤트 제외):", 4, 14, 8)
            with col_s2:
                extra_source_text = st.text_area(
                    "추가 반영할 사내 수칙 / 한계기준 수치 (선택):",
                    height=135,
                    placeholder="예: 검사구역 조도 540 Lux 이상, 금속 이물 2mm 이상 검출 불가, 자석봉 오전/오후 1회 이상 세척 등"
                )
        else:
            existing_pdf_file = st.file_uploader("완성된 교육 자료 PDF 파일 업로드:", type=["pdf"])

        st.markdown("---")

        # --- Step 2: 당부 말씀 (얼굴 사진 자동 추출 & 슬라이드 이미지 직행 모두 지원) ---
        st.markdown("### 👔 Step 2. 오프닝: 월간 위생교육 전 당부말씀 (상단 생산본부장 + 하단 3부문장 & 얼굴 사진 연동)")
        intro_mode = st.radio(
            "당부 말씀 적용 방식:",
            [
                "PPTX 업로드 또는 텍스트 입력 (부문장님 얼굴 사진 자동 추출 + 상단1·하단3 표준 레이아웃 렌더링)",
                "당부 말씀 슬라이드 완성본 이미지/PDF 통째로 올리기 (원본 슬라이드 그림 100% 그대로 사용)",
                "당부 말씀 슬라이드 생략"
            ]
        )

        intro_custom_text = ""
        ldr_prod = ldr_dairy = ldr_ambient = ldr_qc = ""

if "PPTX 업로드" in intro_mode:
            st.session_state["intro_full_slide_bytes"] = None
            # 깃허브 고정 사진이 비어있다면 다시 한 번 스캔하여 항상 고정 유지
            if not st.session_state.get("leader_photos"):
                st.session_state["leader_photos"] = load_default_leader_photos_from_repo()

            leader_pptx = st.file_uploader("📄 '위생교육 전 말씀.pptx' 업로드 (4분 말씀 텍스트만 자동 추출 / 사진은 깃허브 고정 사진 유지):", type=["pptx"], key="ldr_pptx")
            parsed_leaders = {}
            if leader_pptx:
                parsed_leaders = parse_leader_pptx_text_only(leader_pptx)
                st.success("PPTX에서 4분의 당부 말씀 텍스트만 깔끔하게 추출했습니다! (프로필 사진은 깃허브 원본 고정)")
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

            with st.expander("📸 부문장님 프로필 사진 개별 확인 / 교체 (선택)"):
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
                    if ext_i == ".pdf" and PYMUPDF_AVAILABLE:
                        p_list = extract_pages_from_pdf(intro_slide_file.getvalue())
                        if p_list:
                            st.session_state["intro_full_slide_bytes"] = p_list[0]["image_bytes"]
                            intro_custom_text = p_list[0]["text"]
                    else:
                        st.session_state["intro_full_slide_bytes"] = intro_slide_file.getvalue()
            with col_fs2:
                intro_custom_text = st.text_area("성우가 읽을 당부 말씀 요약/원문 입력:", value=intro_custom_text or "생산본부장님 및 유제품, 상온, 품질안전 각 부문장님께서 철저한 이물 관리와 기본 위생 수칙 준수를 당부하셨습니다.", height=120)

        st.markdown("---")

        # --- Step 3: 고정 양식 이벤트 안내 ---
        st.markdown(f"### 🎁 Step 3. 클로징: {edu_month_str} 위생교육 이벤트 안내 (월별 순번 자동 계산 고정 양식)")
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

        if st.button("🚀 고해상도 시각 슬라이드 & 구어체 성우 대본 일괄 기획 시작", type="primary", use_container_width=True):
            if not GEMINI_API_KEY:
                st.error("Streamlit Secrets에 GEMINI_API_KEY를 설정해주세요.")
            elif "표준 모드" in workflow_mode:
                with st.spinner("원본 자료 분석, 구어체 성우 대본 작성 및 각 슬라이드별 시각 도면/일러스트를 생성하고 있습니다..."):
                    gemini_files = []
                    combined_text = extra_source_text
                    extracted_ref_images = []
                    st.session_state["uploaded_video_local_path"] = None

                    for f in uploaded_sources:
                        ext = os.path.splitext(f.name)[1].lower()
                        if ext == ".pptx":
                            p_txt, p_imgs = extract_text_and_Media_from_pptx(f)
                            combined_text += f"\n\n[참고 PPTX ({f.name}) 내용]:\n{p_txt}"
                            extracted_ref_images.extend([x["bytes"] for x in p_imgs])
                        else:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                                tmp.write(f.getbuffer())
                                tmp_path = tmp.name
                            if ext == ".mp4":
                                st.session_state["uploaded_video_local_path"] = tmp_path
                            up_f = genai.upload_file(tmp_path)
                            while up_f.state.name == "PROCESSING":
                                time.sleep(2)
                                up_f = genai.get_file(up_f.name)
                            gemini_files.append(up_f)

                    intro_cfg = {"mode": intro_mode, "custom_text": intro_custom_text}
                    slides, pre_log, err = generate_ai_storyboard(
                        selected_model, topic, edu_month_str, body_slide_count, intro_cfg, gemini_files, combined_text
                    )
                    if err:
                        st.error(err)
                    else:
                        slides = [s for s in slides if s.get("layout") != "OUTRO_EVENT"]

                        # 오프닝 당부말씀 슬라이드 데이터 확정
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

# [수정 핵심] 서브 스레드 진입 전, 메인 스레드에서 st.session_state 값을 일반 변수로 미리 추출!
                        local_video_path = st.session_state.get("uploaded_video_local_path")
                        local_custom_fonts = dict(st.session_state.get("custom_font_paths", {}))

                        # 병렬로 슬라이드 시각 일러스트/영상 프레임 채우기 (내부에서 st.session_state 호출 절대 금지)
                        def _fill_visual(idx_s, s_obj):
                            if s_obj.get("layout") in ("SPLIT_VISUAL", "COVER") and not s_obj.get("visual_bytes"):
                                ts = s_obj.get("video_timestamp_sec")
                                if ts is not None and local_video_path:
                                    fb = capture_frame_from_video(local_video_path, ts)
                                    if fb:
                                        s_obj["visual_bytes"] = fb
                                        return idx_s, s_obj
                                if enable_ai_drawing:
                                    s_obj["visual_bytes"] = generate_ai_illustration(
                                        s_obj.get("visual_prompt", ""),
                                        s_obj.get("title", topic),
                                        theme_mode,
                                        openai_key=openai_key_input,
                                        gemini_key=GEMINI_API_KEY,
                                        custom_font_paths=local_custom_fonts
                                    )
                            return idx_s, s_obj

                        with ThreadPoolExecutor(max_workers=4) as vis_exec:
                            vis_futures = [vis_exec.submit(_fill_visual, i, s) for i, s in enumerate(slides)]
                            for vf in as_completed(vis_futures):
                                r_i, r_s = vf.result()
                                slides[r_i] = r_s

                        if include_outro:
                            slides.append(build_fixed_event_slide(edu_month_num, ev_rank1, ev_rank2, ev_rank3, ev_prize_amount))

                        st.session_state["slides_data"] = slides
                        st.session_state["pre_calc_log"] = pre_log
                        st.session_state["final_video_bytes"] = None
                        st.session_state["final_pptx_bytes"] = None
                        st.success(f"총 {len(slides)}장의 고해상도 시각 슬라이드 및 구어체 대본 생성이 완료되었습니다! [2번 탭]에서 확인하세요.")
            else:
                if not existing_pdf_file:
                    st.warning("PDF 파일을 먼저 업로드해주세요.")
                elif not PYMUPDF_AVAILABLE:
                    st.error("requirements.txt에 PyMuPDF를 추가해주세요.")
                else:
                    with st.spinner("PDF 전 페이지 고해상도 추출 및 구어체 더빙 대본 생성 중..."):
                        pages_info = extract_pages_from_pdf(existing_pdf_file.getvalue())
                        scripts_list = generate_scripts_for_existing_pdf(selected_model, pages_info, edu_month_str, topic)
                        built_slides = []

                        if intro_mode != "당부 말씀 슬라이드 생략":
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
                                "script": f"안녕하세요, 연세유업 아산공장 임직원 여러분! {edu_month_str} 정기 위생교육을 시작하기에 앞서, 생산본부장님과 각 부문장님들의 당부 말씀을 먼저 전해드립니다. {ldr_prod} 이어서 유제품생산부문장님께서는 {ldr_dairy} 상온제품생산부문장님께서는 {ldr_ambient} 끝으로 품질안전부문장님께서는 {ldr_qc}"
                            }
                            if st.session_state.get("intro_full_slide_bytes") and "통째로" in intro_mode:
                                intro_slide_obj["custom_image_bytes"] = st.session_state["intro_full_slide_bytes"]
                            built_slides.append(intro_slide_obj)

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
                            built_slides.append(build_fixed_event_slide(edu_month_num, ev_rank1, ev_rank2, ev_rank3, ev_prize_amount))

                        st.session_state["slides_data"] = built_slides
                        st.session_state["final_video_bytes"] = None
                        st.session_state["final_pptx_bytes"] = None
                        st.success(f"총 {len(built_slides)}장의 슬라이드 및 구어체 대본 준비 완료! [2번 탭]으로 이동하세요.")

    # ==========================================
    # 탭 2: Full-HD 슬라이드 미리보기 & 시각 그림/대본 수정
    # ==========================================
    with tab2:
        if not st.session_state.get("slides_data"):
            st.info("먼저 [1번 탭]에서 슬라이드 기획을 실행해 주십시오.")
        else:
            st.markdown("### 🖥️ Full-HD(1920×1080) 슬라이드 미리보기 및 현장 사진·대본 편집")
            st.caption("이 화면에 보이는 고해상도 이미지 그대로 파워포인트(.pptx) 슬라이드와 MP4 교육 영상에 100% 똑같이 들어갑니다.")

            total_s = len(st.session_state["slides_data"])
            updated_slides = []

            with tempfile.TemporaryDirectory() as preview_dir:
                for idx, slide in enumerate(st.session_state["slides_data"]):
                    is_expanded = (idx in (0, 1, total_s - 1))
                    with st.expander(f"슬라이드 {idx + 1} [{slide.get('layout', 'SLIDE')}] : {slide.get('title', '')}", expanded=is_expanded):
                        col_p1, col_p2 = st.columns([1.2, 1])
                        with col_p2:
                            new_title = st.text_input("슬라이드 제목", value=slide.get("title", ""), key=f"t_{idx}")
                            new_script = st.text_area("성우 내레이션 대본 (구어체)", value=slide.get("script", ""), height=130, key=f"s_{idx}")

                            c_u1, c_u2 = st.columns(2)
                            with c_u1:
                                part_vis = st.file_uploader("좌측 시각 그림/현장사진만 교체", type=["png", "jpg", "jpeg"], key=f"pv_{idx}")
                                if part_vis:
                                    slide["visual_bytes"] = part_vis.getvalue()
                                    slide["layout"] = "SPLIT_VISUAL"
                            with c_u2:
                                full_rep = st.file_uploader("슬라이드 전체 이미지 통째 교체", type=["png", "jpg", "jpeg"], key=f"fr_{idx}")
                                if full_rep:
                                    slide["custom_image_bytes"] = full_rep.getvalue()

                        slide["title"] = new_title
                        slide["script"] = new_script

                        with col_p1:
                            preview_img_path = os.path.join(preview_dir, f"preview_{idx}.png")
                            create_slide_image_hd(slide, idx, total_s, preview_img_path, font_config, theme_mode, leader_photos=st.session_state["leader_photos"])
                            st.image(preview_img_path, caption=f"슬라이드 {idx + 1} (1920x1080 고해상도 렌더링)", use_container_width=True)

                        updated_slides.append(slide)

            st.session_state["slides_data"] = updated_slides

    # ==========================================
    # 탭 3: 고해상도 이미지 PPTX 및 자연스러운 성우 MP4 렌더링
    # ==========================================
    with tab3:
        if not st.session_state.get("slides_data"):
            st.info("먼저 [1번 탭]에서 슬라이드 기획을 실행해 주십시오.")
        else:
            slides_to_render = st.session_state["slides_data"]
            st.markdown(f"### 🎬 최종 결과물 렌더링 및 다운로드 (총 {len(slides_to_render)}장 · 1920×1080 Full-HD)")

            col_b1, col_b2 = st.columns(2)
            with col_b1:
                if st.button("📊 1단계: 고해상도 이미지 슬라이드 PPTX 생성 (깨짐 0%)", use_container_width=True):
                    with st.spinner("1920x1080 고해상도 슬라이드 이미지 기반 PPTX 파일을 생성하고 있습니다..."):
                        st.session_state["final_pptx_bytes"] = generate_highres_image_pptx(
                            slides_to_render,
                            font_config,
                            theme_mode,
                            leader_photos=st.session_state["leader_photos"]
                        )
                        st.success("고해상도 이미지 PPTX 생성 완료! (미리보기와 100% 동일하며 발표자 노트에 대본이 포함되어 있습니다)")

                if st.session_state.get("final_pptx_bytes"):
                    st.download_button(
                        label="📥 파워포인트(.pptx) 파일 다운로드",
                        data=st.session_state["final_pptx_bytes"],
                        file_name=f"연세유업_{edu_month_str}_위생교육_{datetime.date.today().strftime('%Y%m%d')}.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        use_container_width=True
                    )

            with col_b2:
                if st.button("🎥 2단계: 자연스러운 성우 더빙 + Full-HD MP4 영상 렌더링", type="primary", use_container_width=True):
                    t_start = time.time()
                    with tempfile.TemporaryDirectory() as temp_dir:
                        total_cnt = len(slides_to_render)
                        p_bar = st.progress(0, text="병렬 엔진 가동: Full-HD 슬라이드 및 자연스러운 성우 오디오 동시 합성 중...")

                      media_results = [None] * total_cnt
                        local_leader_photos = dict(st.session_state.get("leader_photos", {}))
                        with ThreadPoolExecutor(max_workers=4) as executor:
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

                        p_bar.progress(total_cnt / (total_cnt + 1), text="최종 Full-HD MP4 교육 영상 병합 중...")
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
                                fps=10,
                                codec="libx264",
                                audio_codec="aac",
                                preset="ultrafast",
                                threads=4,
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
                    file_name=f"연세유업_{edu_month_str}_위생교육영상_{datetime.date.today().strftime('%Y%m%d')}.mp4",
                    mime="video/mp4",
                    type="primary",
                    use_container_width=True
                )

if __name__ == "__main__":
    if check_password():
        main()
