import streamlit as st
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import socket
import time
import json
import os
import io
import re
import datetime
import fitz  # PyMuPDF (PDF 고해상도 이미지 변환 및 추출용)
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip, AudioFileClip, VideoFileClip, concatenate_videoclips
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# ==========================================
# [1] UI 레이아웃 및 네트워크 방어 설정 (최상단 필수)
# ==========================================
st.set_page_config(page_title="연세유업 AI 교육 영상 스튜디오", layout="wide")
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
# [3] API 키 로드 및 모델 설정
# ==========================================
API_KEY = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# ==========================================
# [4] 커스텀 CSS (연세체 및 맑은 고딕 웹 폰트 우선 적용)
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
# [5] 전용 글씨체(연세제목체, 연세본문체, 맑은고딕) 로더
# ==========================================
FONT_SEARCH_MAP = {
    "연세제목체": {
        "exact_files": [
            "연세제목체.ttf", "연세제목체B.ttf", "연세제목체L.ttf", "연세제목체.otf",
            "YonseiTitle.ttf", "YonseiTitleB.ttf", "YonseiTitle.otf",
            "fonts/연세제목체.ttf", "fonts/YonseiTitle.ttf",
            "C:/Windows/Fonts/연세제목체.ttf", "C:/Windows/Fonts/YonseiTitle.ttf"
        ],
        "keywords": ["연세제목", "yonseititle", "연세_제목"],
        "pptx_name": "연세제목체"
    },
    "연세본문체": {
        "exact_files": [
            "연세본문체.ttf", "연세본문체B.ttf", "연세본문체L.ttf", "연세본문체.otf",
            "YonseiBody.ttf", "YonseiBodyB.ttf", "YonseiBody.otf",
            "fonts/연세본문체.ttf", "fonts/YonseiBody.ttf",
            "C:/Windows/Fonts/연세본문체.ttf", "C:/Windows/Fonts/YonseiBody.ttf"
        ],
        "keywords": ["연세본문", "yonseibody", "연세_본문"],
        "pptx_name": "연세본문체"
    },
    "맑은고딕": {
        "exact_files": [
            "malgunbd.ttf", "malgun.ttf", "맑은고딕.ttf", "MalgunGothic.ttf",
            "fonts/malgunbd.ttf", "fonts/malgun.ttf",
            "C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/malgun.ttf"
        ],
        "keywords": ["malgun", "맑은고딕", "맑은 고딕"],
        "pptx_name": "맑은 고딕"
    }
}

def find_font_file(font_choice, is_bold=False, custom_font_paths=None):
    if custom_font_paths and custom_font_paths.get(font_choice):
        custom_p = custom_font_paths[font_choice]
        if os.path.exists(custom_p):
            return custom_p

    info = FONT_SEARCH_MAP.get(font_choice, FONT_SEARCH_MAP["맑은고딕"])
    if font_choice == "맑은고딕":
        preferred = ["malgunbd.ttf", "fonts/malgunbd.ttf", "C:/Windows/Fonts/malgunbd.ttf"] if is_bold else ["malgun.ttf", "fonts/malgun.ttf", "C:/Windows/Fonts/malgun.ttf"]
        for p in preferred:
            if os.path.exists(p):
                return p

    for p in info["exact_files"]:
        if os.path.exists(p):
            return p

    for search_dir in [".", "fonts"]:
        if os.path.exists(search_dir) and os.path.isdir(search_dir):
            for fname in os.listdir(search_dir):
                lower_f = fname.lower()
                if lower_f.endswith((".ttf", ".otf", ".ttc")):
                    for kw in info["keywords"]:
                        if kw in lower_f:
                            return os.path.join(search_dir, fname)

    fallback_candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf" if is_bold else "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "C:/Windows/Fonts/malgunbd.ttf" if is_bold else "C:/Windows/Fonts/malgun.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc"
    ]
    for fb in fallback_candidates:
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

def extract_text_from_pptx(file_obj):
    try:
        prs = Presentation(file_obj)
        full_text = []
        for idx, slide in enumerate(prs.slides):
            slide_texts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        txt = paragraph.text.strip()
                        if txt:
                            slide_texts.append(txt)
            if slide_texts:
                full_text.append(f"[슬라이드 {idx+1}]\n" + "\n".join(slide_texts))
        return "\n\n".join(full_text)
    except Exception as e:
        return f"[PPTX 추출 오류]: {e}"

def parse_leader_pptx(file_obj):
    raw = extract_text_from_pptx(file_obj)
    result = {
        "생산본부": "",
        "유제품생산부문": "",
        "상온제품생산부문": "",
        "품질안전부문": ""
    }
    lines = [l.strip() for l in raw.split("\n") if l.strip() and not l.startswith("[슬라이드") and "당부말씀" not in l]
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
            result[current_key] = (result[current_key] + " " + line).strip()
    return result

def extract_pages_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_info = []
    for page_idx in range(len(doc)):
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
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
        safe_sec = min(max(0.0, float(sec)), max(0.0, clip.duration - 0.2))
        frame = clip.get_frame(safe_sec)
        clip.close()
        pil_img = Image.fromarray(frame)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None

def generate_ai_illustration(visual_prompt, theme_style):
    if not visual_prompt:
        return None
    style_prefix = (
        "Clean technical blueprint architectural diagram on a light grid background, isometric 3D food factory hygiene illustration, high contrast, no gibberish text: "
        if "10월" in theme_style else
        "Cybernetic dark navy background with glowing neon green and cyan isometric vector line art, food safety and factory inspection infographic, clean minimal: "
    )
    for model_candidate in ["gemini-omni-1.2-flash", "imagen-3.0-generate-002"]:
        try:
            model = genai.GenerativeModel(model_candidate)
            resp = model.generate_content(style_prefix + visual_prompt)
            if resp.candidates and resp.candidates[0].content.parts:
                for part in resp.candidates[0].content.parts:
                    if hasattr(part, "inline_data") and part.inline_data and part.inline_data.data:
                        return part.inline_data.data
        except Exception:
            continue
    return None

# ==========================================
# [6] 고정 이벤트 슬라이드 빌더 및 아이콘 드로잉 헬퍼
# ==========================================
def build_fixed_event_slide(edu_month_str, ranks_str, prize_amount_str, prize_item_str):
    """
    '이벤트 페이지.pptx' 양식 그대로 월과 당첨 순번만 바꿔 고정 슬라이드 데이터와 성우 대본을 생성합니다.
    """
    m_clean = str(edu_month_str).strip()
    if not m_clean.endswith("월"):
        m_clean += "월"
    r_clean = str(ranks_str).strip()
    p_amt = str(prize_amount_str).strip()
    p_item = str(prize_item_str).strip()

    content_summary = (
        f"마지막으로 {m_clean} 위생교육 이벤트 안내입니다!\n"
        f"1) {m_clean}은 만점자 중 순번 {r_clean} 분에게\n"
        f"2) {p_amt} {p_item}\n"
        f"3) 많은 관심과 적극적인 참여 부탁드립니다."
    )
    script_text = (
        f"이상으로 교육을 마치겠습니다. 마지막으로 {m_clean} 위생교육 이벤트 안내입니다. "
        f"{m_clean}은 위생교육 평가 만점자 중 순번 {r_clean} 분에게 "
        f"{p_amt} {p_item}을 드립니다. "
        f"임직원 여러분의 많은 관심과 적극적인 참여 부탁드립니다. 감사합니다."
    )
    return {
        "layout": "OUTRO_EVENT",
        "type": "OUTRO",
        "title": f"{m_clean} 위생교육 이벤트 안내",
        "subtitle": f"마지막으로 {m_clean} 위생교육 이벤트 안내입니다!",
        "event_month": m_clean,
        "event_ranks": r_clean,
        "event_prize_amount": p_amt,
        "event_prize_desc": p_item,
        "content": content_summary,
        "script": script_text
    }

def draw_event_row_icon(draw, icon_type, cx, cy):
    """이벤트 페이지.pptx의 3가지 좌측 아이콘(클립보드 체크, 선물상자, 손+달러)을 깔끔한 벡터 도형으로 그립니다."""
    col = (100, 116, 139)
    bg = (248, 250, 252)
    if icon_type == "clipboard":
        draw.rounded_rectangle([(cx - 13, cy - 15), (cx + 13, cy + 17)], radius=3, fill=col)
        draw.rounded_rectangle([(cx - 6, cy - 19), (cx + 6, cy - 12)], radius=2, fill=(71, 85, 105))
        draw.line([(cx - 5, cy + 2), (cx - 1, cy + 7), (cx + 7, cy - 3)], fill=bg, width=3)
    elif icon_type == "gift":
        draw.rectangle([(cx - 14, cy - 2), (cx + 14, cy + 15)], fill=col)
        draw.rectangle([(cx - 16, cy - 9), (cx + 16, cy - 3)], fill=col)
        draw.ellipse([(cx - 11, cy - 17), (cx - 1, cy - 7)], outline=col, width=3)
        draw.ellipse([(cx + 1, cy - 17), (cx + 11, cy - 7)], outline=col, width=3)
        draw.line([(cx, cy - 9), (cx, cy + 15)], fill=bg, width=3)
        draw.line([(cx - 14, cy + 5), (cx + 14, cy + 5)], fill=bg, width=2)
    elif icon_type == "hand":
        draw.polygon([(cx - 16, cy + 8), (cx + 4, cy + 8), (cx + 15, cy + 2), (cx + 17, cy + 6), (cx + 5, cy + 14), (cx - 16, cy + 14)], fill=col)
        draw.ellipse([(cx - 8, cy - 17), (cx + 8, cy - 1)], fill=col)
        draw.line([(cx, cy - 14), (cx, cy - 4)], fill=bg, width=2)

# ==========================================
# [7] 다중 레이아웃 렌더링 엔진 (상단1+하단3 당부말씀 & 고정 이벤트 페이지 완벽 재현)
# ==========================================
def draw_background_grid(draw, width, height, theme_mode):
    if "10월" in theme_mode:
        draw.rectangle([(0, 0), (width, height)], fill=(244, 247, 251))
        for x in range(0, width, 25):
            draw.line([(x, 0), (x, height)], fill=(224, 232, 243), width=1)
        for y in range(0, height, 25):
            draw.line([(0, y), (width, y)], fill=(224, 232, 243), width=1)
        for x in range(0, width, 125):
            draw.line([(x, 0), (x, height)], fill=(203, 216, 235), width=1)
        for y in range(0, height, 125):
            draw.line([(0, y), (width, y)], fill=(203, 216, 235), width=1)
    else:
        draw.rectangle([(0, 0), (width, height)], fill=(9, 17, 30))
        for x in range(0, width, 40):
            draw.line([(x, 0), (x, height)], fill=(16, 30, 52), width=1)
        for y in range(0, height, 40):
            draw.line([(0, y), (width, y)], fill=(16, 30, 52), width=1)
        draw.rectangle([(18, 18), (width - 18, height - 18)], outline=(16, 185, 129), width=2)
        draw.rectangle([(24, 24), (width - 24, height - 24)], outline=(6, 95, 70), width=1)

def create_slide_image(slide_data, slide_idx, total_slides, output_path, font_config, theme_mode):
    if slide_data.get("custom_image_bytes"):
        img = Image.open(io.BytesIO(slide_data["custom_image_bytes"])).convert("RGB")
        img = img.resize((1280, 720), Image.Resampling.LANCZOS)
        img.save(output_path)
        return output_path

    width, height = 1280, 720
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    title_font_name = font_config.get("title_font", "연세제목체")
    body_font_name = font_config.get("body_font", "연세본문체")
    custom_paths = font_config.get("custom_paths", {})

    f_hero = load_pil_font(title_font_name, 52, is_bold=True, custom_font_paths=custom_paths)
    f_event_title = load_pil_font(title_font_name, 46, is_bold=True, custom_font_paths=custom_paths)
    f_quote = load_pil_font(title_font_name, 44, is_bold=True, custom_font_paths=custom_paths)
    f_title = load_pil_font(title_font_name, 34, is_bold=True, custom_font_paths=custom_paths)
    f_sub = load_pil_font(title_font_name, 24, is_bold=True, custom_font_paths=custom_paths)
    f_event_row = load_pil_font(title_font_name, 25, is_bold=True, custom_font_paths=custom_paths)
    f_card_title = load_pil_font(title_font_name, 20, is_bold=True, custom_font_paths=custom_paths)
    f_badge_dept = load_pil_font(title_font_name, 14, is_bold=True, custom_font_paths=custom_paths)
    f_badge_role = load_pil_font(title_font_name, 18, is_bold=True, custom_font_paths=custom_paths)
    f_body = load_pil_font(body_font_name, 18, is_bold=False, custom_font_paths=custom_paths)
    f_small = load_pil_font(body_font_name, 15, is_bold=False, custom_font_paths=custom_paths)
    f_metric = load_pil_font(title_font_name, 46, is_bold=True, custom_font_paths=custom_paths)

    layout = slide_data.get("layout", "CARDS_3COL")
    is_light = "10월" in theme_mode

    # ---------------------------------------------------------
    # [레이아웃 1] INTRO_1TOP_3BOT: '위생교육 전 말씀.pptx' 원본 완벽 재현
    # ---------------------------------------------------------
    if layout in ("INTRO_1TOP_3BOT", "INTRO_4GRID"):
        draw.rectangle([(0, 0), (width, height)], fill=(244, 246, 250))
        title_txt = slide_data.get("title", "월간 위생교육 전 당부말씀")
        draw.text((42, 24), title_txt, font=f_title, fill=(15, 23, 42))

        cards = slide_data.get("cards", [])
        top_card = cards[0] if len(cards) > 0 else {"header": "생산본부", "sub": "생산본부장", "body": ""}

        # 1) 상단 와이드 카드 (생산본부 / 생산본부장)
        tx1, ty1, tx2, ty2 = 40, 82, 1240, 265
        red_col = (220, 38, 38)
        draw.rounded_rectangle([(tx1 + 2, ty1 + 3), (tx2 + 2, ty2 + 3)], radius=14, fill=(215, 222, 232))
        draw.rounded_rectangle([(tx1, ty1), (tx2, ty2)], radius=14, fill=(255, 255, 255), outline=(218, 226, 236), width=2)
        draw.rounded_rectangle([(tx1, ty1), (tx1 + 10, ty2)], radius=5, fill=red_col)

        cx_top, cy_top, r_top = tx1 + 92, (ty1 + ty2) // 2, 56
        draw.ellipse([(cx_top - r_top, cy_top - r_top), (cx_top + r_top, cy_top + r_top)], fill=(255, 255, 255), outline=red_col, width=4)

        dept_top = top_card.get("header", "생산본부")
        role_top = top_card.get("sub", "생산본부장")
        b1 = draw.textbbox((0, 0), dept_top, font=f_badge_dept)
        b2 = draw.textbbox((0, 0), role_top, font=f_badge_role)
        draw.text((cx_top - (b1[2] - b1[0]) // 2, cy_top - 20), dept_top, font=f_badge_dept, fill=(100, 116, 139))
        draw.text((cx_top - (b2[2] - b2[0]) // 2, cy_top + 2), role_top, font=f_badge_role, fill=(15, 23, 42))

        draw.line([(tx1 + 172, ty1 + 28), (tx1 + 172, ty2 - 28)], fill=(226, 232, 240), width=2)
        draw.text((tx1 + 188, ty1 + 16), "“", font=f_quote, fill=(203, 213, 225))

        top_body_lines = wrap_text_by_pixel(top_card.get("body", ""), f_small, (tx2 - tx1) - 245, draw)
        t_y = ty1 + 30
        for line in top_body_lines[:5]:
            draw.text((tx1 + 228, t_y), line, font=f_small, fill=(30, 41, 59))
            t_y += 26

        # 2) 하단 3열 세로 카드 (유제품 / 상온 / 품질안전)
        bottom_defaults = [
            ("유제품생산부문", "유제품생산부문장"),
            ("상온제품생산부문", "상온제품생산부문장"),
            ("품질안전부문", "품질안전부문장")
        ]
        navy_col = (15, 62, 124)
        card_w = 384
        gap_x = 24
        by1, by2 = 285, 692

        for i in range(3):
            bx1 = 40 + i * (card_w + gap_x)
            bx2 = bx1 + card_w
            def_dept, def_role = bottom_defaults[i]
            c_data = cards[i + 1] if (i + 1) < len(cards) else {}
            dept_str = c_data.get("header", def_dept)
            role_str = c_data.get("sub", def_role)
            body_str = c_data.get("body", "")

            draw.rounded_rectangle([(bx1 + 2, by1 + 3), (bx2 + 2, by2 + 3)], radius=14, fill=(215, 222, 232))
            draw.rounded_rectangle([(bx1, by1), (bx2, by2)], radius=14, fill=(255, 255, 255), outline=(218, 226, 236), width=2)
            draw.rounded_rectangle([(bx1, by1), (bx1 + 9, by2)], radius=5, fill=navy_col)

            cx_b, cy_b, r_b = bx1 + 52, by1 + 48, 30
            draw.ellipse([(cx_b - r_b, cy_b - r_b), (cx_b + r_b, cy_b + r_b)], fill=(255, 255, 255), outline=navy_col, width=3)
            short_tag = "유제품" if "유제품" in dept_str else ("상온" if "상온" in dept_str else "품질")
            bb_tag = draw.textbbox((0, 0), short_tag, font=f_badge_dept)
            draw.text((cx_b - (bb_tag[2] - bb_tag[0]) // 2, cy_b - 8), short_tag, font=f_badge_dept, fill=navy_col)

            draw.text((bx1 + 94, by1 + 24), dept_str, font=f_badge_dept, fill=(100, 116, 139))
            draw.text((bx1 + 94, by1 + 44), role_str, font=f_card_title, fill=(15, 23, 42))
            draw.text((bx2 - 46, by1 + 16), "“", font=f_quote, fill=(203, 213, 225))
            draw.line([(bx1 + 22, by1 + 88), (bx2 - 22, by1 + 88)], fill=(235, 240, 246), width=1)

            b_lines = wrap_text_by_pixel(body_str, f_small, card_w - 46, draw)
            cy_text = by1 + 102
            for line in b_lines[:11]:
                if cy_text > by2 - 24:
                    break
                draw.text((bx1 + 24, cy_text), line, font=f_small, fill=(45, 55, 72))
                cy_text += 25

        img.save(output_path)
        return output_path

    # ---------------------------------------------------------
    # [레이아웃 2] OUTRO_EVENT: '이벤트 페이지.pptx' 원본 100% 완벽 재현
    # (모눈 배경 + 이중 테두리 프레임 + 3단 회색 바 + 빨간색 순번/금액 하이라이트)
    # ---------------------------------------------------------
    if layout == "OUTRO_EVENT":
        # 1) 밝은 모눈종이 배경
        draw.rectangle([(0, 0), (width, height)], fill=(250, 252, 255))
        for x in range(0, width, 32):
            draw.line([(x, 0), (x, height)], fill=(235, 240, 247), width=1)
        for y in range(0, height, 32):
            draw.line([(0, y), (width, y)], fill=(235, 240, 247), width=1)

        # 2) 외곽 이중 다크 네이비 프레임
        draw.rectangle([(34, 28), (width - 34, height - 28)], outline=(15, 23, 42), width=3)
        draw.rectangle([(41, 35), (width - 41, height - 35)], outline=(15, 23, 42), width=1)

        # 3) 중앙 화이트 카드 + 우측/하단 입체 그림자
        ix1, iy1, ix2, iy2 = 160, 86, 1120, 634
        draw.rectangle([(ix1 + 12, iy1 + 12), (ix2 + 12, iy2 + 12)], fill=(226, 232, 240))
        draw.rectangle([(ix1, iy1), (ix2, iy2)], fill=(255, 255, 255), outline=(15, 23, 42), width=2)

        ev_month = slide_data.get("event_month", "11월")
        ev_ranks = slide_data.get("event_ranks", "11번째, 41번째, 71번째")
        ev_prize_amt = slide_data.get("event_prize_amount", "10,000원")
        ev_prize_desc = slide_data.get("event_prize_desc", "상당의 커피 기프티콘 증정")

        # 4) 상단 메인 타이틀 & 서브타이틀 중앙 정렬
        main_t = slide_data.get("title", f"{ev_month} 위생교육 이벤트 안내")
        sub_t = slide_data.get("subtitle", f"마지막으로 {ev_month} 위생교육 이벤트 안내입니다!")

        bt = draw.textbbox((0, 0), main_t, font=f_event_title)
        draw.text(((width - (bt[2] - bt[0])) // 2, 138), main_t, font=f_event_title, fill=(15, 23, 42))

        bs = draw.textbbox((0, 0), sub_t, font=f_card_title)
        draw.text(((width - (bs[2] - bs[0])) // 2, 212), sub_t, font=f_card_title, fill=(100, 116, 139))

        # 5) 3단 가로 안내 박스 (좌측 다크 포인트 바 + 아이콘 + 부분 컬러 텍스트)
        bx1, bx2 = 240, 1040
        bar_h = 74
        row_ys = [278, 384, 490]
        icons = ["clipboard", "gift", "hand"]

        row_segments = [
            [
                (f"{ev_month}은 만점자 중 순번  ", (15, 23, 42)),
                (ev_ranks, (220, 38, 38)),
                ("  분에게", (15, 23, 42))
            ],
            [
                (ev_prize_amt, (220, 38, 38)),
                (f"  {ev_prize_desc}", (15, 23, 42))
            ],
            [
                ("많은 관심과 적극적인 참여 부탁드립니다.", (15, 23, 42))
            ]
        ]

        for r_idx, ry1 in enumerate(row_segments):
            y_top = row_ys[r_idx]
            y_bot = y_top + bar_h
            # 연회색 라운드 박스 + 좌측 짙은 네이비 포인트 캡
            draw.rounded_rectangle([(bx1, y_top), (bx2, y_bot)], radius=10, fill=(248, 250, 252))
            draw.rounded_rectangle([(bx1, y_top), (bx1 + 8, y_bot)], radius=4, fill=(15, 23, 42))

            # 좌측 벡터 아이콘
            draw_event_row_icon(draw, icons[r_idx], bx1 + 52, y_top + bar_h // 2)

            # 세그먼트별 컬러 구분 렌더링 (검은색 + 빨간색 강조)
            cur_x = bx1 + 98
            text_y = y_top + 22
            for seg_txt, seg_col in ry1:
                draw.text((cur_x, text_y), seg_txt, font=f_event_row, fill=seg_col)
                seg_box = draw.textbbox((0, 0), seg_txt, font=f_event_row)
                cur_x += (seg_box[2] - seg_box[0])

        img.save(output_path)
        return output_path

    # 기본 배경 그리드 렌더링 (본론 및 표지용)
    draw_background_grid(draw, width, height, theme_mode)
    text_main_col = (15, 23, 42) if is_light else (248, 250, 252)
    text_sub_col = (51, 65, 85) if is_light else (203, 213, 225)
    card_bg_col = (255, 255, 255) if is_light else (15, 28, 48)
    card_border_col = (30, 64, 175) if is_light else (16, 185, 129)

    # ---------------------------------------------------------
    # [레이아웃 3] COVER: 표지 스타일
    # ---------------------------------------------------------
    if layout == "COVER":
        box_x1, box_y1, box_x2, box_y2 = 90, 120, 1190, 600
        draw.rounded_rectangle([(box_x1, box_y1), (box_x2, box_y2)], radius=18, fill=card_bg_col, outline=card_border_col, width=4)
        draw.rectangle([(box_x1, box_y1), (box_x1 + 22, box_y2)], fill=card_border_col)

        badge_txt = "YONSEI DAIRY FOOD SAFETY EDUCATION"
        draw.text((box_x1 + 60, box_y1 + 55), badge_txt, font=f_card_title, fill=(2, 132, 199) if is_light else (52, 211, 153))

        main_title = slide_data.get("title", "정기 위생교육")
        t_lines = wrap_text_by_pixel(main_title, f_hero, 980, draw)
        ty = box_y1 + 115
        for tl in t_lines[:2]:
            draw.text((box_x1 + 60, ty), tl, font=f_hero, fill=text_main_col)
            ty += 68

        sub_txt = slide_data.get("subtitle", "") or slide_data.get("content", "")
        s_lines = wrap_text_by_pixel(sub_txt, f_sub, 980, draw)
        sy = ty + 20
        for sl in s_lines[:2]:
            draw.text((box_x1 + 60, sy), sl, font=f_sub, fill=text_sub_col)
            sy += 38

        draw.line([(box_x1 + 60, box_y2 - 95), (box_x2 - 60, box_y2 - 95)], fill=card_border_col, width=2)
        footer_info = f"품질안전부문 식품안전팀 곽정혁  |  {datetime.date.today().strftime('%Y년 %m월')}"
        draw.text((box_x1 + 60, box_y2 - 68), footer_info, font=f_card_title, fill=text_main_col)
        img.save(output_path)
        return output_path

    # 공통 상단 타이틀 바 (일반 본론 슬라이드용)
    header_bg = (30, 58, 138) if is_light else (15, 28, 48)
    draw.rectangle([(40, 32), (1240, 102)], fill=header_bg, outline=card_border_col, width=2)
    draw.rectangle([(40, 32), (54, 102)], fill=(56, 189, 248) if is_light else (16, 185, 129))
    draw.text((72, 48), slide_data.get("title", f"슬라이드 {slide_idx + 1}"), font=f_title, fill=(255, 255, 255))

    # ---------------------------------------------------------
    # [레이아웃 4] TABLE_MATRIX: 표 매트릭스 스타일
    # ---------------------------------------------------------
    if layout == "TABLE_MATRIX" and slide_data.get("table_headers") and slide_data.get("table_rows"):
        headers = slide_data["table_headers"]
        rows = slide_data["table_rows"]
        col_cnt = max(1, len(headers))
        table_x1, table_y1, table_x2, table_y2 = 45, 125, 1235, 655
        col_w = (table_x2 - table_x1) // col_cnt

        th_h = 58
        for c_i, h_text in enumerate(headers):
            cx1 = table_x1 + c_i * col_w
            cx2 = table_x2 if c_i == col_cnt - 1 else (cx1 + col_w)
            draw.rectangle([(cx1, table_y1), (cx2, table_y1 + th_h)], fill=(30, 64, 175) if is_light else (6, 78, 59), outline=card_border_col, width=2)
            draw.text((cx1 + 16, table_y1 + 15), str(h_text), font=f_card_title, fill=(255, 255, 255))

        row_cnt = max(1, len(rows))
        avail_h = (table_y2 - (table_y1 + th_h))
        row_h = avail_h // row_cnt
        for r_i, r_data in enumerate(rows):
            ry1 = table_y1 + th_h + r_i * row_h
            ry2 = table_y2 if r_i == row_cnt - 1 else (ry1 + row_h)
            row_bg = (255, 255, 255) if (is_light and r_i % 2 == 0) else ((238, 244, 252) if is_light else (17, 32, 56))
            for c_i in range(col_cnt):
                cx1 = table_x1 + c_i * col_w
                cx2 = table_x2 if c_i == col_cnt - 1 else (cx1 + col_w)
                cell_txt = str(r_data[c_i]) if c_i < len(r_data) else ""
                draw.rectangle([(cx1, ry1), (cx2, ry2)], fill=row_bg, outline=card_border_col, width=1)
                c_font = f_card_title if c_i == 0 else f_body
                c_lines = wrap_text_by_pixel(cell_txt, c_font, col_w - 28, draw)
                cy = ry1 + 14
                for cl in c_lines:
                    if cy > ry2 - 24:
                        break
                    draw.text((cx1 + 14, cy), cl, font=c_font, fill=text_main_col if c_i == 0 else text_sub_col)
                    cy += 27

    # ---------------------------------------------------------
    # [레이아웃 5] SPLIT_VISUAL: 좌측 도면/사진 + 우측 설명 카드
    # ---------------------------------------------------------
    elif layout == "SPLIT_VISUAL" and slide_data.get("visual_bytes"):
        vx1, vy1, vx2, vy2 = 45, 125, 615, 650
        draw.rounded_rectangle([(vx1, vy1), (vx2, vy2)], radius=12, fill=card_bg_col, outline=card_border_col, width=3)
        try:
            v_img = Image.open(io.BytesIO(slide_data["visual_bytes"])).convert("RGB")
            v_img.thumbnail((vx2 - vx1 - 20, vy2 - vy1 - 20), Image.Resampling.LANCZOS)
            paste_x = vx1 + ((vx2 - vx1) - v_img.width) // 2
            paste_y = vy1 + ((vy2 - vy1) - v_img.height) // 2
            img.paste(v_img, (paste_x, paste_y))
        except Exception:
            pass

        cards = slide_data.get("cards", [])
        if not cards:
            cards = [{"header": "핵심 관리 기준", "body": slide_data.get("content", "")}]
        rx1, rx2 = 640, 1235
        c_cnt = len(cards[:3])
        gap = 16
        card_h = (525 - gap * (c_cnt - 1)) // max(1, c_cnt)
        for i, c_item in enumerate(cards[:3]):
            cy1 = 125 + i * (card_h + gap)
            cy2 = cy1 + card_h
            draw.rounded_rectangle([(rx1, cy1), (rx2, cy2)], radius=10, fill=card_bg_col, outline=card_border_col, width=2)
            draw.rectangle([(rx1, cy1), (rx1 + 12, cy2)], fill=card_border_col)
            draw.text((rx1 + 26, cy1 + 16), c_item.get("header", ""), font=f_card_title, fill=card_border_col if is_light else (52, 211, 153))
            b_lines = wrap_text_by_pixel(c_item.get("body", ""), f_body, (rx2 - rx1) - 45, draw)
            by = cy1 + 52
            for bl in b_lines:
                if by > cy2 - 26:
                    break
                draw.text((rx1 + 26, by), bl, font=f_body, fill=text_sub_col)
                by += 29

    # ---------------------------------------------------------
    # [레이아웃 6] CARDS_3COL / TOC_STEP / SELF_AUDIT_2X2
    # ---------------------------------------------------------
    else:
        cards = slide_data.get("cards", [])
        if not cards:
            raw_lines = [l.strip() for l in slide_data.get("content", "").split("\n") if l.strip()]
            cards = [{"header": f"핵심 포인트 {i+1}", "body": line, "metric": ""} for i, line in enumerate(raw_lines[:4])]

        c_cnt = min(len(cards), 5)
        if c_cnt == 4 and layout == "SELF_AUDIT_2X2":
            grid_coords = [
                (45, 125, 625, 380),
                (655, 125, 1235, 380),
                (45, 395, 625, 650),
                (655, 395, 1235, 650)
            ]
            for i in range(4):
                gx1, gy1, gx2, gy2 = grid_coords[i]
                c_item = cards[i]
                draw.rounded_rectangle([(gx1, gy1), (gx2, gy2)], radius=12, fill=card_bg_col, outline=card_border_col, width=2)
                draw.rectangle([(gx1, gy1), (gx2, gy1 + 46)], fill=header_bg)
                draw.text((gx1 + 20, gy1 + 10), c_item.get("header", ""), font=f_card_title, fill=(255, 255, 255))
                b_lines = wrap_text_by_pixel(c_item.get("body", ""), f_body, (gx2 - gx1) - 40, draw)
                by = gy1 + 62
                for bl in b_lines:
                    if by > gy2 - 26:
                        break
                    draw.text((gx1 + 20, by), bl, font=f_body, fill=text_sub_col)
                    by += 29
        else:
            total_w = 1190
            gap = 18
            card_w = (total_w - gap * (max(1, c_cnt) - 1)) // max(1, c_cnt)
            for i, c_item in enumerate(cards[:c_cnt]):
                cx1 = 45 + i * (card_w + gap)
                cx2 = cx1 + card_w
                cy1, cy2 = 125, 650
                draw.rounded_rectangle([(cx1, cy1), (cx2, cy2)], radius=12, fill=card_bg_col, outline=card_border_col, width=3)
                draw.rectangle([(cx1, cy1), (cx2, cy1 + 58)], fill=header_bg)
                h_lines = wrap_text_by_pixel(c_item.get("header", f"STEP {i+1}"), f_card_title, card_w - 24, draw)
                if h_lines:
                    draw.text((cx1 + 14, cy1 + 15), h_lines[0], font=f_card_title, fill=(255, 255, 255))

                by = cy1 + 80
                metric_str = str(c_item.get("metric", "")).strip()
                if metric_str:
                    m_lines = wrap_text_by_pixel(metric_str, f_metric, card_w - 24, draw)
                    for ml in m_lines[:2]:
                        draw.text((cx1 + 18, by), ml, font=f_metric, fill=(239, 68, 68))
                        by += 56
                    draw.line([(cx1 + 18, by + 6), (cx2 - 18, by + 6)], fill=card_border_col, width=2)
                    by += 22

                b_lines = wrap_text_by_pixel(c_item.get("body", ""), f_body, card_w - 32, draw)
                for bl in b_lines:
                    if by > cy2 - 30:
                        break
                    draw.text((cx1 + 16, by), bl, font=f_body, fill=text_sub_col)
                    by += 30

    bottom_banner = slide_data.get("bottom_summary", "")
    if bottom_banner:
        draw.rounded_rectangle([(45, 662), (1100, 706)], radius=8, fill=(254, 242, 242) if is_light else (6, 78, 59), outline=(239, 68, 68) if is_light else (16, 185, 129), width=2)
        draw.text((65, 672), f"핵심 원칙: {bottom_banner}", font=f_body, fill=(185, 28, 28) if is_light else (255, 255, 255))

    draw.text((width - 155, height - 38), f"Page {slide_idx + 1} / {total_slides}", font=f_small, fill=text_sub_col)
    img.save(output_path)
    return output_path

# ==========================================
# [8] 편집 가능한 PPTX 생성기 (상단1+하단3 당부말씀 & 고정 이벤트 도형 완벽 구현)
# ==========================================
def generate_editable_pptx(slides_data, font_config, theme_mode):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    title_font_choice = font_config.get("title_font", "연세제목체")
    body_font_choice = font_config.get("body_font", "연세본문체")
    pptx_title_font = FONT_SEARCH_MAP.get(title_font_choice, FONT_SEARCH_MAP["맑은고딕"])["pptx_name"]
    pptx_body_font = FONT_SEARCH_MAP.get(body_font_choice, FONT_SEARCH_MAP["맑은고딕"])["pptx_name"]
    is_light = "10월" in theme_mode

    for idx, slide_data in enumerate(slides_data):
        slide = prs.slides.add_slide(blank_layout)
        layout = slide_data.get("layout", "CARDS_3COL")

        if slide_data.get("custom_image_bytes"):
            slide.shapes.add_picture(io.BytesIO(slide_data["custom_image_bytes"]), Inches(0), Inches(0), width=Inches(13.333), height=Inches(7.5))

        elif layout in ("INTRO_1TOP_3BOT", "INTRO_4GRID"):
            background = slide.background
            fill = background.fill
            fill.solid()
            fill.fore_color.rgb = RGBColor(244, 246, 250)

            tx_t = slide.shapes.add_textbox(Inches(0.45), Inches(0.22), Inches(12.0), Inches(0.6))
            p_t = tx_t.text_frame.paragraphs[0]
            p_t.text = slide_data.get("title", "월간 위생교육 전 당부말씀")
            p_t.font.name = pptx_title_font
            p_t.font.size = Pt(24)
            p_t.font.bold = True
            p_t.font.color.rgb = RGBColor(15, 23, 42)

            cards = slide_data.get("cards", [])
            top_c = cards[0] if len(cards) > 0 else {"header": "생산본부", "sub": "생산본부장", "body": ""}

            top_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.45), Inches(0.88), Inches(12.43), Inches(1.92))
            top_box.fill.solid()
            top_box.fill.fore_color.rgb = RGBColor(255, 255, 255)
            top_box.line.color.rgb = RGBColor(220, 38, 38)
            top_box.line.width = Pt(2.2)

            oval_top = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.72), Inches(1.22), Inches(1.25), Inches(1.25))
            oval_top.fill.solid()
            oval_top.fill.fore_color.rgb = RGBColor(255, 255, 255)
            oval_top.line.color.rgb = RGBColor(220, 38, 38)
            oval_top.line.width = Pt(3)
            tf_ot = oval_top.text_frame
            tf_ot.word_wrap = True
            p_ot1 = tf_ot.paragraphs[0]
            p_ot1.text = f"{top_c.get('header', '생산본부')}\n{top_c.get('sub', '생산본부장')}"
            p_ot1.alignment = PP_ALIGN.CENTER
            p_ot1.font.name = pptx_title_font
            p_ot1.font.size = Pt(12)
            p_ot1.font.bold = True
            p_ot1.font.color.rgb = RGBColor(15, 23, 42)

            tb_top = slide.shapes.add_textbox(Inches(2.2), Inches(1.02), Inches(10.45), Inches(1.65))
            tf_top = tb_top.text_frame
            tf_top.word_wrap = True
            p_tb = tf_top.paragraphs[0]
            p_tb.text = top_c.get("body", "")
            p_tb.font.name = pptx_body_font
            p_tb.font.size = Pt(13)
            p_tb.font.color.rgb = RGBColor(30, 41, 59)

            bot_defaults = [
                ("유제품생산부문", "유제품생산부문장"),
                ("상온제품생산부문", "상온제품생산부문장"),
                ("품질안전부문", "품질안전부문장")
            ]
            for i in range(3):
                bx = Inches(0.45 + i * 4.21)
                by = Inches(3.02)
                bw = Inches(4.01)
                bh = Inches(4.15)
                def_d, def_r = bot_defaults[i]
                c_item = cards[i + 1] if (i + 1) < len(cards) else {}

                b_card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, bx, by, bw, bh)
                b_card.fill.solid()
                b_card.fill.fore_color.rgb = RGBColor(255, 255, 255)
                b_card.line.color.rgb = RGBColor(15, 62, 124)
                b_card.line.width = Pt(2)

                oval_b = slide.shapes.add_shape(MSO_SHAPE.OVAL, bx + Inches(0.22), by + Inches(0.22), Inches(0.65), Inches(0.65))
                oval_b.fill.solid()
                oval_b.fill.fore_color.rgb = RGBColor(255, 255, 255)
                oval_b.line.color.rgb = RGBColor(15, 62, 124)
                oval_b.line.width = Pt(2.2)

                tb_bh = slide.shapes.add_textbox(bx + Inches(0.95), by + Inches(0.20), Inches(2.9), Inches(0.7))
                tf_bh = tb_bh.text_frame
                tf_bh.word_wrap = True
                p_bh = tf_bh.paragraphs[0]
                p_bh.text = f"{c_item.get('header', def_d)}\n{c_item.get('sub', def_r)}"
                p_bh.font.name = pptx_title_font
                p_bh.font.size = Pt(14)
                p_bh.font.bold = True
                p_bh.font.color.rgb = RGBColor(15, 23, 42)

                tb_bb = slide.shapes.add_textbox(bx + Inches(0.20), by + Inches(1.02), Inches(3.6), Inches(2.95))
                tf_bb = tb_bb.text_frame
                tf_bb.word_wrap = True
                p_bb = tf_bb.paragraphs[0]
                p_bb.text = c_item.get("body", "")
                p_bb.font.name = pptx_body_font
                p_bb.font.size = Pt(12)
                p_bb.font.color.rgb = RGBColor(45, 55, 72)

        elif layout == "OUTRO_EVENT":
            # '이벤트 페이지.pptx' 원본과 동일한 편집 가능 PPTX 슬라이드 도형 생성
            background = slide.background
            fill = background.fill
            fill.solid()
            fill.fore_color.rgb = RGBColor(250, 252, 255)

            outer_frame = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.38), Inches(0.32), Inches(12.57), Inches(6.86))
            outer_frame.fill.background()
            outer_frame.line.color.rgb = RGBColor(15, 23, 42)
            outer_frame.line.width = Pt(2.5)

            inner_card = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.65), Inches(0.90), Inches(10.03), Inches(5.70))
            inner_card.fill.solid()
            inner_card.fill.fore_color.rgb = RGBColor(255, 255, 255)
            inner_card.line.color.rgb = RGBColor(15, 23, 42)
            inner_card.line.width = Pt(1.5)

            ev_month = slide_data.get("event_month", "11월")
            ev_ranks = slide_data.get("event_ranks", "11번째, 41번째, 71번째")
            ev_prize_amt = slide_data.get("event_prize_amount", "10,000원")
            ev_prize_desc = slide_data.get("event_prize_desc", "상당의 커피 기프티콘 증정")

            tb_title = slide.shapes.add_textbox(Inches(2.0), Inches(1.30), Inches(9.33), Inches(1.3))
            tf_t = tb_title.text_frame
            tf_t.word_wrap = True
            p1 = tf_t.paragraphs[0]
            p1.text = slide_data.get("title", f"{ev_month} 위생교육 이벤트 안내")
            p1.alignment = PP_ALIGN.CENTER
            p1.font.name = pptx_title_font
            p1.font.size = Pt(36)
            p1.font.bold = True
            p1.font.color.rgb = RGBColor(15, 23, 42)

            p2 = tf_t.add_paragraph()
            p2.text = slide_data.get("subtitle", f"마지막으로 {ev_month} 위생교육 이벤트 안내입니다!")
            p2.alignment = PP_ALIGN.CENTER
            p2.font.name = pptx_body_font
            p2.font.size = Pt(16)
            p2.font.bold = True
            p2.font.color.rgb = RGBColor(100, 116, 139)

            row_texts = [
                f"☑️  {ev_month}은 만점자 중 순번  {ev_ranks}  분에게",
                f"🎁  {ev_prize_amt}  {ev_prize_desc}",
                "🤲  많은 관심과 적극적인 참여 부탁드립니다."
            ]
            for r_i, r_str in enumerate(row_texts):
                ry = Inches(2.90 + r_i * 1.10)
                bar_shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(2.50), ry, Inches(8.33), Inches(0.82))
                bar_shape.fill.solid()
                bar_shape.fill.fore_color.rgb = RGBColor(248, 250, 252)
                bar_shape.line.color.rgb = RGBColor(226, 232, 240)
                bar_shape.line.width = Pt(1)

                tf_b = bar_shape.text_frame
                tf_b.word_wrap = True
                p_b = tf_b.paragraphs[0]
                p_b.text = r_str
                p_b.font.name = pptx_title_font
                p_b.font.size = Pt(20)
                p_b.font.bold = True
                p_b.font.color.rgb = RGBColor(15, 23, 42)

        else:
            background = slide.background
            fill = background.fill
            fill.solid()
            fill.fore_color.rgb = RGBColor(244, 247, 251) if is_light else RGBColor(9, 17, 30)

            tx_title = slide.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(12.1), Inches(1.0))
            tf_title = tx_title.text_frame
            tf_title.word_wrap = True
            p_title = tf_title.paragraphs[0]
            p_title.text = slide_data.get("title", f"슬라이드 {idx + 1}")
            p_title.font.name = pptx_title_font
            p_title.font.size = Pt(28)
            p_title.font.bold = True
            p_title.font.color.rgb = RGBColor(30, 58, 138) if is_light else RGBColor(52, 211, 153)

            cards = slide_data.get("cards", [])
            if cards:
                c_cnt = min(len(cards), 4)
                card_w = 12.0 / max(1, c_cnt)
                for c_i, c_item in enumerate(cards[:c_cnt]):
                    bx = Inches(0.6 + c_i * card_w)
                    by = Inches(1.6)
                    bw = Inches(card_w - 0.15)
                    bh = Inches(5.2)
                    tb = slide.shapes.add_textbox(bx, by, bw, bh)
                    tf = tb.text_frame
                    tf.word_wrap = True

                    p_h = tf.paragraphs[0]
                    p_h.text = f"■ {c_item.get('header', '')} {c_item.get('sub', '')}".strip()
                    p_h.font.name = pptx_title_font
                    p_h.font.size = Pt(18)
                    p_h.font.bold = True
                    p_h.font.color.rgb = RGBColor(2, 132, 199) if is_light else RGBColor(56, 189, 248)

                    if c_item.get("metric"):
                        p_m = tf.add_paragraph()
                        p_m.text = str(c_item["metric"])
                        p_m.font.name = pptx_title_font
                        p_m.font.size = Pt(26)
                        p_m.font.bold = True
                        p_m.font.color.rgb = RGBColor(239, 68, 68)

                    for line in str(c_item.get("body", "")).split("\n"):
                        p_b = tf.add_paragraph()
                        p_b.text = line.strip()
                        p_b.font.name = pptx_body_font
                        p_b.font.size = Pt(15)
                        p_b.font.color.rgb = RGBColor(30, 41, 59) if is_light else RGBColor(241, 245, 249)
            else:
                tx_body = slide.shapes.add_textbox(Inches(0.6), Inches(1.6), Inches(12.1), Inches(5.2))
                tf_body = tx_body.text_frame
                tf_body.word_wrap = True
                for i, line in enumerate(str(slide_data.get("content", "")).split("\n")):
                    p = tf_body.paragraphs[0] if i == 0 else tf_body.add_paragraph()
                    p.text = line.strip()
                    p.font.name = pptx_body_font
                    p.font.size = Pt(18)
                    p.font.color.rgb = RGBColor(30, 41, 59) if is_light else RGBColor(241, 245, 249)

        notes_slide = slide.notes_slide
        notes_slide.notes_text_frame.text = f"[AI 성우 내레이션 대본]\n{slide_data.get('script', '')}"

    pptx_io = io.BytesIO()
    prs.save(pptx_io)
    pptx_io.seek(0)
    return pptx_io.getvalue()

def create_tts_audio(script_text, output_path):
    clean_script = re.sub(r'[\*\#\[\]\(\)]', ' ', str(script_text)).strip()
    if not clean_script:
        clean_script = "다음 교육 내용을 확인해 주시기 바랍니다."
    tts = gTTS(text=clean_script, lang='ko', slow=False)
    tts.save(output_path)
    return output_path

def media_worker(idx, slide_data, total_slides, temp_dir, font_config, theme_mode):
    img_path = os.path.join(temp_dir, f"slide_{idx:03d}.png")
    audio_path = os.path.join(temp_dir, f"audio_{idx:03d}.mp3")
    create_slide_image(slide_data, idx, total_slides, img_path, font_config, theme_mode)
    create_tts_audio(slide_data.get("script", ""), audio_path)
    return idx, img_path, audio_path

# ==========================================
# [9] Gemini AI 스토리보드 기획 엔진 (이벤트 슬라이드는 파이썬 고정 빌더로 분리)
# ==========================================
def generate_ai_storyboard(model_name, topic, edu_month, body_slide_count, intro_config, uploaded_gemini_files, extra_source_text):
    model = genai.GenerativeModel(model_name)
    generation_config = genai.types.GenerationConfig(temperature=0.1, max_output_tokens=65536)

    if intro_config["mode"] != "인트로 생략":
        intro_instruction = f"""
        [슬라이드 1 (고정 오프닝: '월간 위생교육 전 당부말씀' - 상단 생산본부장 1단 와이드 + 하단 3개 부문장 3열 구조)]:
        - 반드시 첫 번째 슬라이드는 "layout": "INTRO_1TOP_3BOT", "title": "월간 위생교육 전 당부말씀" 으로 설정하십시오.
        - "cards" 배열에 정확히 아래 순서대로 4명의 리더 당부 말씀을 작성하십시오 (사용자가 입력한 원문이 있으면 그대로 반영):
          1) 상단 메인 와이드 박스: {{"header": "생산본부", "sub": "생산본부장", "body": "..."}}
          2) 하단 좌측 세로 카드: {{"header": "유제품생산부문", "sub": "유제품생산부문장", "body": "..."}}
          3) 하단 중앙 세로 카드: {{"header": "상온제품생산부문", "sub": "상온제품생산부문장", "body": "..."}}
          4) 하단 우측 세로 카드: {{"header": "품질안전부문", "sub": "품질안전부문장", "body": "..."}}
        - "script"에는 교육 시작에 앞서 생산본부장님과 유제품·상온·품질안전 3개 부문장님의 당부 말씀 핵심 요약을 현장 작업자들에게 정중히 브리핑하는 내레이션 대본을 작성하십시오.
        - [사용자 입력 당부 말씀 원문]:\n{intro_config.get("custom_text", "")}
        """
    else:
        intro_instruction = "[인트로 생략]: INTRO_1TOP_3BOT 슬라이드를 생략하고 표지(COVER)부터 시작하십시오."

    system_prompt = f"""
    당신은 연세유업 아산공장 식품안전팀(담당자: 곽정혁)의 수석 위생교육 슬라이드 디자이너이자 HACCP 교육 전문가입니다.
    9월('이물혼입 예방관리') 및 10월('영업장 관리') 교육 자료와 100% 동일한 수준의 고밀도 시각 레이아웃과 구어체 더빙 대본을 설계해야 합니다.
    임의의 추측을 배제하고 오직 법적·과학적 팩트와 업로드된 영상/문서의 내용만 정확히 반영하십시오.
    (※ 주의: 마지막 이벤트 안내 슬라이드는 파이썬 시스템이 고정 템플릿으로 자동 생성하므로, 당신은 본론 마지막 '자가 진단 체크리스트(SELF_AUDIT_2X2)'까지만 생성하십시오.)

    [교육 월]: {edu_month}
    [교육 핵심 주제]: {topic}
    [본론 슬라이드 장수]: 표지(COVER) + 목차(TOC_STEP) + 본론 상세({body_slide_count}장)

    {intro_instruction}

    [필수 포함 슬라이드 구조 및 레이아웃("layout") 배정 규칙]:
    1. "INTRO_1TOP_3BOT": 상단 생산본부장(와이드) + 하단 3개 부문장(3열) 당부 말씀 (1p)
    2. "COVER": 교육 표지 (제목: "{edu_month} 정기 위생교육 : {topic}", 부제목 포함)
    3. "TOC_STEP": 전체 교육 목차 (3~5단계 아키텍처 카드)
    4. 본론 슬라이드 ({body_slide_count}장)는 내용 성격에 따라 아래 레이아웃을 골고루 배정하십시오:
       - "CARDS_3COL": 법적 수치 기준(예: 3mm 이상, 10.0mg/kg 미만, 220/540 Lux 등)이나 2~4단 핵심 원칙 비교 시 사용. 수치 강조가 필요하면 각 카드의 "metric" 필드에 "3mm 이상" 처럼 짧은 핵심 수치를 넣으십시오.
       - "TABLE_MATRIX": 행정처분 기준(1차/2차/3차), 구역 설정(분리/구획/구분), 4M+1E 원인 분석, 해충 분류표 등 표 형태가 적합할 때 사용 ("table_headers"와 "table_rows" 2차원 배열 필수 작성).
       - "SPLIT_VISUAL": 작업장 동선도, 배수로 U트랩, 탈의실 45도 빗면, 또는 실제 이물 사례 사진이 들어가야 할 때 사용.
         * 만약 업로드된 참고 동영상(MP4)에서 특정 장면을 캡처해 넣어야 한다면 해당 초(초 단위 숫자)를 "video_timestamp_sec"에 적으십시오.
         * 도면/인포그래픽 일러스트가 필요하다면 "visual_prompt"에 영문으로 도면 묘사 프롬프트를 적으십시오.
       - "SELF_AUDIT_2X2": 본론 마지막 장에는 반드시 10월 14p 스타일의 '핵심 자가 진단 체크리스트 (Self-Audit Matrix)' 4개 영역 카드를 구성하십시오 (각 줄 앞에 '[] ' 표시).

    [JSON 출력 포맷 절대 규칙]:
    먼저 <pre_calc> 태그 안에 슬라이드별 레이아웃 배정 이유와 팩트 체크 내역을 적고, 이어서 <json_output> 태그 안에 순수 JSON 배열만 출력하십시오.

    <pre_calc>
    사전 설계 요약
    </pre_calc>
    <json_output>
    [
      {{
        "layout": "INTRO_1TOP_3BOT",
        "type": "INTRO",
        "title": "월간 위생교육 전 당부말씀",
        "subtitle": "",
        "content": "",
        "cards": [
          {{"header": "생산본부", "sub": "생산본부장", "metric": "", "body": "당부 말씀 전문..."}},
          {{"header": "유제품생산부문", "sub": "유제품생산부문장", "metric": "", "body": "당부 말씀 전문..."}},
          {{"header": "상온제품생산부문", "sub": "상온제품생산부문장", "metric": "", "body": "당부 말씀 전문..."}},
          {{"header": "품질안전부문", "sub": "품질안전부문장", "metric": "", "body": "당부 말씀 전문..."}}
        ],
        "table_headers": [],
        "table_rows": [],
        "bottom_summary": "",
        "visual_prompt": "",
        "video_timestamp_sec": null,
        "script": "안녕하십니까. 본격적인 위생교육에 앞서 생산본부장님과 각 부문장님들의 당부 말씀을 전해드립니다..."
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

def generate_scripts_for_existing_pdf(model_name, pdf_pages_info, edu_month, topic):
    model = genai.GenerativeModel(model_name)
    slides_text_dump = "\n\n".join([f"[슬라이드 {p['page_num']} 텍스트]:\n{p['text']}" for p in pdf_pages_info])
    prompt = f"""
    다음은 연세유업 아산공장 '{edu_month} 정기 위생교육 ({topic})' 슬라이드 총 {len(pdf_pages_info)}장의 텍스트입니다.
    슬라이드 1번부터 {len(pdf_pages_info)}번까지 단 한 장도 누락하지 말고, 각 슬라이드 화면을 보며 성우가 자연스럽게 설명할 구어체 내레이션 대본(script)을 작성하십시오.
    만약 1번 슬라이드가 '부문장님 당부말씀'이라면 생산본부장님과 각 부문장님의 핵심 메시지를 요약해서 정중히 전달하는 대본으로 작성하십시오.

    반드시 아래 순수 JSON 배열 포맷으로만 출력하십시오 (개수: 정확히 {len(pdf_pages_info)}개):
    [
      {{"page_num": 1, "title": "슬라이드 1 요약 제목", "script": "1번 슬라이드 성우 내레이션 대본"}},
      ...
    ]

    [추출된 슬라이드 텍스트 전체]:
    {slides_text_dump}
    """
    resp = model.generate_content(prompt)
    clean_j = re.sub(r'^```json\s*|\s*```$', '', resp.text.strip(), flags=re.MULTILINE).strip()
    return json.loads(clean_j)

# ==========================================
# [10] 메인 애플리케이션 UI
# ==========================================
def main():
    st.markdown(custom_theme_css, unsafe_allow_html=True)

    for key in ["slides_data", "pre_calc_log", "final_video_bytes", "final_pptx_bytes", "custom_font_paths", "uploaded_video_local_path"]:
        if key not in st.session_state:
            st.session_state[key] = {} if key == "custom_font_paths" else None

    st.markdown("<h1>🎬 연세유업 AI 위생교육 스튜디오 <span style='font-size:0.45em; color:#64748b;'>(상단1+하단3 당부말씀 & 고정 이벤트 양식 완비)</span></h1>", unsafe_allow_html=True)
    st.markdown("생산본부장님(상단 와이드)·3부문장님(하단 3열) 당부 말씀과 월/순번 자동 연동 고정 이벤트 페이지, 본론 슬라이드 및 성우 더빙 MP4 영상까지 원스톱으로 제작합니다.")
    st.markdown("<hr>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ 엔진 및 디자인 테마 설정")
        selected_model = st.selectbox(
            "Gemini 두뇌 모델",
            ["gemini-3.8-flash", "gemini-omni-1.2-flash", "gemini-2.5-flash", "gemini-4-argon"],
            index=0
        )
        theme_mode = st.radio(
            "본론 슬라이드 디자인 테마 선택",
            ["10월 스타일 (라이트 블루프린트 모눈 테마)", "9월 스타일 (다크 사이버네틱 네온 테마)"],
            index=0
        )
        enable_ai_drawing = st.checkbox("SPLIT_VISUAL 슬라이드에 AI 도면 자동 생성", value=True)

        st.markdown("---")
        st.header("🔤 슬라이드 글씨체 설정")
        FONT_OPTIONS = ["연세제목체", "연세본문체", "맑은고딕"]
        selected_title_font = st.selectbox("슬라이드 제목 글씨체", FONT_OPTIONS, index=0)
        selected_body_font = st.selectbox("슬라이드 본문 글씨체", FONT_OPTIONS, index=1)

        with st.expander("📁 폰트 파일(.ttf/.otf) 직접 업로드"):
            up_yt = st.file_uploader("연세제목체", type=["ttf", "otf"], key="u_yt")
            up_yb = st.file_uploader("연세본문체", type=["ttf", "otf"], key="u_yb")
            up_mg = st.file_uploader("맑은고딕", type=["ttf", "otf"], key="u_mg")
            for f_label, f_obj in [("연세제목체", up_yt), ("연세본문체", up_yb), ("맑은고딕", up_mg)]:
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
        "1. 교육 기획 & 고정 양식(당부말씀/이벤트) 설정",
        "2. 슬라이드 디자인 미리보기 & 대본 검수",
        "3. PPTX 문서 & MP4 교육 영상 렌더링"
    ])

    # ==========================================
    # 탭 1: 기획 및 고정 양식 설정
    # ==========================================
    with tab1:
        workflow_mode = st.radio(
            "제작 워크플로우 선택:",
            [
                "모드 A: 처음부터 AI로 슬라이드 디자인(당부말씀 + 9월·10월 고밀도 본론 + 고정 이벤트) + 대본 자동 생성",
                "모드 B: 기존 PDF(제미나이 노트북 생성본 등)를 그대로 올리고 당부말씀/이벤트 합치기 + AI 음성 영상화"
            ],
            index=0
        )
        st.markdown("---")

        col_m1, col_m2 = st.columns([1, 2.2])
        with col_m1:
            edu_month_num = st.number_input("교육 기준 월 (숫자만 선택 시 순번 자동 계산):", min_value=1, max_value=12, value=11, step=1)
            edu_month = f"{edu_month_num}월"
        with col_m2:
            topic = st.text_input("교육 핵심 주제:", value="작업장 이물혼입 예방 및 영업장 위생관리")

        # --- 고정 양식 1: 오프닝 4대 부문장 당부 말씀 (상단 생산본부장 1단 + 하단 3부문장 3열) ---
        st.markdown("### 👔 고정 양식 1 (오프닝): 월간 위생교육 전 당부말씀 (`위생교육 전 말씀.pptx` 동일 레이아웃)")
        intro_mode = st.radio(
            "당부 말씀 슬라이드 반영:",
            ["당부 말씀 슬라이드 포함 (상단 생산본부장 + 하단 3부문장)", "당부 말씀 슬라이드 생략 (업로드할 PDF에 이미 포함된 경우)"],
            horizontal=True
        )

        intro_custom_text = ""
        if "포함" in intro_mode:
            leader_pptx = st.file_uploader("📄 '위생교육 전 말씀.pptx' 파일 업로드 시 4분 말씀 자동 채우기 (선택):", type=["pptx"], key="ldr_pptx")
            parsed_leaders = parse_leader_pptx(leader_pptx) if leader_pptx else {}

            default_prod = parsed_leaders.get("생산본부") or f"{edu_month}은 아침저녁으로 일교차가 커지면서 외부 해충이나 이물질이 따뜻한 실내 작업장으로 유입되기 쉬워 각별한 주의가 필요한 시기입니다. 우리 연세유업 아산공장의 흔들림 없는 식품 안전은 결국 현장에 계신 여러분의 빈틈없는 이물 관리와 밀폐 수칙 준수에서 시작됩니다. 현장에 정착된 철저한 HACCP 실행력을 바탕으로, 발생할 수 있는 위해요소를 사전에 100% 차단하는 든든한 방어선 역할을 해 주시길 당부 드립니다."
            default_dairy = parsed_leaders.get("유제품생산부문") or "아침저녁으로 일교차가 심한 환절기에 들어왔습니다. 유제품은 미생물과 주변 환경변화에 매우 민감한 특성을 가지고 있습니다. 환절기에는 주간 고온다습한 공기가 야간 냉각된 공장 천정, 덕트, 배관표면등에 결로 발생하여 곰팡이 발생이 쉬운 조건이 됩니다. 작업장 환경 청소와 철저한 건조 관리에 많은 관심이 필요합니다. 그리고, 일일 건강 모니터링을 각자 하셔서 호흡기 질환 확산을 방지할 수 있도록 해주시기 바랍니다."
            default_ambient = parsed_leaders.get("상온제품생산부문") or "상온제품의 안전성을 책임지는 핵심은 멸균 온도 관리와 여과망 점검 등 철저한 CCP 모니터링에 있습니다. 각 공정의 한계기준을 다시 한번 명확히 숙지하고 이탈 시 즉각적인 조치가 이루어질 수 있도록 만전을 기해 주십시오. 더불어, 여과망 파손이나 사소한 이물 혼입도 제품의 치명적인 결함으로 이어질 수 있음을 명심하고 철저한 작업장 밀폐 등 빈틈없는 이물 관리를 당부드립니다."
            default_qc = parsed_leaders.get("품질안전부문") or f"식품 제조에 있어 '이물 혼입'은 우리가 쌓아온 고객의 신뢰를 한순간에 무너뜨리는 가장 치명적인 사고입니다. {edu_month}은 이물유입 변수가 많아지는 시기로 HACCP 시스템에 입각한 철저하고 선제적인 예방 관리가 필수적입니다. 품질과 안전에는 어떠한 타협도 있을 수 없음을 명심하고, 철저한 밀폐와 꼼꼼한 개인위생 등 기본에 충실한 원칙 중심의 현장관리를 실천해 주실 것을 당부드립니다."

            ldr_prod = st.text_area("🔴 [상단 와이드] 생산본부 (생산본부장)", value=default_prod, height=90)
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                ldr_dairy = st.text_area("🔵 [하단 좌측] 유제품생산부문 (유제품생산부문장)", value=default_dairy, height=145)
            with col_b2:
                ldr_ambient = st.text_area("🔵 [하단 중앙] 상온제품생산부문 (상온제품생산부문장)", value=default_ambient, height=145)
            with col_b3:
                ldr_qc = st.text_area("🔵 [하단 우측] 품질안전부문 (품질안전부문장)", value=default_qc, height=145)

            intro_custom_text = (
                f"[생산본부장]: {ldr_prod}\n"
                f"[유제품생산부문장]: {ldr_dairy}\n"
                f"[상온제품생산부문장]: {ldr_ambient}\n"
                f"[품질안전부문장]: {ldr_qc}"
            )

        st.markdown("---")

        # --- 고정 양식 2: 클로징 이벤트 안내 페이지 (`이벤트 페이지.pptx` 100% 동일 고정 템플릿) ---
        st.markdown("### 🎁 고정 양식 2 (클로징): 월간 위생교육 이벤트 안내 (`이벤트 페이지.pptx` 동일 레이아웃)")
        include_outro = st.checkbox("마지막 슬라이드에 고정 양식 이벤트 안내 페이지 자동 추가", value=True)
        auto_r1, auto_r2, auto_r3 = edu_month_num, edu_month_num + 30, edu_month_num + 60
        default_ranks_str = f"{auto_r1}번째, {auto_r2}번째, {auto_r3}번째"

        if include_outro:
            col_ev1, col_ev2, col_ev3 = st.columns([1.4, 1, 1.4])
            with col_ev1:
                event_ranks_input = st.text_input(
                    "당첨 순번 (붉은색 강조 표기 - 월 변경 시 자동 연동):",
                    value=default_ranks_str,
                    help="예: 10월 -> 10번째, 40번째, 70번째 / 11월 -> 11번째, 41번째, 71번째"
                )
            with col_ev2:
                event_prize_amt_input = st.text_input("경품 금액 (붉은색 강조):", value="10,000원")
            with col_ev3:
                event_prize_desc_input = st.text_input("경품 내용:", value="상당의 커피 기프티콘 증정")
            st.caption(f"📌 **자동 완성 미리보기**: `{edu_month} 위생교육 이벤트 안내` | `{edu_month}은 만점자 중 순번 {event_ranks_input} 분에게` | `{event_prize_amt_input} {event_prize_desc_input}` | `많은 관심과 적극적인 참여 부탁드립니다.`")

        st.markdown("---")

        if "모드 A" in workflow_mode:
            st.markdown("### 📘 본론 교육 자료 설정")
            body_slide_count = st.slider("생성할 본론 슬라이드 장수 (표지/목차/당부말씀/이벤트 제외):", 4, 14, 8)
            uploaded_sources = st.file_uploader(
                "참고할 원본 교육 영상(MP4) 또는 문서(PDF, PPTX) 업로드 (선택사항 - 복수 가능):",
                type=["mp4", "pdf", "pptx"],
                accept_multiple_files=True
            )
            extra_source_text = st.text_area("추가 반영할 핵심 내용 / 법적 수치 기준 (선택):", height=80, placeholder="예: 보고 대상 이물 기준(3mm 이상 유리·플라스틱, 쇳가루 10.0mg/kg 미만), 조도 540 Lux 이상 등")

            if st.button("🚀 고정 당부말씀 + 고밀도 본론 + 고정 이벤트 슬라이드 일괄 생성", type="primary", use_container_width=True):
                with st.spinner("원본 분석 및 슬라이드/대본을 생성 중입니다..."):
                    gemini_files = []
                    combined_text = extra_source_text
                    st.session_state["uploaded_video_local_path"] = None

                    for f in uploaded_sources:
                        ext = os.path.splitext(f.name)[1].lower()
                        if ext == ".pptx":
                            combined_text += f"\n\n[PPTX 추출 내용]:\n{extract_text_from_pptx(f)}"
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
                        selected_model, topic, edu_month, body_slide_count, intro_cfg, gemini_files, combined_text
                    )
                    if err:
                        st.error(err)
                    else:
                        # AI가 혹시 자체적으로 OUTRO_EVENT를 중복 생성했다면 제거 후 파이썬 고정 빌더로 정확히 교체
                        slides = [s for s in slides if s.get("layout") != "OUTRO_EVENT" and s.get("type") != "OUTRO"]

                        if slides and "포함" in intro_mode and slides[0].get("layout") in ("INTRO_1TOP_3BOT", "INTRO_4GRID"):
                            slides[0]["layout"] = "INTRO_1TOP_3BOT"
                            slides[0]["cards"] = [
                                {"header": "생산본부", "sub": "생산본부장", "body": ldr_prod},
                                {"header": "유제품생산부문", "sub": "유제품생산부문장", "body": ldr_dairy},
                                {"header": "상온제품생산부문", "sub": "상온제품생산부문장", "body": ldr_ambient},
                                {"header": "품질안전부문", "sub": "품질안전부문장", "body": ldr_qc}
                            ]

                        for s in slides:
                            if s.get("layout") == "SPLIT_VISUAL" and not s.get("visual_bytes"):
                                ts = s.get("video_timestamp_sec")
                                if ts is not None and st.session_state.get("uploaded_video_local_path"):
                                    frame_b = capture_frame_from_video(st.session_state["uploaded_video_local_path"], ts)
                                    if frame_b:
                                        s["visual_bytes"] = frame_b
                                if not s.get("visual_bytes") and enable_ai_drawing and s.get("visual_prompt"):
                                    s["visual_bytes"] = generate_ai_illustration(s.get("visual_prompt"), theme_mode)

                        # 마지막 슬라이드에 고정 양식 이벤트 페이지 추가
                        if include_outro:
                            slides.append(build_fixed_event_slide(edu_month, event_ranks_input, event_prize_amt_input, event_prize_desc_input))

                        st.session_state["slides_data"] = slides
                        st.session_state["pre_calc_log"] = pre_log
                        st.session_state["final_video_bytes"] = None
                        st.session_state["final_pptx_bytes"] = None
                        st.success(f"총 {len(slides)}장의 슬라이드 기획이 완료되었습니다! [2. 슬라이드 디자인 미리보기 & 대본 검수] 탭으로 이동하세요.")

        else:
            st.markdown("### 📄 기존 교육 자료 PDF 업로드 (원본 슬라이드 디자인 100% 보존)")
            pdf_file = st.file_uploader("제미나이 노트북 또는 파워포인트에서 저장한 PDF 파일을 업로드하세요:", type=["pdf"])
            if st.button("🚀 PDF 전 페이지 이미지 분리 + 고정 당부말씀/이벤트 합치기 + AI 성우 대본 생성", type="primary", use_container_width=True):
                if not pdf_file:
                    st.warning("PDF 파일을 먼저 업로드해주세요.")
                else:
                    with st.spinner("PDF의 모든 페이지를 고해상도 슬라이드로 추출하고 맞춤형 더빙 대본을 작성 중입니다..."):
                        pages_info = extract_pages_from_pdf(pdf_file.getvalue())
                        scripts_list = generate_scripts_for_existing_pdf(selected_model, pages_info, edu_month, topic)

                        built_slides = []
                        if "포함" in intro_mode:
                            built_slides.append({
                                "layout": "INTRO_1TOP_3BOT",
                                "type": "INTRO",
                                "title": "월간 위생교육 전 당부말씀",
                                "cards": [
                                    {"header": "생산본부", "sub": "생산본부장", "body": ldr_prod},
                                    {"header": "유제품생산부문", "sub": "유제품생산부문장", "body": ldr_dairy},
                                    {"header": "상온제품생산부문", "sub": "상온제품생산부문장", "body": ldr_ambient},
                                    {"header": "품질안전부문", "sub": "품질안전부문장", "body": ldr_qc}
                                ],
                                "script": f"안녕하십니까. {edu_month} 정기 위생교육에 앞서 생산본부장님과 각 부문장님들의 당부 말씀을 전해드립니다. 먼저 생산본부장님께서는 {ldr_prod} 이어서 유제품생산부문장님께서는 {ldr_dairy} 상온제품생산부문장님께서는 {ldr_ambient} 마지막으로 품질안전부문장님께서는 {ldr_qc}"
                            })

                        for idx, p_info in enumerate(pages_info):
                            matched_script = scripts_list[idx].get("script", "") if idx < len(scripts_list) else p_info["text"]
                            matched_title = scripts_list[idx].get("title", f"PDF 슬라이드 {p_info['page_num']}") if idx < len(scripts_list) else f"슬라이드 {p_info['page_num']}"
                            built_slides.append({
                                "layout": "CUSTOM_PDF_PAGE",
                                "type": "BODY",
                                "title": matched_title,
                                "content": p_info["text"],
                                "custom_image_bytes": p_info["image_bytes"],
                                "script": matched_script
                            })

                        if include_outro:
                            built_slides.append(build_fixed_event_slide(edu_month, event_ranks_input, event_prize_amt_input, event_prize_desc_input))

                        st.session_state["slides_data"] = built_slides
                        st.session_state["final_video_bytes"] = None
                        st.session_state["final_pptx_bytes"] = None
                        st.success(f"총 {len(built_slides)}장의 슬라이드 이미지 및 성우 더빙 대본 세팅이 완료되었습니다! [2. 슬라이드 디자인 미리보기 & 대본 검수] 탭을 확인하세요.")

    # ==========================================
    # 탭 2: 실시간 슬라이드 렌더링 미리보기 및 대본 수정
    # ==========================================
    with tab2:
        if not st.session_state.get("slides_data"):
            st.info("먼저 [1. 교육 기획 & 고정 양식 설정] 탭에서 슬라이드를 생성해 주십시오.")
        else:
            st.markdown("### 🖥️ 슬라이드별 실제 화면 미리보기 및 대본 편집")
            total_s = len(st.session_state["slides_data"])
            updated_slides = []

            with tempfile.TemporaryDirectory() as preview_dir:
                for idx, slide in enumerate(st.session_state["slides_data"]):
                    is_edge_slide = (idx == 0 or idx == total_s - 1)
                    with st.expander(f"슬라이드 {idx + 1} [{slide.get('layout', 'SLIDE')}] : {slide.get('title', '')}", expanded=is_edge_slide):
                        col_p1, col_p2 = st.columns([1.15, 1])
                        with col_p2:
                            new_title = st.text_input("슬라이드 제목", value=slide.get("title", ""), key=f"t_{idx}")
                            if slide.get("layout") == "OUTRO_EVENT":
                                slide["event_month"] = st.text_input("이벤트 월 표기", value=slide.get("event_month", "11월"), key=f"evm_{idx}")
                                slide["event_ranks"] = st.text_input("당첨 순번 표기 (빨간색 강조)", value=slide.get("event_ranks", "11번째, 41번째, 71번째"), key=f"evr_{idx}")
                                slide["event_prize_amount"] = st.text_input("경품 금액 표기 (빨간색 강조)", value=slide.get("event_prize_amount", "10,000원"), key=f"evp_{idx}")
                                slide["event_prize_desc"] = st.text_input("경품 설명 표기", value=slide.get("event_prize_desc", "상당의 커피 기프티콘 증정"), key=f"evd_{idx}")
                                slide["subtitle"] = f"마지막으로 {slide['event_month']} 위생교육 이벤트 안내입니다!"
                            new_script = st.text_area("성우 내레이션 대본 (오디오 합성용)", value=slide.get("script", ""), height=120, key=f"s_{idx}")
                            replace_img = st.file_uploader("이 슬라이드 전체 화면을 다른 이미지로 교체 (선택)", type=["png", "jpg", "jpeg"], key=f"r_{idx}")
                            if replace_img:
                                slide["custom_image_bytes"] = replace_img.getvalue()

                        slide["title"] = new_title
                        slide["script"] = new_script

                        with col_p1:
                            preview_img_path = os.path.join(preview_dir, f"preview_{idx}.png")
                            create_slide_image(slide, idx, total_s, preview_img_path, font_config, theme_mode)
                            st.image(preview_img_path, caption=f"슬라이드 {idx + 1} 실제 렌더링 화면", use_container_width=True)

                        updated_slides.append(slide)

            st.session_state["slides_data"] = updated_slides

    # ==========================================
    # 탭 3: PPTX 문서 및 MP4 교육 영상 병렬 렌더링
    # ==========================================
    with tab3:
        if not st.session_state.get("slides_data"):
            st.info("먼저 [1. 교육 기획 & 고정 양식 설정] 탭에서 슬라이드를 생성해 주십시오.")
        else:
            slides_to_render = st.session_state["slides_data"]
            st.markdown(f"### 🎬 최종 결과물 다운로드 (총 {len(slides_to_render)}장)")

            col_b1, col_b2 = st.columns(2)
            with col_b1:
                if st.button("📊 1단계: 파워포인트(.pptx) 파일 생성", use_container_width=True):
                    with st.spinner("PPTX 파일을 생성 중입니다..."):
                        st.session_state["final_pptx_bytes"] = generate_editable_pptx(slides_to_render, font_config, theme_mode)
                        st.success("PPTX 파일 생성 완료! (당부말씀·고정이벤트 도형 및 발표자 대본 포함)")

                if st.session_state.get("final_pptx_bytes"):
                    st.download_button(
                        label="📥 파워포인트(.pptx) 다운로드",
                        data=st.session_state["final_pptx_bytes"],
                        file_name=f"연세유업_{edu_month}_위생교육_{datetime.date.today().strftime('%Y%m%d')}.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        use_container_width=True
                    )

            with col_b2:
                if st.button("🎥 2단계: 성우 음성 합성 및 MP4 교육 영상 렌더링", type="primary", use_container_width=True):
                    t_start = time.time()
                    with tempfile.TemporaryDirectory() as temp_dir:
                        total_cnt = len(slides_to_render)
                        p_bar = st.progress(0, text="병렬 엔진 가동: 고해상도 슬라이드 및 성우 오디오 동시 생성 중...")

                        media_results = [None] * total_cnt
                        with ThreadPoolExecutor(max_workers=5) as executor:
                            futures = [
                                executor.submit(media_worker, idx, s_data, total_cnt, temp_dir, font_config, theme_mode)
                                for idx, s_data in enumerate(slides_to_render)
                            ]
                            done_cnt = 0
                            for future in as_completed(futures):
                                idx, img_p, aud_p = future.result()
                                media_results[idx] = (img_p, aud_p)
                                done_cnt += 1
                                p_bar.progress(done_cnt / (total_cnt + 1), text=f"({done_cnt}/{total_cnt}) 슬라이드 및 음성 생성 완료...")

                        p_bar.progress(total_cnt / (total_cnt + 1), text="최종 MP4 교육 영상 병합 중...")
                        v_clips, a_clips = [], []
                        try:
                            for img_path, audio_path in media_results:
                                a_clip = AudioFileClip(audio_path)
                                i_clip = ImageClip(img_path).set_duration(a_clip.duration + 0.5)
                                v_clips.append(i_clip.set_audio(a_clip))
                                a_clips.append(a_clip)

                            final_video = concatenate_videoclips(v_clips, method="compose")
                            out_mp4 = os.path.join(temp_dir, "yonsei_edu.mp4")
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
                                try: c.close()
                                except: pass
                            for a in a_clips:
                                try: a.close()
                                except: pass

                        p_bar.empty()
                        st.success(f"🎉 MP4 교육 영상 제작 완료! (소요시간: {time.time() - t_start:.1f}초)")

            if st.session_state.get("final_video_bytes"):
                st.markdown("---")
                st.video(st.session_state["final_video_bytes"])
                st.download_button(
                    label="📥 완성된 MP4 교육 영상 다운로드",
                    data=st.session_state["final_video_bytes"],
                    file_name=f"연세유업_{edu_month}_위생교육영상_{datetime.date.today().strftime('%Y%m%d')}.mp4",
                    mime="video/mp4",
                    type="primary",
                    use_container_width=True
                )

if __name__ == "__main__":
    if check_password():
        main()
