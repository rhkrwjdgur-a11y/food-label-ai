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
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ==========================================
# [1] UI 레이아웃 및 네트워크 방어 설정 (최상단 필수)
# ==========================================
st.set_page_config(page_title="연세유업 AI 교육 영상 스튜디오", layout="wide")
socket.setdefaulttimeout(600)

# ==========================================
# [2] 보안 접속 설정 (기존 시스템 동일 비밀번호 적용)
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

DEFAULT_MODEL = "gemini-3.8-flash"

# ==========================================
# [4] 커스텀 CSS (모던 다크 슬레이트 테마)
# ==========================================
custom_theme_css = """
<style>
@media screen {
    html, body, [class*="css"] {
        color: #1e293b !important;
        font-family: 'Pretendard', -apple-system, sans-serif !important;
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
# [5] 유틸리티 및 파일 파싱 함수
# ==========================================
def get_korean_font(size, is_bold=False):
    """
    로컬 폴더, 리눅스 서버(fonts-nanum), 윈도우 환경을 모두 탐색하여
    한글이 절대 깨지지 않도록 폰트를 로드하는 방어 함수입니다.
    """
    font_candidates = [
        "NanumGothicBold.ttf" if is_bold else "NanumGothic.ttf",
        "malgunbd.ttf" if is_bold else "malgun.ttf",
        "Pretendard-Bold.ttf" if is_bold else "Pretendard-Regular.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf" if is_bold else "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "C:/Windows/Fonts/malgunbd.ttf" if is_bold else "C:/Windows/Fonts/malgun.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc"
    ]
    for path in font_candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()

def extract_text_from_pptx(file_obj):
    """업로드된 PPTX 파일에서 슬라이드별 텍스트를 100% 추출합니다."""
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

def wrap_text_by_pixel(text, font, max_width, draw):
    """긴 문장이 슬라이드 화면 밖으로 나가지 않도록 픽셀 단위로 계산해 자동 줄바꿈합니다."""
    lines = []
    for paragraph in text.split('\n'):
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
            continue
        words = paragraph.split(' ')
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip() if current_line else word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_width = bbox[2] - bbox[0]
            if line_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
    return lines

# ==========================================
# [6] 미디어 생성 코어 함수 (Slide Image, PPTX, TTS, Video)
# ==========================================
def create_slide_image(slide_data, slide_idx, total_slides, output_path):
    """
    만약 사용자가 직접 업로드한 슬라이드 원본 이미지가 있으면 그것을 그대로 사용하고,
    없으면 연세유업 표준 와이드(16:9) 디자인으로 고해상도 슬라이드 이미지를 렌더링합니다.
    """
    if slide_data.get("custom_image_bytes"):
        img = Image.open(io.BytesIO(slide_data["custom_image_bytes"])).convert("RGB")
        img = img.resize((1280, 720), Image.Resampling.LANCZOS)
        img.save(output_path)
        return output_path

    width, height = 1280, 720
    img = Image.new('RGB', (width, height), color=(15, 23, 42)) # 다크 네이비 배경
    draw = ImageDraw.Draw(img)

    # 상단 헤더 바 및 하단 푸터 바 디자인
    draw.rectangle([(0, 0), (width, 95)], fill=(30, 41, 59))
    draw.rectangle([(0, 95), (width, 100)], fill=(56, 189, 248)) # 스카이 블루 포인트 라인
    draw.rectangle([(0, height - 50), (width, height)], fill=(15, 23, 42))

    font_badge = get_korean_font(20, is_bold=True)
    font_title = get_korean_font(42, is_bold=True)
    font_content = get_korean_font(30, is_bold=False)
    font_footer = get_korean_font(18, is_bold=False)

    # 슬라이드 타입 뱃지 표시
    slide_type = slide_data.get("type", "BODY")
    badge_map = {
        "INTRO": ("LEADER MESSAGE | 부문장 당부 말씀", (2, 132, 199)),
        "BODY": ("SAFETY & QUALITY | 본 교육 과정", (15, 118, 110)),
        "OUTRO": ("EVENT & NOTICE | 교육 이벤트 및 공지", (109, 40, 217))
    }
    badge_text, badge_color = badge_map.get(slide_type, ("EDUCATION", (71, 85, 105)))

    draw.rounded_rectangle([(60, 25), (460, 70)], radius=8, fill=badge_color)
    draw.text((80, 34), badge_text, font=font_badge, fill=(255, 255, 255))

    # 우측 상단 기관 로고 텍스트
    draw.text((width - 360, 36), "연세유업 아산공장 식품안전팀", font=font_badge, fill=(148, 163, 184))

    # 메인 제목 렌더링
    title_text = slide_data.get("title", f"슬라이드 {slide_idx + 1}")
    draw.text((70, 130), title_text, font=font_title, fill=(255, 255, 255))

    # 본문 컨텐츠 박스 배경
    draw.rounded_rectangle([(60, 210), (width - 60, height - 70)], radius=12, fill=(30, 41, 59), outline=(51, 65, 85), width=2)

    # 본문 텍스트 자동 줄바꿈 및 렌더링
    content_text = slide_data.get("content", "")
    wrapped_lines = wrap_text_by_pixel(content_text, font_content, width - 180, draw)

    y_text = 240
    for line in wrapped_lines:
        if y_text > height - 110:
            break
        draw.text((90, y_text), line, font=font_content, fill=(226, 232, 240))
        y_text += 48

    # 하단 페이지 번호
    draw.text((width - 150, height - 38), f"Page {slide_idx + 1} / {total_slides}", font=font_footer, fill=(100, 116, 139))

    img.save(output_path)
    return output_path

def generate_editable_pptx(slides_data):
    """
    생성된 슬라이드 기획안을 바탕으로 공장장님 보고 및 실무 수정이 가능한
    실제 파워포인트(.pptx) 파일을 생성하여 메모리 버퍼로 반환합니다.
    """
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    for idx, slide_data in enumerate(slides_data):
        slide = prs.slides.add_slide(blank_layout)

        # 배경색 설정
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = RGBColor(15, 23, 42)

        # 제목 텍스트 상자
        tx_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.6), Inches(11.7), Inches(1.2))
        tf_title = tx_title.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = f"[{slide_data.get('type', 'SLIDE')}] {slide_data.get('title', '')}"
        p_title.font.size = Pt(32)
        p_title.font.bold = True
        p_title.font.color.rgb = RGBColor(56, 189, 248)

        # 본문 텍스트 상자
        tx_body = slide.shapes.add_textbox(Inches(0.8), Inches(2.0), Inches(11.7), Inches(4.8))
        tf_body = tx_body.text_frame
        tf_body.word_wrap = True

        lines = slide_data.get("content", "").split('\n')
        for i, line in enumerate(lines):
            p = tf_body.paragraphs[0] if i == 0 else tf_body.add_paragraph()
            p.text = line.strip()
            p.font.size = Pt(20)
            p.font.color.rgb = RGBColor(241, 245, 249)
            p.space_after = Pt(14)

        # 슬라이드 노트(발표자 노트)에 음성 더빙 대본 자동 삽입
        notes_slide = slide.notes_slide
        text_frame = notes_slide.notes_text_frame
        text_frame.text = f"[AI 성우 내레이션 대본]\n{slide_data.get('script', '')}"

    pptx_io = io.BytesIO()
    prs.save(pptx_io)
    pptx_io.seek(0)
    return pptx_io.getvalue()

def create_tts_audio(script_text, output_path):
    """대본 텍스트를 읽어 MP3 오디오 파일로 변환합니다."""
    clean_script = re.sub(r'[\*\#\[\]\(\)]', ' ', script_text).strip()
    if not clean_script:
        clean_script = "다음 화면을 확인해 주시기 바랍니다."
    tts = gTTS(text=clean_script, lang='ko', slow=False)
    tts.save(output_path)
    return output_path

def media_worker(idx, slide_data, total_slides, temp_dir):
    """ThreadPoolExecutor에서 병렬로 이미지와 오디오를 동시에 찍어내는 워커 함수입니다."""
    img_path = os.path.join(temp_dir, f"slide_{idx:03d}.png")
    audio_path = os.path.join(temp_dir, f"audio_{idx:03d}.mp3")
    create_slide_image(slide_data, idx, total_slides, img_path)
    create_tts_audio(slide_data.get("script", ""), audio_path)
    return idx, img_path, audio_path

# ==========================================
# [7] Gemini AI 기획 엔진 (CoT + JSON 추출)
# ==========================================
def generate_ai_storyboard(model_name, topic, body_slide_count, intro_config, outro_config, uploaded_gemini_files, extra_source_text):
    model = genai.GenerativeModel(model_name)
    generation_config = genai.types.GenerationConfig(temperature=0.1, max_output_tokens=65536)

    # 1. 인트로(4대 부문장 당부 말씀) 지시문 조립
    if intro_config["mode"] == "AI 자동 생성 (4대 부문장 맞춤형)":
        intro_instruction = f"""
        [슬라이드 1 (고정 인트로 - 4대 부문장 당부 말씀)]:
        - 반드시 첫 번째 슬라이드("type": "INTRO")는 교육 시작에 앞서 당사 4명의 리더(공장장, 상온부문장, 유제품부문장, 품질안전부문장)가 '{topic}' 주제와 관련하여 현장 작업자들에게 전하는 핵심 당부 말씀으로 구성하십시오.
        - 화면 텍스트(content)에는 아래 4줄 포맷을 그대로 유지하여 핵심 요약을 적으십시오:
          • 공장장: (주제 관련 당부 1줄)
          • 상온부문장: (상온/멸균 제품 라인 특성에 맞춘 당부 1줄)
          • 유제품부문장: (우유/발효유 등 유가공 라인 특성에 맞춘 당부 1줄)
          • 품질안전부문장: (HACCP 및 식품안전 규격 준수 당부 1줄)
        - 내레이션 대본(script)은 이 4명의 리더 말씀을 자연스럽게 브리핑하며 교육의 중요성을 일깨우는 아나운서 톤으로 작성하십시오.
        """
    elif intro_config["mode"] == "확정된 PPT/텍스트 직접 반영 (공장장님 컨펌본)":
        intro_instruction = f"""
        [슬라이드 1 (고정 인트로 - 컨펌 완료본 반영)]:
        - 첫 번째 슬라이드("type": "INTRO")는 사용자가 입력한 아래의 [확정된 당부 말씀 원문]을 훼손하지 말고 화면 텍스트(content)에 깔끔하게 정리하여 배치하십시오.
        - 내레이션 대본(script)은 해당 원문을 바탕으로 각 부문장님의 말씀 요약을 정중하고 명확하게 전달하는 대본으로 작성하십시오.
        - [확정된 당부 말씀 원문]:\n{intro_config["custom_text"]}
        """
    else:
        intro_instruction = "[인트로 생략]: 별도의 부문장 당부 말씀 인트로 없이 곧바로 본론 교육 슬라이드(\"type\": \"BODY\")부터 시작하십시오."

    # 2. 아웃트로(이벤트 안내) 지시문 조립
    if outro_config["include"]:
        outro_instruction = f"""
        [마지막 슬라이드 (고정 아웃트로 - 이벤트 및 공지 안내)]:
        - 반드시 배열의 가장 마지막 슬라이드("type": "OUTRO")는 교육 종료 후 참여 이벤트/퀴즈 안내 화면으로 구성하십시오.
        - 아래 제공된 [이벤트 안내 내용]을 바탕으로 화면 텍스트(content)와 독려 내레이션 대본(script)을 작성하십시오.
        - [이벤트 안내 내용]:\n{outro_config["text"]}
        """
    else:
        outro_instruction = "[아웃트로 생략]: 별도의 이벤트 안내 슬라이드를 추가하지 마십시오."

    system_prompt = f"""
    당신은 대한민국 최고의 식품안전(HACCP) 및 품질관리(QC) 사내 교육 기획 전문가입니다.
    연세유업 아산공장 임직원을 대상으로 한 교육 영상 슬라이드(PPT 화면 텍스트)와 성우 더빙용 내레이션 대본을 설계해야 합니다.
    추측이나 거짓 정보를 배제하고 오직 명확한 법적/과학적 팩트와 제공된 원본 자료만을 기반으로 작성하십시오.

    [교육 주제]: {topic}
    [본론(BODY) 슬라이드 생성 장수]: 정확히 {body_slide_count}장 (인트로/아웃트로 제외)

    {intro_instruction}

    [본론 슬라이드 ("type": "BODY") 작성 원칙]:
    - 업로드된 영상/문서 자료가 있다면 화면 속의 표, 한계기준 수치(온도/시간/농도 등), 핵심 수칙을 100% 반영하여 {body_slide_count}장의 슬라이드로 논리적으로 분할하십시오.
    - 각 슬라이드의 화면 텍스트(content)는 작업자가 한눈에 보기 쉽게 개조식(• 기호 활용)으로 3~5줄 이내로 명확히 요약하십시오.
    - 각 슬라이드의 내레이션 대본(script)은 화면 텍스트를 그대로 기계적으로 읽지 말고, 실제 강사가 설명하듯 자연스러운 구어체 존댓말("~입니다", "~해주시기 바랍니다")로 상세히 풀어서 작성하십시오.

    {outro_instruction}

    [출력 형식 절대 규칙]:
    1. 먼저 <pre_calc> 와 </pre_calc> 태그 사이에 전체 슬라이드 구성 계획과 핵심 팩트 체크 과정을 요약하십시오.
    2. 그 다음, 반드시 <json_output> 과 </json_output> 태그 사이에 아래 포맷의 순수 JSON 배열만 출력하십시오. (생략 기호 '...' 절대 사용 금지)

    <pre_calc>
    여기에 사전 기획 및 팩트 검증 과정을 작성
    </pre_calc>
    <json_output>
    [
      {{
        "type": "INTRO",
        "title": "슬라이드 제목",
        "content": "• 첫 번째 줄\\n• 두 번째 줄\\n• 세 번째 줄",
        "script": "성우가 읽을 자연스러운 구어체 내레이션 대본 전체"
      }},
      {{
        "type": "BODY",
        "title": "본론 슬라이드 1 제목",
        "content": "• 핵심 내용 1\\n• 핵심 내용 2",
        "script": "본론 1 상세 설명 내레이션 대본"
      }}
    ]
    </json_output>
    """

    payload = []
    if uploaded_gemini_files:
        payload.extend(uploaded_gemini_files)
    if extra_source_text.strip():
        payload.append(f"[참고 원문/문서 데이터]:\n{extra_source_text}")
    payload.append(system_prompt)

    for attempt in range(3):
        try:
            response = model.generate_content(payload, generation_config=generation_config)
            raw_text = response.text

            pre_calc_match = re.search(r'<pre_calc>(.*?)</pre_calc>', raw_text, re.DOTALL)
            pre_calc_log = pre_calc_match.group(1).strip() if pre_calc_match else "사전 연산 로그 없음"

            json_match = re.search(r'<json_output>(.*?)</json_output>', raw_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                json_str = raw_text

            json_str = re.sub(r'^```json\s*', '', json_str)
            json_str = re.sub(r'^```\s*', '', json_str)
            json_str = re.sub(r'\s*```$', '', json_str).strip()

            slides_data = json.loads(json_str)
            return slides_data, pre_calc_log, None
        except Exception as e:
            if attempt < 2:
                time.sleep(4)
                continue
            return None, None, f"AI 기획안 생성 및 JSON 파싱 실패: {e}\n\n[원본 응답]:\n{raw_text if 'raw_text' in locals() else '응답 없음'}"

# ==========================================
# [8] 메인 애플리케이션 화면 구성
# ==========================================
def main():
    st.markdown(custom_theme_css, unsafe_allow_html=True)

    # 세션 스테이트 초기화
    for key in ["slides_data", "pre_calc_log", "final_video_bytes", "final_pptx_bytes", "intro_custom_img", "outro_custom_img"]:
        if key not in st.session_state:
            st.session_state[key] = None

    st.markdown("<h1>🎬 AI 교육 영상 & PPT 원스톱 자동화 스튜디오 <span style='font-size:0.45em; color:#64748b;'>(연세유업 품질안전부문 전용)</span></h1>", unsafe_allow_html=True)
    st.markdown("원본 교육 영상/문서 또는 주제만 입력하면 **[부문장님 당부 말씀 ➡️ 본 교육 PPT/대본 ➡️ 이벤트 안내 ➡️ MP4 영상 병합]**까지 한 번에 완성합니다.")
    st.markdown("<hr>", unsafe_allow_html=True)

    # --- 사이드바 설정 ---
    with st.sidebar:
        st.header("⚙️ 엔진 및 기본 설정")
        selected_model = st.selectbox(
            "사용할 Gemini 모델 선택",
            ["gemini-3.8-flash", "gemini-omni-1.2-flash", "gemini-2.5-flash", "gemini-4-argon"],
            index=0,
            help="일반 기획 및 영상 분석에는 고속 모델인 gemini-3.8-flash 또는 omni-1.2-flash를 권장합니다."
        )
        body_slide_count = st.slider("본론(핵심 교육) 슬라이드 장수", min_value=2, max_value=12, value=4, step=1)
        st.markdown("---")
        st.info("💡 **작동 순서 안내**\n1. **[탭 1]**에서 인트로(당부말씀), 본론 자료, 아웃트로(이벤트)를 설정하고 기획안을 생성합니다.\n2. **[탭 2]**에서 생성된 슬라이드 텍스트와 더빙 대본을 눈으로 확인 및 수정합니다.\n3. **[탭 3]**에서 버튼 하나로 PPTX 파일과 MP4 교육 영상을 렌더링하여 다운로드합니다.")

    tab1, tab2, tab3 = st.tabs([
        "1. 교육 자료 입력 및 AI 대본 기획",
        "2. 슬라이드 & 대본 검수/수정 (컨펌용)",
        "3. PPTX 및 MP4 교육 영상 렌더링"
    ])

    # ==========================================
    # 탭 1: 입력 및 기획안 생성
    # ==========================================
    with tab1:
        st.markdown("### Step 1. 교육 주제 및 본론 원본 자료 입력")
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            topic = st.text_input("이번 교육의 핵심 주제 (필수):", value="영업장 위생 관리 및 교차오염 방지 수칙")
            source_mode = st.radio(
                "본론 제작 방식 선택:",
                ["주제만 입력하여 AI 팩트 기반 자동 기획", "참고 영상(MP4) 또는 문서(PPTX/PDF) 업로드 기반 기획"],
                index=0
            )
        with col_t2:
            uploaded_source_files = []
            if "업로드" in source_mode:
                uploaded_source_files = st.file_uploader(
                    "참고할 원본 교육 영상(MP4) 또는 문서(PDF, PPTX) 업로드 (복수 가능):",
                    type=["mp4", "pdf", "pptx", "png", "jpg", "jpeg"],
                    accept_multiple_files=True
                )
            extra_source_text = st.text_area("추가 반영할 사내 기준/핵심 키워드 (선택사항):", height=90, placeholder="예: 작업장 온도 15℃ 이하 유지, 알코올 소독액 70% 사용 등 반드시 포함할 수치나 내용을 적어주세요.")

        st.markdown("---")

        # --- 인트로 설정 (4대 부문장 당부 말씀) ---
        st.markdown("### Step 2. 오프닝: 4대 부문장님 당부 말씀 설정 (Slide 1)")
        intro_mode = st.radio(
            "당부 말씀 구성 방식:",
            ["AI 자동 생성 (4대 부문장 맞춤형)", "확정된 PPT/텍스트 직접 반영 (공장장님 컨펌본)", "인트로 생략"],
            horizontal=True
        )

        intro_custom_text = ""
        if intro_mode == "확정된 PPT/텍스트 직접 반영 (공장장님 컨펌본)":
            col_i1, col_i2 = st.columns(2)
            with col_i1:
                intro_pptx_file = st.file_uploader("컨펌받은 당부 말씀 PPTX 파일 업로드 (자동 텍스트 추출):", type=["pptx"], key="intro_pptx")
                if intro_pptx_file:
                    extracted_intro = extract_text_from_pptx(intro_pptx_file)
                    st.success("PPTX 텍스트 추출 완료!")
                else:
                    extracted_intro = "• 공장장: 기본 안전 수칙 준수가 최우선입니다.\n• 상온부문장: 멸균 공정 모니터링을 철저히 해주세요.\n• 유제품부문장: 냉장 온도 관리를 생활화합시다.\n• 품질안전부문장: 타협 없는 위생 관리를 당부드립니다."

                intro_custom_text = st.text_area("당부 말씀 원문 (직접 수정 가능):", value=extracted_intro, height=130)

            with col_i2:
                intro_img_file = st.file_uploader("또는 PPT 슬라이드 캡처 이미지(PNG/JPG) 직접 업로드 (선택):", type=["png", "jpg", "jpeg"], key="intro_img", help="업로드 시 AI가 그린 배경 대신 이 이미지가 영상 첫 화면으로 그대로 사용됩니다.")
                if intro_img_file:
                    st.session_state["intro_custom_img"] = intro_img_file.getvalue()
                    st.image(intro_img_file, caption="적용될 인트로 슬라이드 화면", use_container_width=True)
                else:
                    st.session_state["intro_custom_img"] = None

        st.markdown("---")

        # --- 아웃트로 설정 (이벤트 안내) ---
        st.markdown("### Step 3. 클로징: 교육 이벤트 안내 설정 (마지막 Slide)")
        include_outro = st.checkbox("마지막 슬라이드에 이벤트/퀴즈 안내 포함하기", value=True)
        outro_text = ""
        if include_outro:
            col_o1, col_o2 = st.columns(2)
            with col_o1:
                outro_pptx_file = st.file_uploader("이벤트 안내 PPTX 파일 업로드 (선택):", type=["pptx"], key="outro_pptx")
                default_outro = extract_text_from_pptx(outro_pptx_file) if outro_pptx_file else "• 참여 방법: 사내 게시판(또는 QR코드)을 통해 퀴즈 3문항 제출\n• 참여 기간: 금주 금요일 17시까지\n• 경품 안내: 만점자 추첨을 통해 소정의 상품 증정!"
                outro_text = st.text_area("이벤트 안내 문구 입력:", value=default_outro, height=110)
            with col_o2:
                outro_img_file = st.file_uploader("이벤트 슬라이드 캡처 이미지(PNG/JPG) 직접 업로드 (선택):", type=["png", "jpg", "jpeg"], key="outro_img")
                if outro_img_file:
                    st.session_state["outro_custom_img"] = outro_img_file.getvalue()
                    st.image(outro_img_file, caption="적용될 이벤트 슬라이드 화면", use_container_width=True)
                else:
                    st.session_state["outro_custom_img"] = None

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 Step 1~3 통합 AI 슬라이드 & 더빙 대본 기획 시작", type="primary", use_container_width=True):
            if not API_KEY:
                st.error("API 키가 설정되지 않았습니다. Streamlit Secrets에 GEMINI_API_KEY를 등록해주세요.")
            elif not topic.strip():
                st.warning("교육 주제를 입력해주세요.")
            else:
                with st.spinner("원본 자료 분석 및 맞춤형 슬라이드/대본을 기획 중입니다... (영상 포함 시 약 30초~1분 소요)"):
                    uploaded_gemini_files = []
                    combined_extra_text = extra_source_text

                    # 업로드된 참고 파일 처리 (PPTX는 텍스트로 즉시 추출하여 토큰 절약, MP4/PDF는 Gemini 서버로 업로드)
                    for f in uploaded_source_files:
                        ext = os.path.splitext(f.name)[1].lower()
                        if ext == ".pptx":
                            pptx_txt = extract_text_from_pptx(f)
                            combined_extra_text += f"\n\n[업로드된 PPTX ({f.name}) 추출 내용]:\n{pptx_txt}"
                        else:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                                tmp.write(f.getbuffer())
                                tmp_path = tmp.name
                            up_file = genai.upload_file(tmp_path)
                            while up_file.state.name == "PROCESSING":
                                time.sleep(2)
                                up_file = genai.get_file(up_file.name)
                            uploaded_gemini_files.append(up_file)

                    intro_cfg = {"mode": intro_mode, "custom_text": intro_custom_text}
                    outro_cfg = {"include": include_outro, "text": outro_text}

                    slides, pre_log, err = generate_ai_storyboard(
                        selected_model, topic, body_slide_count, intro_cfg, outro_cfg, uploaded_gemini_files, combined_extra_text
                    )

                    if err:
                        st.error(err)
                    else:
                        # 사용자가 직접 업로드한 인트로/아웃트로 이미지가 있으면 슬라이드 데이터에 매핑
                        if slides and st.session_state.get("intro_custom_img") and slides[0].get("type") == "INTRO":
                            slides[0]["custom_image_bytes"] = st.session_state["intro_custom_img"]
                        if slides and st.session_state.get("outro_custom_img") and slides[-1].get("type") == "OUTRO":
                            slides[-1]["custom_image_bytes"] = st.session_state["outro_custom_img"]

                        st.session_state["slides_data"] = slides
                        st.session_state["pre_calc_log"] = pre_log
                        st.session_state["final_video_bytes"] = None
                        st.session_state["final_pptx_bytes"] = None
                        st.success(f"총 {len(slides)}장의 슬라이드 화면 및 더빙 대본 기획이 완료되었습니다! 상단의 [2. 슬라이드 & 대본 검수/수정] 탭으로 이동하세요.")

    # ==========================================
    # 탭 2: 슬라이드 & 대본 검수 및 실시간 수정
    # ==========================================
    with tab2:
        if not st.session_state.get("slides_data"):
            st.info("먼저 [1. 교육 자료 입력 및 AI 대본 기획] 탭에서 기획안을 생성해 주십시오.")
        else:
            st.markdown("### 📝 슬라이드별 화면 텍스트 및 성우 내레이션 검수")
            st.caption("공장장님 보고 전이나 영상 렌더링 전에 오탈자, 수치, 당부 말씀 문구를 자유롭게 수정할 수 있습니다. 수정하신 내용은 PPTX와 MP4 영상에 100% 그대로 반영됩니다.")

            if st.session_state.get("pre_calc_log"):
                with st.expander("🕵️‍♂️ [AI 시스템 로그] 사전 연산(CoT) 기획 의도 확인하기"):
                    st.markdown(st.session_state["pre_calc_log"])

            updated_slides = []
            for idx, slide in enumerate(st.session_state["slides_data"]):
                s_type = slide.get("type", "BODY")
                badge = "👔 [오프닝 - 부문장 당부 말씀]" if s_type == "INTRO" else ("🎁 [클로징 - 이벤트 안내]" if s_type == "OUTRO" else f"📘 [본론 교육 {idx}]")

                with st.expander(f"슬라이드 {idx + 1}: {badge} - {slide.get('title', '')}", expanded=True):
                    col_e1, col_e2 = st.columns([1, 1])
                    with col_e1:
                        new_title = st.text_input(f"슬라이드 {idx+1} 제목", value=slide.get("title", ""), key=f"title_{idx}")
                        new_content = st.text_area(f"PPT 화면 표시 텍스트 (줄바꿈 유지)", value=slide.get("content", ""), height=140, key=f"content_{idx}")
                    with col_e2:
                        new_script = st.text_area(f"성우 음성 더빙 대본 (구어체)", value=slide.get("script", ""), height=140, key=f"script_{idx}")
                        custom_slide_img = st.file_uploader(f"이 슬라이드만 별도 이미지(PPT 캡처본)로 교체 (선택)", type=["png", "jpg", "jpeg"], key=f"custom_img_{idx}")

                    current_img_bytes = custom_slide_img.getvalue() if custom_slide_img else slide.get("custom_image_bytes")

                    updated_slides.append({
                        "type": s_type,
                        "title": new_title,
                        "content": new_content,
                        "script": new_script,
                        "custom_image_bytes": current_img_bytes
                    })

            st.session_state["slides_data"] = updated_slides
            st.success("✅ 위에 입력된 수정 사항이 실시간으로 메모리에 반영되었습니다. 이제 [3. PPTX 및 MP4 교육 영상 렌더링] 탭으로 이동하세요.")

    # ==========================================
    # 탭 3: PPTX 및 MP4 영상 병렬 렌더링 및 다운로드
    # ==========================================
    with tab3:
        if not st.session_state.get("slides_data"):
            st.info("먼저 [1. 교육 자료 입력 및 AI 대본 기획] 탭에서 기획안을 생성해 주십시오.")
        else:
            st.markdown("### 🎬 최종 결과물 생성 (PPTX 문서 + MP4 교육 영상)")
            slides_to_render = st.session_state["slides_data"]
            st.write(f"현재 총 **{len(slides_to_render)}장**의 슬라이드(오프닝/본론/클로징 포함)가 렌더링 대기 중입니다.")

            col_btn1, col_btn2 = st.columns(2)

            with col_btn1:
                if st.button("📊 1단계: 편집 가능한 파워포인트(.pptx) 즉시 생성", use_container_width=True):
                    with st.spinner("실무 수정 및 보고용 PPTX 파일을 생성하고 있습니다..."):
                        pptx_bytes = generate_editable_pptx(slides_to_render)
                        st.session_state["final_pptx_bytes"] = pptx_bytes
                        st.success("파워포인트(.pptx) 생성 완료! (발표자 노트에 대본이 함께 들어있습니다)")

                if st.session_state.get("final_pptx_bytes"):
                    st.download_button(
                        label="📥 파워포인트(.pptx) 파일 다운로드",
                        data=st.session_state["final_pptx_bytes"],
                        file_name=f"연세유업_교육자료_{datetime.date.today().strftime('%Y%m%d')}.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        use_container_width=True
                    )

            with col_btn2:
                if st.button("🎥 2단계: 음성 합성 및 MP4 교육 영상 렌더링 시작", type="primary", use_container_width=True):
                    t_start = time.time()
                    with tempfile.TemporaryDirectory() as temp_dir:
                        total_cnt = len(slides_to_render)
                        progress_bar = st.progress(0, text="병렬 엔진 가동: 슬라이드 이미지 및 성우 음성(TTS) 동시 생성 중...")

                        # 1. 병렬 처리(ThreadPoolExecutor)로 슬라이드 이미지 + TTS 오디오 고속 생성
                        media_results = [None] * total_cnt
                        with ThreadPoolExecutor(max_workers=5) as executor:
                            futures = [
                                executor.submit(media_worker, idx, slide, total_cnt, temp_dir)
                                for idx, slide in enumerate(slides_to_render)
                            ]
                            completed = 0
                            for future in as_completed(futures):
                                idx, img_p, aud_p = future.result()
                                media_results[idx] = (img_p, aud_p)
                                completed += 1
                                progress_bar.progress(
                                    completed / (total_cnt + 1),
                                    text=f"({completed}/{total_cnt}) 슬라이드 화면 및 음성 파일 생성 완료..."
                                )

                        # 2. MoviePy + FFmpeg를 이용한 비디오 클립 병합
                        progress_bar.progress(total_cnt / (total_cnt + 1), text="최종 MP4 비디오 렌더링 중입니다 (오디오 길이에 맞춰 화면 동기화 중)...")
                        video_clips = []
                        audio_clips = []
                        try:
                            for img_path, audio_path in media_results:
                                a_clip = AudioFileClip(audio_path)
                                # 슬라이드 전환 시 너무 급하게 넘어가지 않도록 오디오 끝에 0.5초 여유 부여
                                i_clip = ImageClip(img_path).set_duration(a_clip.duration + 0.5)
                                v_clip = i_clip.set_audio(a_clip)
                                video_clips.append(v_clip)
                                audio_clips.append(a_clip)

                            final_video = concatenate_videoclips(video_clips, method="compose")
                            output_mp4_path = os.path.join(temp_dir, "final_education.mp4")

                            # 정지 화면 위주의 교육 영상에 최적화된 초고속 렌더링 옵션 적용 (fps=10, ultrafast)
                            final_video.write_videofile(
                                output_mp4_path,
                                fps=10,
                                codec="libx264",
                                audio_codec="aac",
                                preset="ultrafast",
                                threads=4,
                                logger=None
                            )

                            with open(output_mp4_path, "rb") as f:
                                st.session_state["final_video_bytes"] = f.read()

                        finally:
                            for c in video_clips:
                                try: c.close()
                                except: pass
                            for a in audio_clips:
                                try: a.close()
                                except: pass

                        progress_bar.empty()
                        st.success(f"🎉 교육 영상 렌더링 완료! (총 소요시간: {time.time() - t_start:.1f}초)")

            if st.session_state.get("final_video_bytes"):
                st.markdown("---")
                st.markdown("#### 📺 완성된 교육 영상 미리보기 및 다운로드")
                st.video(st.session_state["final_video_bytes"])
                st.download_button(
                    label="📥 완성된 MP4 교육 영상 다운로드",
                    data=st.session_state["final_video_bytes"],
                    file_name=f"연세유업_교육영상_{datetime.date.today().strftime('%Y%m%d')}.mp4",
                    mime="video/mp4",
                    type="primary",
                    use_container_width=True
                )

if __name__ == "__main__":
    if check_password():
        main()
