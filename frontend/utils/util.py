from pathlib import Path
import pandas as pd
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import streamlit as st
import streamlit.components.v1 as components
import os

# ================================
# Text area
# ================================
def text_area_style(text):
    html_content = text.replace("\n", "<br>")
    st.markdown(f"""
        <style>
        .readonly-box {{
            background-color: var(--secondary-background-color)
            padding: 10px 12px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', 'Consolas', monospace;
            font-size: 14px;
            line-height: 1.5;
            max-height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-word;
        }}
        </style>

        <div class="readonly-box .st-7b">{html_content}</div>
    """, unsafe_allow_html=True)


# ================================
# Log
# ================================


# ================================
# Time 
# ================================
def make_download_sort(file_path, name):
    col1, col2 = st.columns([5, 1])  # col1: 파일 이름, col2: 버튼
    with col1:
        st.markdown(
            f"<div style='display: flex; align-items: center; height: 40px;'>📄 {name}</div>",
            unsafe_allow_html=True
        )
            
    with col2:
        with open(file_path, "rb") as f:
            st.download_button(
                label="Download",
                data=f.read(),
                file_name=os.path.basename(file_path),
                mime="application/octet-stream"
            )
            
            
def make_download_preview(file):
    col1, col2 = st.columns([5, 1])  # col1: 파일 이름, col2: 버튼       
    with col1:
        st.markdown(
            f"<div style='display: flex; align-items: center; height: 40px;'>📄 {file.name}</div>",
            unsafe_allow_html=True
        )
    with col2:
        st.download_button(
            label="Download",
            data=file.getvalue(),
            file_name=file.name,
            mime=file.type
        )
        
# def center_image(file_path, caption):
#     st.markdown(f"""
#     <figure style="text-align: center;">
#         <img src="{str(file_path)}" width="300">
#         <figcaption>{caption}</figcaption>
#     </figure>
#     """, unsafe_allow_html=True)


def make_list(uploaded_file):
    uploaded_file.seek(0)
    ext = Path(uploaded_file.name).suffix.lower()

    # ================================
    # 1) CSV / TXT
    # ================================
    if ext in {".csv", ".txt"}:
        try:
            df = pd.read_csv(uploaded_file, delimiter=',', header=None)
        except Exception as e:
            raise ValueError(
                f"CSV/TXT 파일은 쉼표로 구분된 숫자 데이터여야 합니다. "
                f"파일 읽기 중 오류 발생: {e}"
            )

        # 평탄화
        values = df.values.flatten().tolist()

        clean_values = []
        for v in values:
            try:
                clean_values.append(float(v))
            except Exception as e:
                raise ValueError(
                    f"CSV/TXT 파일에는 숫자만 들어있어야 합니다. 입력: {values}"
                )
        return clean_values

    # ================================
    # 2) PKL
    # ================================
    elif ext == ".pkl":
        try:
            data = pickle.load(uploaded_file)
        except Exception as e:
            raise ValueError(f"PKL 파일 로딩 오류: {e}")
        return data

    else:
        raise ValueError(f"지원하지 않는 확장자: {ext}")
    
def make_ts_image(length, values):
    plt.clf()
    plt.figure(figsize=(12, 4))
    plt.plot([num for num in range(length)], [score for score in values])
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    pil_image = Image.open(buf)
    return pil_image