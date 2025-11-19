import streamlit as st
from PIL import Image
import json
from pathlib import Path
import os
import re
from .util import text_area_style

HISTORY_DIR = Path(__file__).parent.parent.parent / "history" / "Log"


def log_pre(result: dict):
    pass


def log_unpre(result: dict):
    
    texts = result.get("text", [])
    log_type = result.get("log_type", [])
    input_paths = result.get("input_paths", [])
    detection_results = result.get("detection_results", [])

    # 입력 다운로드 
    low_input_paths = HISTORY_DIR / st.session_state.title
    
    for i in range(len(input_paths)):
        st.markdown("---")
        
        file_path = input_paths[i]
        col1, col2 = st.columns([5, 1])  # col1: 파일 이름, col2: 버튼
        with col1:
            st.write(f"📄 {Path(file_path).name}")
        with col2:
            with open(file_path, "rb") as f:
                st.download_button(
                    label="Download",
                    data=f.read(),
                    file_name=os.path.basename(file_path),
                    mime="application/octet-stream"
                )
        
        if texts[i]:
            st.markdown(f"<h3 style='text-align: left;'>User Text</h3>", unsafe_allow_html=True)
            # text_area_style(texts[i])
            st.text_area("Input Text", texts[i], label_visibility="collapsed", key=f"text_{i}")
        else:
            st.markdown(f"<h3 style='text-align: left;'>No Input Text</h3>", unsafe_allow_html=True)
        
        st.subheader(Path(file_path).name)

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        numbered = "\n".join([
            f"{i+1}. {line}"
            for i, line in enumerate(content.splitlines())
        ])
        # text_area_style(numbered)
        st.text_area("File Content", numbered, height=400, label_visibility="collapsed", key=f"File_{i}")
        
        # prediction 보여주기
        merged = []
        for chunk in detection_results[i]:  
            lines = chunk.split("\n")
            merged.extend(lines)

        # 기존 번호 제거 → "1. abnormal" → "abnormal"
        cleaned = [re.sub(r"^\s*\d+\.\s*", "", line) for line in merged]
        
        # 새 번호 다시 붙이기
        renumbered = "\n".join([f"{i+1}. {line}" for i, line in enumerate(cleaned)])
        # print(renumbered)
        
        st.write("### Detection Results")
        # text_area_style(renumbered)
        st.text_area("Detections", renumbered, height=400, label_visibility="collapsed", key=f"predict_{i}")