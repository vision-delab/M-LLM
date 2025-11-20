import streamlit as st
from PIL import Image
import json
from pathlib import Path
import os
from .util import text_area_style, make_download_sort

HISTORY_DIR = Path(__file__).parent.parent.parent / "history" / "Time"

def time_method(result: dict):
    row_names = result.get("row_name", [])
    method = result.get("method", [])
    input_paths = result.get("input_paths", [])
    result_paths = result.get("result_paths", [])
    texts = result.get("text", [])
    result_text = result.get("inference_result", [])

    # 입력 다운로드 
    low_input_paths = HISTORY_DIR / st.session_state.title
    
    for i in range(len(input_paths)):
        st.markdown("---")
        
        subheader_name = row_names[i] if i < len(row_names) else f"Sample {i+1}"
        st.subheader(f"Input File name: {subheader_name}")
        
        file_path = low_input_paths / "input" / row_names[i]
        make_download_sort(file_path, row_names[i])
        make_download_sort(input_paths[i], Path(input_paths[i]).name)
        make_download_sort(result_paths[i], Path(result_paths[i]).name)

        # 입력 텍스트 표시
        if texts[i]:
            st.markdown(f"<h3 style='text-align: left;'>User Text</h3>", unsafe_allow_html=True)
            # text_area_style()
            st.text_area("Input Text", texts[i], disabled=False, label_visibility="collapsed", key=f"text_area_{i}")
        else:
            st.markdown(f"<h3 style='text-align: left;'>No Input Text</h3>", unsafe_allow_html=True)

        # 입력 이미지 표시
        img = Image.open(input_paths[i])
        st.image(img, caption=f"Input: {row_names[i]}")

        # 결과 이미지 표시
        res_img = Image.open(result_paths[i])
        st.image(res_img, caption=f"Result: Predicted by Time Anomaly Detector - {method[i]}")
        
        if not result_text[i] or result_text[i].strip() == "[]":
            st.text("Model Response: No anomalies detected.")
        else:
            anomalies = json.loads(result_text[i])
            for idx, a in enumerate(anomalies):
                start = a.get("start")
                end = a.get("end")
                st.write(f"{idx+1}. Start: {start}, End: {end}")


def time_method3(result: dict):
    pass