import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from pathlib import Path
from io import BytesIO
from qwen_vl_utils import process_vision_info
import json
import re
import numpy as np


################################
# Qwen
################################
def make_list(data_path):
    ext = Path(data_path).suffix.lower()

    if ext in {".csv", ".txt"}:
        df = pd.read_csv(data_path, delimiter=',', header=None)
        values = df.values.flatten().tolist()
        return values

    else:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    
def convert_to_png(image_path):
    path = Path(image_path)
    name = path.stem
    ext = path.suffix.lstrip('.')
    
    new_name = f'{name}-{ext}.png'
    new_path = path.with_name(new_name)
    
    img = Image.open(path)
    img.save(new_path, format='PNG')
    
    path.unlink()
    
    return str(new_path)

def make_ts_image(length, values, label=None):
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


def make_ts_image_for_test(image_path, length, values, predicted):
    predicted_idx = [i for i,l in enumerate(predicted) if l==1] 

    plt.clf()
    plt.figure(figsize=(12, 4))
    plt.plot(range(length), values)
    
    ymin, ymax = min(values), max(values)
    plt.bar(predicted_idx, height=ymax - ymin,
            bottom=ymin, width=1, color='red', alpha=0.3)

    plt.xlabel("Time")
    plt.ylabel("Value")

    plt.tight_layout()
    plt.savefig(image_path)


def generate_and_save_input_image(sample_values, output_dir, input_name):
    sample_length = len(sample_values)
    sample_image = make_ts_image(sample_length, sample_values)
    input_save_path = os.path.join(output_dir, f"{Path(input_name).stem}-{Path(input_name).suffix[1:]}.png")
    sample_image.save(input_save_path, format='PNG')
    return input_save_path


def run_inference(model, processor, image_path, device):
    sample_image = Image.open(image_path)
    messages = [
        {"role": "system", "content": "You are a time series anomaly detector."},
        {"role": "user", "content": [
            {"type": "image", "image": sample_image},
            {"type": "text", "text": """\
                Detect ranges of anomalies in this time series, in terms of the x-axis coordinate.
                List one by one, in JSON format. 
                If there are no anomalies, answer with an empty list [].
                Output template:
                [{"start": ..., "end": ...}, {"start": ..., "end": ...}...]
                """
            },
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
    try:
        qwen_output = re.findall(r'\[.*?\]', output_text[0])[0]
    except:
        qwen_output = '[]'
    return qwen_output


def visualize(data, inference_result, result_path, sample_length):
    parsed = json.loads(inference_result)
    result = [(item["start"], item["end"]) for item in parsed]
    
    predicted = [0 for _ in range(sample_length)]
    for answer in result:
        start, end = answer[0], answer[1]
        start = max(0, int(start))
        end = min(sample_length, int(end))
        if start < end: predicted[start:end] = [1 for _ in range(start, end)]
        
    make_ts_image_for_test(result_path, sample_length, data, predicted)
    
    
################################
# Qwen With FSM
################################
def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == 'RGBA':
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])
        return white_background
    else:
        return pil_image.convert("RGB")


def stack_images_vertically(image_list):
    """이미지 리스트를 수직으로 쌓아 하나의 이미지로 병합"""
    if not image_list:
        return None
    widths, heights = zip(*(img.size for img in image_list))
    max_width = max(widths)
    total_height = sum(heights)
    new_image = Image.new('RGB', (max_width, total_height), (255, 255, 255))
    y_offset = 0
    for img in image_list:
        new_image.paste(img, (0, y_offset))
        y_offset += img.height
    return new_image


def segment_image_process(values, xlim_arr, ylim_arr=(-1, 1)):
    """
    주어진 값(values)을 그래프로 그려 PIL 이미지로 변환
    """
    plt.close()
    plt.figure(figsize=(12, 2))

    x_range = range(xlim_arr[0], xlim_arr[1])

    plt.plot(x_range, values)
    plt.xlim(xlim_arr)
    plt.ylim(ylim_arr)

    # x축 눈금 설정
    if (xlim_arr[1] - xlim_arr[0]) < 5:
        app_range = 1
    else:
        app_range = (xlim_arr[1] - xlim_arr[0]) // 5

    xticks = list(range(xlim_arr[0], xlim_arr[1] + 1, app_range))
    plt.xticks(xticks, fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    pil_image = to_rgb(Image.open(buf))
    plt.close()
    return pil_image


def make_json_for_prediction(pred):
    ranges = []

    in_anomaly = False
    start = None

    for i, val in enumerate(pred):
        if val == 1 and not in_anomaly:
            # 새로운 anomaly 구간 시작
            start = i
            in_anomaly = True
        elif val == 0 and in_anomaly:
            # anomaly 구간 종료
            end = i - 1
            ranges.append({"start": start, "end": end})
            in_anomaly = False

    # 배열 마지막이 1로 끝나는 경우 처리
    if in_anomaly:
        ranges.append({"start": start, "end": len(arr) - 1})

    # JSON 문자열로 변환
    json_str = json.dumps(ranges)
    return json_str



def get_period_arr(values, time_length, top_k=2, interval=100):
    """FFT를 사용하여 시계열의 주요 주기(Period)를 계산"""
    half = len(values) // 2
    fft_result = np.fft.fft(values)
    magnitude = np.abs(fft_result)[:half]

    m_tuple = [(i + 1, v) for i, v in enumerate(magnitude[1:])]
    sm_tuple = sorted(m_tuple, key=lambda x: x[1])[::-1]

    high_indice = [v[0] for v in sm_tuple]
    magnitude_arr = [v[1] for v in sm_tuple]

    period_arr = [round(time_length / i) for i in high_indice]

    if interval > 0 and top_k > 0:
        set_arr = [(magnitude_arr[i], period_arr[i]) for i in range(len(period_arr))]
        selected = []
        selected_periods = []

        for magnitude, period in set_arr:
            if period > 8:  # 너무 짧은 주기는 제외
                is_valid = True
                for p in selected_periods:
                    if abs(period - p) < interval:
                        is_valid = False
                        break
                if is_valid:
                    selected.append((magnitude, period))
                    selected_periods.append(period)
                    if len(selected) == top_k:
                        break

        period_arr = [v[1] for v in selected]
    else:
        period_arr = []

    return period_arr


def time_segmentation(values, time_length, period_arr, top_k=2):
    """주기를 기반으로 시계열 데이터를 세그먼트로 분할"""
    all_segment_list = []
    run_k = min(top_k, len(period_arr))
    for k in range(run_k):
        period = period_arr[k]
        if period == 0: continue

        segment_list_size = time_length // period
        if segment_list_size == 0: continue

        segment_length = time_length // segment_list_size

        segment_list = [
            values[i * segment_length: i * segment_length + segment_length]
            for i in range(0, segment_list_size)
        ]
        all_segment_list.append(segment_list)
    return all_segment_list


def seg_grouping(segment_list, group_length=4):
    """세그먼트들을 그룹핑"""
    group_list = []
    group = []
    segment_list_size = len(segment_list)
    for i in range(segment_list_size):
        group.append(segment_list[i])
        if i != 0 and (i + 1) % group_length == 0:
            group_list.append(group)
            group = []
        elif i + 1 == segment_list_size:
            group_list.append(group)
    return group_list


def local_qwen_inference(model, processor, image_input, prompt, device):
    """내부 Qwen 추론 래퍼 함수"""
    messages = [
        {"role": "system", "content": "You are a time series anomaly detector."},
        {"role": "user", "content": [
            {"type": "image", "image": image_input},
            {"type": "text", "text": prompt},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)

    return output_text[0]


def group_inference(model, processor, prompt, group_list, time_length, group_length=4, device='cuda'):
    """그룹화된 세그먼트에 대해 추론 수행"""
    output_list = []
    print(f"  - Running Group Inference (Groups: {len(group_list)})...")

    for i, group in enumerate(group_list):
        pil_image_list = []
        for j, segment in enumerate(group):
            segment_length = len(segment)
            index = group_length * i + j
            start_x = index * segment_length
            xticks_range = (start_x, start_x + segment_length)

            pil_image = segment_image_process(segment, xticks_range, (-1, 1))
            pil_image_list.append(pil_image)

        group_image = stack_images_vertically(pil_image_list)

        raw_output = local_qwen_inference(model, processor, group_image, prompt, device)

        try:
            output_str = re.findall(r'\[.*?\]', raw_output)[0]
        except:
            output_str = '[]'

        try:
            parsed = json.loads(output_str)
            for item in parsed:
                if item["start"] >= 0:
                    temp_dict = {}
                    temp_dict['start'] = item["start"]
                    temp_dict['end'] = item["end"]
                    output_list.append(temp_dict)
        except:
            pass

    return output_list


def segment_inference(model, processor, prompt, segment_list, time_length, device='cuda'):
    """개별 세그먼트에 대해 순차 추론 수행"""
    output_list = []
    segment_length = len(segment_list[0])
    print(f"  - Running Segment Inference (Segments: {len(segment_list)})...")

    for i, segment in enumerate(segment_list):
        current_start = i * segment_length
        current_end = current_start + len(segment)

        pil_image = segment_image_process(segment, (current_start, current_end), (-1, 1))

        raw_output = local_qwen_inference(model, processor, pil_image, prompt, device)

        try:
            output_str = re.findall(r'\[.*?\]', raw_output)[0]
        except:
            output_str = '[]'

        try:
            parsed = json.loads(output_str)
            for item in parsed:
                if item["start"] >= 0:
                    temp_dict = {}
                    temp_dict['start'] = item["start"]
                    temp_dict['end'] = item["end"]
                    output_list.append(temp_dict)
        except:
            pass

    return output_list

    
def run_fsm_pipeline(sample_values, args, model, processor, output_path, device):
    time_length = len(sample_values)

    # 1. Period Calculation (FFT)
    period_arr = get_period_arr(sample_values, time_length, top_k=args.top_k, interval=args.interval)

    # 2. Prepare Prompt
    prompt_text = """\
        Detect ranges of anomalies in this time series, in terms of the x-axis coordinate.
        List one by one, in JSON format.
        If there are no anomalies, answer with an empty list [].
        Output template:
        [{"start": ..., "end": ...}, {"start": ..., "end": ...}...]
    """

    all_predicted_masks = []

    # --- A. Original (Whole Series) Inference ---
    print("2. Running Original (Global) Inference...")
    org_image = segment_image_process(sample_values, (0, time_length), (-1, 1))
    raw_output_org = local_qwen_inference(model, processor, org_image, prompt_text, device)

    try:
        json_org = re.findall(r'\[.*?\]', raw_output_org)[0]
        parsed_org = json.loads(json_org)
    except:
        parsed_org = []

    mask_org = np.zeros(time_length)
    for item in parsed_org:
        s, e = int(item['start']), int(item['end'])
        s, e = max(0, s), min(time_length, e)
        if s < e: mask_org[s:e] = 1

    # --- B. Segmented Inference (per Period) ---
    segments_results_masks = []

    if args.top_k > 0 and len(period_arr) > 0:
        print("3. Running Segmented Inference...")
        all_segment_lists = time_segmentation(sample_values, time_length, period_arr, top_k=args.top_k)

        for k, segment_list in enumerate(all_segment_lists):
            print(f"   -> Processing Period Index {k} (Period: {period_arr[k]})...")

            if args.grouping:
                group_list = seg_grouping(segment_list, group_length=args.group_length)
                results = group_inference(model, processor, prompt_text, group_list, time_length, args.group_length,
                                          device)
            else:
                results = segment_inference(model, processor, prompt_text, segment_list, time_length, device)

            mask_seg = np.zeros(time_length)
            for item in results:
                s, e = int(item['start']), int(item['end'])
                s, e = max(0, s), min(time_length, e)
                if s < e: mask_seg[s:e] = 1
            segments_results_masks.append(mask_seg)

    if len(segments_results_masks) > 0:
        ffp_predicted = np.mean(np.array(segments_results_masks), axis=0)
    else:
        ffp_predicted = mask_org

    org_predicted = mask_org

    # 결과 합치기
    alpha = 0.5
    final_score_map = alpha * ffp_predicted + (1 - alpha) * org_predicted
    pred_binary = (final_score_map >= 0.5).astype(int)

    make_ts_image_for_test(output_path, time_length, sample_values, pred_binary)

    # inference result로 변경 필요
    json_prediction = make_json_for_prediction(pred_binary)
    
    return json_prediction