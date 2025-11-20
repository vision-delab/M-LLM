import os
import torch
from pathlib import Path
import pandas as pd
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from openpyxl.drawing.image import Image
from tqdm import tqdm
import csv


ANSI_ESCAPE_PATTERN = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

################################
# un-pretrained model: Qwen
################################
def read_log_file(path, chunk_size=15):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File does not exist: {path}")

    batches = []
    buffer = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            buffer.append(line)

            if len(buffer) >= chunk_size:
                batches.append(buffer)
                buffer = []

        if buffer:
            batches.append(buffer)

    return batches


def build_user_defined_messages(log_lines, user_anomaly_definition):
    indexed_logs = "\n".join([f"{i+1}. {line}" for i, line in enumerate(log_lines)])

    prompt_text = f"""
You are an expert system log anomaly detector.

Analyze each log entry and determine whether it represents NORMAL or ABNORMAL system behavior based solely on the following user-defined rule:
{user_anomaly_definition}

If a line is ABNORMAL, provide a short and clear explanation of why it is abnormal, including the main cause or affected component. 
Keep the explanation to one or two short sentences.

Output format (strictly follow this, no explanations beyond what is requested):
1. normal
2. abnormal - explanation
3. normal
4. abnormal - explanation
...

Logs:
{indexed_logs}

Now produce the classifications following the format above.
""".strip()

    messages = [
        {"role": "system", "content": "You are Qwen, a helpful assistant specialized in system log anomaly detection."},
        {"role": "user", "content": prompt_text}
    ]
    return messages


def build_log_only_messages(log_lines):
    indexed_logs = "\n".join([f"{i+1}. {line}" for i, line in enumerate(log_lines)])
    
    prompt_text = f"""
You are an expert system log anomaly detector.

Analyze each log entry and determine whether it represents NORMAL or ABNORMAL system behavior.

Rules:
• NORMAL: Regular operational events such as successful startups, connections, completions, or expected state updates.
• ABNORMAL: Any indication of errors, failures, crashes, exceptions, timeouts, unexpected shutdowns, or suspicious activities.
• If a line is ABNORMAL, provide a brief but informative explanation describing the reason for the anomaly. 
  - Include the main cause or affected component.
  - Limit the explanation to one or two short sentences; do not write long paragraphs.

For each input line, output exactly one line.

Output format (strictly follow this, no explanations beyond what is requested):
1. normal
2. abnormal - brief explanation
3. normal
4. abnormal - brief explanation
...

Logs:
{indexed_logs}

Now produce the classifications including a brief explanation for abnormal lines.
""".strip()

    messages = [
        {
            "role": "system",
            "content": "You are Qwen, a helpful assistant specialized in system log anomaly detection."
        },
        {
            "role": "user",
            "content": prompt_text
        }
    ]
    return messages


def generate_user_defined_result(model, tokenizer, dataset_path, user_defined_prompt, chunk_size=15):
    all_responses = []
    log_batches = read_log_file(dataset_path, chunk_size)

    for i, log_batch in enumerate(log_batches):
        messages = build_user_defined_messages(log_batch, user_defined_prompt)

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512
            )

        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        all_responses.append(response)

    return all_responses


def generate_log_only_result(model, tokenizer, dataset_path, chunk_size=15):
    all_responses = []
    log_batches = read_log_file(dataset_path, chunk_size=15)

    for i, log_batch in enumerate(log_batches):
        messages = build_log_only_messages(log_batch)

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512
            )

        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        all_responses.append(response)

    return all_responses


def make_final_file(input_path, preditions, output_path, chunk_size):
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        input_list = f.read()
    input_list = input_list.split('\n')
    
    merged = []
    for chunk in preditions:  
        lines = chunk.split("\n")
        if len(lines) < chunk_size:
            lines.extend([""] * (chunk_size - len(lines)))
        merged.extend(lines)

    cleaned = [re.sub(r"^\s*\d+\.\s*", "", line) for line in merged]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([{"input": i, "cleaned": c} for i, c in zip(input_list, cleaned)], f, ensure_ascii=False, indent=2)


################################
# pretrained model: LogLLM
################################
def fixedSize_window(raw_data, window_size, step_size):
    aggregated = [
        [raw_data['Content'].iloc[i:i + window_size].values]
        for i in range(0, len(raw_data), step_size)
    ]
    return pd.DataFrame(aggregated, columns=list(raw_data.columns))


def sliding_window(raw_data, para):
    """
    split logs into time sliding windows
    :param raw_data: dataframe columns=[timestamp, time duration, content]
    :param para:{window_size: seconds, step_size: seconds}
    :return: dataframe
    """
    log_size = raw_data.shape[0]
    time_data = raw_data['timestamp']
    deltaT_data = raw_data['deltaT']
    content = raw_data['Content']

    new_data = []
    start_end_index_pair = set()

    start_time = time_data[0]
    end_time = start_time + para["window_size"]
    start_index = 0
    end_index = 0

    # get the first start, end index, end time
    for cur_time in time_data:
        if cur_time < end_time:
            end_index += 1
        else:
            break

    start_end_index_pair.add(tuple([start_index, end_index]))

    # move the start and end index until next sliding window
    num_session = 1
    while end_index < log_size:
        start_time = start_time + para['step_size']
        end_time = start_time + para["window_size"]
        for i in range(start_index, log_size):
            if time_data[i] < start_time:
                i += 1
            else:
                break
        for j in range(end_index, log_size):
            if time_data[j] < end_time:
                j += 1
            else:
                break
        start_index = i
        end_index = j

        # when start_index == end_index, there is no value in the window
        if start_index != end_index:
            start_end_index_pair.add(tuple([start_index, end_index]))

        num_session += 1
        if num_session % 1000 == 0:
            print("process {} time window".format(num_session), end='\r')

    for (start_index, end_index) in start_end_index_pair:
        dt = deltaT_data[start_index: end_index].values
        dt[0] = 0
        new_data.append([
            time_data[start_index: end_index].values,
            dt,
            content[start_index: end_index].values,
        ])

    assert len(start_end_index_pair) == len(new_data)
    print('there are %d instances (sliding windows) in this dataset\n' % len(start_end_index_pair))
    return pd.DataFrame(new_data, columns=list(raw_data.columns))


def log_to_dataframe(log_file, regex, headers, start_line, end_line):
    """ Function to transform log file to dataframe
    """
    log_messages = []
    linecount = 0
    cnt = 0

    if end_line is None:
        with open(log_file, 'r', encoding='latin-1') as fin:  # , encoding='latin-1'
            while True:
                line = fin.readline()
                if not line:
                    break
                # for line in fin.readlines():
                cnt += 1
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    # print("\n", line)
                    # print(e)
                    pass

    else:
        line_pos = -1
        with open(log_file, 'r', encoding='latin-1') as fin:
            while True:
                line = fin.readline()
                line_pos += 1
                if line_pos < start_line:
                    continue
                if not line or line_pos >= end_line:
                    break
                cnt += 1
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    # print("\n", line)
                    # print(e)
                    pass

    logdf = pd.DataFrame(log_messages, columns=headers)
    return logdf

def generate_logformat_regex(logformat):
    """ Function to generate regular expression to split log messages
    """
    headers = []
    splitters = re.split(r'(<[^<>]+>)', logformat)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(' +', '\\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    return headers, regex

def structure_log(input_path, output_path, log_name, log_format, start_line = 0, end_line = None):
    headers, regex = generate_logformat_regex(log_format)
    df_log = log_to_dataframe(input_path, regex, headers, start_line, end_line)

    df_log.to_csv(output_path / f'{log_name}_structured.csv', index=False, escapechar='\\')


def sliding_window_data(input_path, result_path, log_type, window_size = 1, step_size = 1):
    
    log_name = Path(input_path).stem
    
    if log_type == 'Thunderbird':
        log_format = '<Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>'   #thunderbird  , spirit, liberty
    else:
        log_format = '<Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>'  #bgl
    
    structure_log(input_path, result_path, log_name, log_format, start_line = 0, end_line = None)
    
    df = pd.read_csv(result_path /f'{log_name}_structured.csv')
    
    session_pred_df = fixedSize_window(
        df[['Content']],
        window_size=window_size, step_size=step_size
    )
    
    col = ['Content']
    spliter=' ;-; '

    session_pred_df = session_pred_df[col]
    session_pred_df['session_length'] = session_pred_df["Content"].apply(len)
    session_pred_df["Content"] = session_pred_df["Content"].apply(lambda x: spliter.join(x))

    mean_session_pred_len = session_pred_df['session_length'].mean()
    max_session_pred_len = session_pred_df['session_length'].max()

    session_pred_df.to_csv(result_path / f'{log_name}_pred.csv',index=False)
    
def session_window_data(input_path, result_path):
    log_name = Path(input_path).stem
    log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>' 
    structure_log(input_path, result_path, log_name, log_format, start_line = 0, end_line = None)
    
    log_structured_file = result_path /f'{log_name}_structured.csv'
    
    df = pd.read_csv(log_structured_file, engine='c', na_filter=False, memory_map=True, dtype={'Date':object, "Time": object})
    
    session_pred_df = df[['Content']].copy()
    session_pred_df['session_length'] = 1

    mean_session_pred_len = session_pred_df['session_length'].mean()
    max_session_pred_len = session_pred_df['session_length'].max()
    session_pred_df.to_csv(result_path / f'{log_name}_pred.csv',index=False)
    
    
def evalModel(model, dataloader, save_paths, log_name, device):
    preds = []
    labels = []
    results_list = []

    output_paths = {} # 반환 파일 저장 dictionary

    model.eval()
    
    with torch.no_grad():
        for bathc_i in tqdm(dataloader):
            inputs = bathc_i['inputs']
            seq_positions = bathc_i['seq_positions']

            inputs = inputs.to(device)
            seq_positions = seq_positions

            outputs_ids = model(inputs, seq_positions)
            outputs_raw_text = model.Llama_tokenizer.batch_decode(outputs_ids)

            original_logs = bathc_i['Content']

            # print(outputs)
            # (추가) abnormal raw data 추출하는 코드 추가
            for i in range(len(outputs_raw_text)):
                raw_prediction = outputs_raw_text[i] # 모델의 원본 예측 텍스트
                original_log_content = original_logs[i] # 원본 로그 템플릿 묶음
                match = re.search(r'normal|anomalous', raw_prediction, re.IGNORECASE)
                clean_prediction = ""
                if match:
                    clean_prediction = match.group().lower()
                    preds.append(clean_prediction)
                else:
                    clean_prediction = "ERROR_OUTPUT"
                    preds.append(clean_prediction)
                
                clean_log_content = ANSI_ESCAPE_PATTERN.sub('', original_log_content)
                results_list.append({
                    'Original_Log_Content': clean_log_content,
                    'Model_Clean_Prediction': clean_prediction
                })
    df_results=pd.DataFrame(results_list)

    # 원본 로그/예측 결과 csv 저장 
    # highlighting anomalous data
    csv_fileName = save_paths / f"{log_name}_detailed_logs.csv"
    df_results.to_csv(csv_fileName, index=False)
    output_paths['detailed_logs_csv'] = str(csv_fileName)

    def highlight_anomalous_row(row):
        if row['Model_Clean_Prediction'] == 'anomalous':
            return ['background-color: #FFCDD2']*len(row)
        else:
            return ['']*len(row)
    
    styled_df = df_results.style.apply(highlight_anomalous_row, axis=1)
    excel_filename= save_paths / f"{log_name}_highlight.xlsx"
    styled_df.to_excel(excel_filename, index=False, engine='openpyxl')
    output_paths['styled_logs_excel'] = str(excel_filename)

    preds_copy = np.array(preds)
    preds = np.zeros_like(preds_copy,dtype=int)
    preds[preds_copy == 'anomalous'] = 1
    preds[preds_copy != 'anomalous'] = 0
    
    pred_num_anomalous = (preds == 1).sum()
    pred_num_normal =  (preds == 0).sum()

    # 1. 총 분석 블록 수 (preds는 현재 numpy 정수 배열)
    total_blocks = len(preds)
    
    # 2. 탐지된 비정상 블록 수 (이미 계산된 'pred_num_anomalous' 값 사용)
    detected_anomalous_blocks = pred_num_anomalous
    
    # 3. 총 분석 로그 라인 수 (루프 초반에 저장한 results_list 사용)
    total_log_lines = 0
    for item in results_list: # 'results_list' (list of dicts) 사용
        # ' ;-; ' 구분자를 기준으로 로그 라인 수 계산
        total_log_lines += item['Original_Log_Content'].count(' ;-; ') + 1
        
    # 4. 비정상 블록 비율
    anomalous_ratio_percent = 0.0
    if total_blocks > 0:
        anomalous_ratio_percent = (detected_anomalous_blocks / total_blocks) * 100

    # === 1. 시각화 데이터 준비 ===
    
    # KPI 데이터 (파이 차트용)
    total_normal_blocks = total_blocks - detected_anomalous_blocks
    pie_sizes = [total_normal_blocks, detected_anomalous_blocks]
    pie_labels = ['Normal Blocks', 'Anomalous Blocks']
    pie_colors = ['#8FDEBD', '#FF9999'] # (정상:초록, 비정상:빨강)

    # 핫스팟 데이터 (롤링 평균 플롯용)
    # preds는 0과 1로 구성된 NumPy 배열
    df_preds = pd.DataFrame({'Anomaly': preds})
    
    # 100개 블록 단위로 비정상 비율을 계산 (창 크기는 조절 가능)
    window_size = 100 
    df_preds['AnomalyRate'] = df_preds['Anomaly'].rolling(
        window=window_size, min_periods=1
    ).mean()

    # === 2. 그래프 생성 및 이미지 파일로 저장 ===
    
    # (A) KPI 파이 차트
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(pie_sizes, 
            labels=pie_labels, 
            colors=pie_colors, 
            autopct='%1.1f%%', 
            startangle=90)
    ax_pie.axis('equal')
    ax_pie.set_title('Anomaly Detection Summary')
    pie_chart_filename = save_paths / f"{log_name}_summary_pie.png"
    fig_pie.savefig(pie_chart_filename)
    output_paths['summary_pie_chart'] = str(pie_chart_filename) # (수정) 경로 저장
    plt.close(fig_pie)

    # (B) 핫스팟 롤링 플롯
    fig_hotspot, ax_hotspot = plt.subplots(figsize=(15, 6))
    df_preds['AnomalyRate'].plot(ax=ax_hotspot, 
                                color='red', 
                                lw=2,
                                title=f'Anomaly Hotspots (Rolling Average over {window_size} blocks)')
    ax_hotspot.set_xlabel('Block Index (Sequence)')
    ax_hotspot.set_ylabel('Anomaly Rate in Window')
    ax_hotspot.grid(True)
    hotspot_plot_filename = save_paths / f"{log_name}_hotspot_plot.png"
    fig_hotspot.savefig(hotspot_plot_filename)
    output_paths['hotspot_plot'] = str(hotspot_plot_filename)
    plt.close(fig_hotspot)

    hotspot_data_filename = save_paths / f"{log_name}_hotspot_data.csv"
    df_preds.to_csv(hotspot_data_filename, index=False)
    output_paths['hotspot_data_csv'] = str(hotspot_data_filename)

    # === 3. ExcelWriter로 다중 시트 리포트 생성 ===
    
    report_excel_filename = save_paths / f"{log_name}_visual_report.xlsx"
    output_paths['integrated_report_excel'] = str(report_excel_filename)
    with pd.ExcelWriter(report_excel_filename, engine='openpyxl') as writer:
        
        # --- 시트 1: Dashboard ---
        # KPI 요약 데이터를 DataFrame으로 생성
        kpi_data = {
            'Metric': [
                "Total Log Lines Analyzed",
                "Total Log Blocks Analyzed",
                "Detected Anomalous Blocks",
                "Anomalous Block Ratio",
            ],
            'Value': [
                f"{total_log_lines:,}",
                f"{total_blocks:,}",
                f"{detected_anomalous_blocks:,}",
                f"{anomalous_ratio_percent:.2f} %",
            ]
        }
        df_kpi = pd.DataFrame(kpi_data)
        kpi_csv_filename = save_paths / f"{log_name}_kpi_summary.csv"
        df_kpi.to_csv(kpi_csv_filename, index=False)
        output_paths['kpi_summary_csv'] = str(kpi_csv_filename)
        
        # 'Dashboard' 시트에 KPI 데이터 작성
        df_kpi.to_excel(writer, sheet_name='Dashboard', startrow=1, startcol=1, index=False)
        
        # 'Dashboard' 시트 가져오기
        sheet_dashboard = writer.sheets['Dashboard']
        
        # 시트에 파이 차트 이미지 삽입
        img_pie = Image(pie_chart_filename)
        img_pie.anchor = 'E2' # (E열 2행 근처에 이미지 위치)
        sheet_dashboard.add_image(img_pie)
        
        # 시트에 핫스팟 플롯 이미지 삽입
        img_hotspot = Image(hotspot_plot_filename)
        img_hotspot.anchor = 'B15'
        sheet_dashboard.add_image(img_hotspot)
        
        # (선택) 시트 컬럼 너비 자동 조절
        sheet_dashboard.column_dimensions['B'].width = 30
        sheet_dashboard.column_dimensions['C'].width = 20

        # --- 시트 2: Detailed Logs (하이라이트 적용) ---
        styled_df.to_excel(writer, sheet_name='Detailed_Logs', index=False)
        
        df_preds.to_excel(writer, sheet_name='Hotspot_Plot_Data', index=False)

    return output_paths