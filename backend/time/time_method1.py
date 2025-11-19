import argparse
import json
import os
from pathlib import Path
from utils import * 
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch


HISTORY_DIR = Path(__file__).parent.parent.parent / "history"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    return parser.parse_args()


def main(): 
    args = parse_args()
    
    input_save_path = HISTORY_DIR / 'Time' / args.title / "input"
    result_paths = HISTORY_DIR / 'Time' / args.title / "result"
    
    samples_path = [p for p in input_save_path.iterdir() if p.is_file() and p.name != "input.txt"]
    
    result = {}
    result['row_name'] = []
    result['method'] = []
    result['input_paths'] = []
    result['result_paths'] = []
    result['text'] = []
    result['inference_result'] = []
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "time/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        "time/Qwen2-VL-7B-Instruct"
    )
    
    for s in samples_path:
        data = make_list(s)
        sample_length = len(data)
        save_path = generate_and_save_input_image(data, input_save_path, s)
    
        inference_result = run_inference(model, processor, save_path, device)
        result_path = result_paths / f"{Path(s).stem}-{Path(s).suffix.lstrip('.')}_result.png"
        visualize(data, inference_result, result_path, sample_length)
    
        result['row_name'].append(s.name)
        result['method'].append(args.method)
        result['input_paths'].append(str(save_path))
        result['result_paths'].append(str(result_path))
        result['text'].append(args.text if args.text else "")
        result['inference_result'].append(inference_result)
        
    with open(result_paths / "result.json", "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()