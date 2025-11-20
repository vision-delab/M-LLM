import argparse
import json
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import generate_log_only_result, generate_user_defined_result, make_final_file

HISTORY_DIR = Path(__file__).parent.parent.parent / "history"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--log_type", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    
    input_save_path = HISTORY_DIR / 'Log' / args.title / "input"
    result_paths = HISTORY_DIR / 'Log' / args.title / "result"
    
    input_text = args.text if args.text else ""
    
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    samples_path = [p for p in input_save_path.iterdir() if p.is_file() and p.name != "input.txt"]
    
    result = {}
    result['text'] = []
    result['log_type'] = []
    result['input_paths'] = []
    result['detection_results'] = []
    result['final_file_path'] = []
    
    chunk_size = 15
    
    for input_path in samples_path:
        if input_text:
            all_response = generate_user_defined_result(
                model = model,
                tokenizer = tokenizer,
                dataset_path = input_path,
                user_defined_prompt = input_text,
                chunk_size = chunk_size
            )
        else:
            all_response = generate_log_only_result(
                model = model,
                tokenizer = tokenizer,
                dataset_path = input_path,
                chunk_size = chunk_size
            )
        
        final_file_path = result_paths / f'{input_path.stem}_final.txt'
        make_final_file(input_path, all_response, final_file_path, chunk_size)
        
        result['text'].append(input_text)
        result['log_type'].append(args.log_type)
        result['input_paths'].append(str(input_path))
        result['detection_results'].append(all_response)
        result['final_file_path'].append(str(final_file_path))

    with open(result_paths / "result.json", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()