#!/bin/bash
set -e
set -o pipefail

# 기본 값 초기화
TEXT=""
TITLE=""
LOG_TYPE=""

# 1. 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --text)
            TEXT="$2"
            shift 2
            ;;
        --title)
            TITLE="$2"
            shift 2
            ;;
        --log_type)
            LOG_TYPE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "TEXT=$TEXT"
echo "TITLE=$TITLE"
echo "LOG_TYPE=$LOG_TYPE"

# 2. Conda 환경 활성화
source ~/anaconda3/etc/profile.d/conda.sh
conda activate logllm

# 3. Python 스크립트 실행
python log/log_unpretrained.py \
    --text "$TEXT" \
    --title "$TITLE" \
    --log_type "$LOG_TYPE"

conda deactivate