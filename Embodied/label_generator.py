#!/usr/bin/env python3
"""
YOLOv7 Auto-Labeling Tool using Eagle VLM

Usage example:
  python label_generator.py \
    --image_dir /data/JPEGImages \
    --output_dir /data/dataset \
    --model_path NVEagle/Eagle-X5-13B-Chat \
    --model_save_dir ./models \
    --prompts "person:0" "car:1" "bicycle:2" \
    --conf_threshold 0.5 \
    --iou_threshold 0.45 \
    --nms_mode per_class \
    --debug_interval 1000 \
    --gpu 0
"""

import os
import sys
import json
import re
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# Eagle 모듈 경로 추가 (같은 레포 내 Eagle 폴더)
_EAGLE_PATH = Path(__file__).resolve().parent.parent / "Eagle"
if str(_EAGLE_PATH) not in sys.path:
    sys.path.insert(0, str(_EAGLE_PATH))

try:
    from eagle.model.builder import load_pretrained_model
    from eagle.utils import disable_torch_init
    from eagle.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images
    from eagle.constants import (
        IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
    )
    from eagle.conversation import conv_templates
except ImportError as e:
    print(f"[ERROR] Eagle 모듈을 찾을 수 없습니다: {e}")
    print(f"  Eagle 경로: {_EAGLE_PATH}")
    print("  Eagle 폴더가 상위 디렉토리에 있는지 확인하세요.")
    sys.exit(1)


# ─── 상수 ──────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

BOX_COLORS = [
    'red', 'lime', 'blue', 'yellow', 'cyan', 'magenta',
    'orange', 'purple', 'pink', 'brown', 'coral', 'gold',
]

# Eagle에 보낼 프롬프트 템플릿
DETECTION_PROMPT = (
    'Detect all instances of "{class_name}" in this image.\n'
    'For each detected object output a bounding box [x1, y1, x2, y2] '
    'where values are normalized 0.0–1.0 '
    '(x1,y1=top-left corner, x2,y2=bottom-right corner) '
    'and a confidence score 0.0–1.0.\n'
    'Respond with ONLY a valid JSON array, no explanation:\n'
    '[{{"bbox": [x1, y1, x2, y2], "confidence": 0.95}}, ...]\n'
    'If no object is found respond with exactly: []'
)


# ─── 파싱 ──────────────────────────────────────────────────────────────────

def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def parse_bbox_response(
    response: str,
    img_w: int,
    img_h: int,
) -> List[Dict]:
    """Eagle 텍스트 응답에서 bounding box 목록을 추출한다."""
    response = response.strip()

    # 1) JSON 배열 파싱 시도
    json_match = re.search(r'\[.*?\]', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            boxes = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                bbox = item.get('bbox') or item.get('box') or item.get('coordinates')
                conf = float(item.get('confidence', item.get('score', item.get('conf', 0.85))))
                if not bbox or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = (float(v) for v in bbox)
                # 픽셀 좌표인 경우 정규화
                if x2 > 1.5 or y2 > 1.5:
                    x1 /= img_w; y1 /= img_h
                    x2 /= img_w; y2 /= img_h
                x1, y1, x2, y2 = (_clamp01(v) for v in (x1, y1, x2, y2))
                if x2 > x1 and y2 > y1:
                    boxes.append({'bbox': [x1, y1, x2, y2], 'confidence': conf})
            return boxes
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # 2) 정규식 폴백: [0.1, 0.2, 0.8, 0.9] 패턴 탐색
    boxes = []
    pattern = r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]'
    for m in re.finditer(pattern, response):
        try:
            x1, y1, x2, y2 = (float(m.group(i)) for i in range(1, 5))
            if x2 > 1.5 or y2 > 1.5:
                x1 /= img_w; y1 /= img_h
                x2 /= img_w; y2 /= img_h
            x1, y1, x2, y2 = (_clamp01(v) for v in (x1, y1, x2, y2))
            if x2 > x1 and y2 > y1:
                boxes.append({'bbox': [x1, y1, x2, y2], 'confidence': 0.8})
        except ValueError:
            continue
    return boxes


# ─── 좌표 변환 ─────────────────────────────────────────────────────────────

def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float) -> Tuple:
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w  = x2 - x1
    h  = y2 - y1
    return cx, cy, w, h


# ─── NMS ───────────────────────────────────────────────────────────────────

def _iou(a: Tuple, b: Tuple) -> float:
    """a, b: (x1, y1, x2, y2) normalized"""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _nms_single(boxes: List, iou_thr: float) -> List:
    """boxes: list of (x1,y1,x2,y2,conf,cls_id), confidence 내림차순 정렬 후 NMS."""
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if _iou(best[:4], b[:4]) < iou_thr]
    return keep


def apply_nms(
    all_boxes: List,
    nms_mode: str,
    conf_thr: float,
    iou_thr: float,
) -> List:
    filtered = [b for b in all_boxes if b[4] >= conf_thr]
    if not filtered:
        return []
    if nms_mode == 'unified':
        return _nms_single(filtered, iou_thr)
    # per_class
    result = []
    for cls_id in {b[5] for b in filtered}:
        result.extend(_nms_single([b for b in filtered if b[5] == cls_id], iou_thr))
    return result


# ─── 디버그 시각화 ─────────────────────────────────────────────────────────

def save_debug_image(
    image_path: str,
    boxes: List,
    class_map: Dict[int, str],
    output_path: str,
) -> None:
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    iw, ih = img.size

    for (x1, y1, x2, y2, conf, cls_id) in boxes:
        color = BOX_COLORS[cls_id % len(BOX_COLORS)]
        px1, py1 = int(x1 * iw), int(y1 * ih)
        px2, py2 = int(x2 * iw), int(y2 * ih)
        draw.rectangle([px1, py1, px2, py2], outline=color, width=3)
        label = f"{class_map.get(cls_id, str(cls_id))} {conf:.2f}"
        draw.rectangle([px1, max(0, py1 - 18), px1 + len(label) * 7, py1], fill=color)
        draw.text((px1 + 2, max(0, py1 - 17)), label, fill='white')

    img.save(output_path)


# ─── Eagle 추론 ────────────────────────────────────────────────────────────

def infer_single(
    model,
    tokenizer,
    image_processor,
    image: Image.Image,
    prompt_text: str,
    conv_mode: str,
    device: str,
) -> str:
    if model.config.mm_use_im_start_end:
        inp = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN +
               DEFAULT_IM_END_TOKEN + '\n' + prompt_text)
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt_text

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_tensor = process_images([image], image_processor, model.config)[0]
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    )

    input_ids = input_ids.unsqueeze(0).to(device=device, non_blocking=True)
    image_tensor = image_tensor.unsqueeze(0).to(
        dtype=torch.float16, device=device, non_blocking=True
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=False,
            temperature=0.0,
            max_new_tokens=512,
            use_cache=True,
        )

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


# ─── 모델 로드 ─────────────────────────────────────────────────────────────

def resolve_model_path(model_path: str, model_save_dir: str, log: logging.Logger) -> str:
    """로컬 경로가 존재하면 그대로 사용, 없으면 HuggingFace에서 다운로드."""
    if os.path.isdir(model_path):
        log.info(f"로컬 모델 경로 사용: {model_path}")
        return model_path

    # 캐시 경로: models/NVEagle__Eagle-X5-13B-Chat
    safe_name = model_path.replace('/', '__')
    cached = os.path.join(model_save_dir, safe_name)
    if os.path.isdir(cached):
        log.info(f"캐시된 모델 사용: {cached}")
        return cached

    log.info(f"'{model_path}' 다운로드 중 → {cached}")
    os.makedirs(model_save_dir, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_path, local_dir=cached)
        log.info(f"모델 저장 완료: {cached}")
    except Exception as e:
        log.error(f"모델 다운로드 실패: {e}")
        log.error("오프라인 환경이라면 --model_path 에 로컬 폴더 경로를 지정하세요.")
        sys.exit(1)
    return cached


# ─── 메인 ──────────────────────────────────────────────────────────────────

def _load_config(config_path: str) -> Dict:
    """JSON 설정 파일을 읽어 dict로 반환."""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    # prompts 키: list of str 또는 list of {"name": ..., "id": ...} 두 형식 모두 허용
    if 'prompts' in cfg and isinstance(cfg['prompts'], list):
        converted = []
        for item in cfg['prompts']:
            if isinstance(item, dict):
                converted.append(f"{item['name']}:{item['id']}")
            else:
                converted.append(str(item))
        cfg['prompts'] = converted
    return cfg


def parse_args():
    # 1단계: --config 만 먼저 파싱
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None, help='JSON 설정 파일 경로')
    pre_args, remaining = pre.parse_known_args()

    # 2단계: config 파일 로드 → argparse 기본값으로 주입
    cfg_defaults: Dict = {}
    if pre_args.config:
        cfg_defaults = _load_config(pre_args.config)

    # config 에 필수 항목이 있으면 required=False 로 전환
    def _req(key: str) -> bool:
        return key not in cfg_defaults

    parser = argparse.ArgumentParser(
        description='Eagle VLM 기반 YOLOv7 자동 라벨 생성 도구',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        parents=[pre],
    )

    # 경로
    parser.add_argument('--image_dir',     required=_req('image_dir'),
                        help='JPEGImages 폴더 경로')
    parser.add_argument('--output_dir',    required=_req('output_dir'),
                        help='출력 루트 경로 (output_dir/labels/ 에 라벨 저장)')
    parser.add_argument('--model_path',    required=_req('model_path'),
                        help='로컬 모델 폴더 경로 또는 HuggingFace ID '
                             '(예: NVEagle/Eagle-X5-13B-Chat)')
    parser.add_argument('--model_save_dir', default='./models',
                        help='모델 다운로드 저장 경로 (기본: ./models)')

    # 클래스 / 프롬프트
    parser.add_argument('--prompts', nargs='+', required=_req('prompts'),
                        help='클래스 프롬프트 "클래스명:클래스ID" 형식 '
                             '(예: "person:0" "car:1" "bicycle:2")')
    parser.add_argument('--conv_mode', default='vicuna_v1',
                        help='Eagle 대화 모드 (기본: vicuna_v1)')

    # NMS
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help='Confidence 임계값 (기본: 0.5)')
    parser.add_argument('--iou_threshold',  type=float, default=0.45,
                        help='NMS IoU 임계값 (기본: 0.45)')
    parser.add_argument('--nms_mode', choices=['unified', 'per_class'],
                        default='per_class',
                        help='NMS 방식: unified=전체통합, per_class=클래스별 (기본: per_class)')

    # 디버그
    parser.add_argument('--debug_interval', type=int, default=0,
                        help='N장마다 디버그 이미지 저장 (0=비활성, 예: 1000)')
    parser.add_argument('--debug_output_dir', default='./debug_images',
                        help='디버그 이미지 저장 경로 (기본: ./debug_images)')

    # 실행 옵션
    parser.add_argument('--gpu', default='0',
                        help='사용할 GPU ID (예: "0", "0,1") (기본: 0)')
    parser.add_argument('--resume', action='store_true',
                        help='이미 라벨 파일이 존재하는 이미지 스킵')

    # config 값을 기본값으로 설정 (CLI 인자가 있으면 CLI 우선)
    parser.set_defaults(**cfg_defaults)

    return parser.parse_args(remaining)


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s %(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    log = logging.getLogger(__name__)

    # GPU 설정
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f"디바이스: {device}  (CUDA_VISIBLE_DEVICES={args.gpu})")

    # 클래스 맵 파싱
    class_map: Dict[int, str] = {}
    for p in args.prompts:
        parts = p.rsplit(':', 1)
        if len(parts) != 2:
            log.error(f"프롬프트 형식 오류 '{p}' → 'class_name:class_id' 형식으로 입력하세요.")
            sys.exit(1)
        name, cid = parts[0].strip(), int(parts[1].strip())
        class_map[cid] = name
    log.info(f"클래스 맵: {class_map}")
    log.info(f"NMS 모드: {args.nms_mode}  conf≥{args.conf_threshold}  IoU<{args.iou_threshold}")

    # 모델 로드
    model_path = resolve_model_path(args.model_path, args.model_save_dir, log)
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    log.info(f"Eagle 모델 로딩: {model_name}")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, model_name, False, False
    )
    model = model.to(device)
    log.info("모델 로드 완료.")

    # 출력 디렉토리
    labels_dir = os.path.join(args.output_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)
    if args.debug_interval > 0:
        os.makedirs(args.debug_output_dir, exist_ok=True)

    # 이미지 목록
    image_dir = Path(args.image_dir)
    image_files = sorted(
        f for f in image_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    log.info(f"이미지 {len(image_files)}장 발견: {image_dir}")

    # 통계
    total_labeled = 0
    total_boxes   = 0
    skipped       = 0

    for idx, img_path in enumerate(tqdm(image_files, desc='라벨 생성')):
        label_path = os.path.join(labels_dir, img_path.stem + '.txt')

        if args.resume and os.path.exists(label_path):
            skipped += 1
            continue

        try:
            image = Image.open(img_path).convert('RGB')
            img_w, img_h = image.size
        except Exception as e:
            log.warning(f"이미지 열기 실패 {img_path.name}: {e}")
            continue

        all_boxes: List[Tuple] = []

        for cls_id, class_name in class_map.items():
            prompt = DETECTION_PROMPT.format(class_name=class_name)
            try:
                response = infer_single(
                    model, tokenizer, image_processor,
                    image, prompt, args.conv_mode, device,
                )
                raw_boxes = parse_bbox_response(response, img_w, img_h)
                for b in raw_boxes:
                    x1, y1, x2, y2 = b['bbox']
                    all_boxes.append((x1, y1, x2, y2, b['confidence'], cls_id))
            except Exception as e:
                log.warning(f"추론 실패 {img_path.name} [{class_name}]: {e}")

        # NMS 적용
        final_boxes = apply_nms(
            all_boxes, args.nms_mode, args.conf_threshold, args.iou_threshold
        )
        total_boxes += len(final_boxes)

        # YOLO 라벨 저장
        with open(label_path, 'w') as f:
            for (x1, y1, x2, y2, _conf, cls_id) in final_boxes:
                cx, cy, w, h = xyxy_to_yolo(x1, y1, x2, y2)
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        total_labeled += 1

        # 디버그 이미지
        if args.debug_interval > 0 and (idx + 1) % args.debug_interval == 0:
            debug_path = os.path.join(
                args.debug_output_dir,
                f"debug_{idx+1:06d}_{img_path.stem}.jpg",
            )
            save_debug_image(str(img_path), final_boxes, class_map, debug_path)
            log.info(
                f"[DEBUG {idx+1}/{len(image_files)}] {img_path.name} "
                f"→ {len(final_boxes)}개 박스 | {debug_path}"
            )

    log.info("=" * 60)
    log.info(f"완료: {total_labeled}장 라벨 생성, {total_boxes}개 박스, {skipped}장 스킵")
    log.info(f"라벨 저장 경로: {labels_dir}")
    if args.debug_interval > 0:
        log.info(f"디버그 이미지: {args.debug_output_dir}")


if __name__ == '__main__':
    main()
