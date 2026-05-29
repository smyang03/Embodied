#!/usr/bin/env python3
"""
YOLOv7 Auto-Labeling Tool using LocateAnything VLM

Usage example:
  python label_generator.py \\
    --image_dir /data/JPEGImages \\
    --output_dir /data/dataset \\
    --model_path models/LocateAnything-3B \\
    --prompts "person:0" "car:1" "bicycle:2" \\
    --iou_threshold 0.45 \\
    --nms_mode per_class \\
    --max_input_size 1024 \\
    --debug_interval 1000 \\
    --gpu 0

  # JSON 설정 파일 사용
  python label_generator.py --config label_config.example.json

  # 설정 파일 + CLI 덮어쓰기
  python label_generator.py --config label_config.example.json --gpu 1 --max_input_size 768
"""

import os
import sys
import re
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# ── 환경변수는 torch/CUDA 초기화 전에 설정해야 확실히 적용됨 ──────────────
# parse_args()를 먼저 호출해 GPU, expandable_segments 값을 확보한 뒤 설정
def _apply_env_early():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config',              default=None)
    pre.add_argument('--gpu',                 default='0')
    pre.add_argument('--expandable_segments', action='store_true')
    known, _ = pre.parse_known_args()

    # config 파일에서도 읽을 수 있도록
    gpu = known.gpu
    expandable = known.expandable_segments
    if not expandable and known.config:
        try:
            with open(known.config) as f:
                cfg = json.load(f)
            gpu = str(cfg.get('gpu', gpu))
            expandable = bool(cfg.get('expandable_segments', expandable))
        except Exception:
            pass

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    if expandable:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

_apply_env_early()
# ──────────────────────────────────────────────────────────────────────────

import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

# locateanything_worker.py 는 이 파일과 같은 디렉토리에 있어야 함
sys.path.insert(0, str(Path(__file__).resolve().parent))
from locateanything_worker import LocateAnythingWorker


# ─── 상수 ──────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

BOX_COLORS = [
    'red', 'lime', 'blue', 'yellow', 'cyan', 'magenta',
    'orange', 'purple', 'pink', 'brown', 'coral', 'gold',
]


# ─── 이미지 리사이즈 ────────────────────────────────────────────────────────

def resize_for_inference(image: Image.Image, max_size: int) -> Image.Image:
    """장변이 max_size를 넘으면 비율 유지하며 축소. 넘지 않으면 원본 반환."""
    if max_size <= 0:
        return image
    if max(image.size) <= max_size:
        return image
    img = image.copy()
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    return img


# ─── 박스 파싱 ─────────────────────────────────────────────────────────────

def parse_boxes_normalized(answer: str) -> List[Tuple[float, float, float, float]]:
    """
    LocateAnything 응답에서 <box><x1><y1><x2><y2></box> 태그를 추출한다.
    값은 0~1000 스케일 → 0.0~1.0 정규화 좌표로 변환.
    """
    boxes = []
    for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
        x1, y1, x2, y2 = (int(m.group(i)) / 1000.0 for i in range(1, 5))
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


# ─── 좌표 변환 ─────────────────────────────────────────────────────────────

def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float) -> Tuple:
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


# ─── NMS ───────────────────────────────────────────────────────────────────

def _iou(a: Tuple, b: Tuple) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _nms_single(boxes: List, iou_thr: float) -> List:
    """boxes: list of (x1,y1,x2,y2, conf, cls_id)"""
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if _iou(best[:4], b[:4]) < iou_thr]
    return keep


def apply_nms(all_boxes: List, nms_mode: str, iou_thr: float) -> List:
    if not all_boxes:
        return []
    if nms_mode == 'unified':
        return _nms_single(all_boxes, iou_thr)
    result = []
    for cls_id in {b[5] for b in all_boxes}:
        result.extend(_nms_single([b for b in all_boxes if b[5] == cls_id], iou_thr))
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

    for (x1, y1, x2, y2, _conf, cls_id) in boxes:
        color = BOX_COLORS[cls_id % len(BOX_COLORS)]
        px1, py1 = int(x1 * iw), int(y1 * ih)
        px2, py2 = int(x2 * iw), int(y2 * ih)
        draw.rectangle([px1, py1, px2, py2], outline=color, width=3)
        label = class_map.get(cls_id, str(cls_id))
        draw.rectangle([px1, max(0, py1 - 18), px1 + len(label) * 7 + 4, py1], fill=color)
        draw.text((px1 + 2, max(0, py1 - 17)), label, fill='white')

    img.save(output_path, quality=92)


# ─── 모델 로드 ─────────────────────────────────────────────────────────────

def resolve_model_path(model_path: str, model_save_dir: str, log: logging.Logger,
                       hf_token: str = '') -> str:
    """로컬 경로가 존재하면 그대로 사용, 없으면 HuggingFace에서 다운로드."""
    if os.path.isdir(model_path):
        log.info(f"로컬 모델 경로 사용: {model_path}")
        return model_path

    safe_name = model_path.replace('/', '__')
    cached = os.path.join(model_save_dir, safe_name)
    if os.path.isdir(cached):
        log.info(f"캐시된 모델 사용: {cached}")
        return cached

    log.info(f"'{model_path}' 다운로드 중 → {cached}")
    os.makedirs(model_save_dir, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_path, local_dir=cached,
                          token=hf_token or None)
        log.info(f"모델 저장 완료: {cached}")
    except Exception as e:
        log.error(f"모델 다운로드 실패: {e}")
        log.error("오프라인 환경이라면 --model_path 에 로컬 폴더 경로를 지정하세요.")
        sys.exit(1)
    return cached


# ─── 설정 파일 ─────────────────────────────────────────────────────────────

def _load_config(config_path: str) -> Dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    # prompts: [{"name":..,"id":..}] 또는 ["name:id"] 두 형식 모두 허용
    if 'prompts' in cfg and isinstance(cfg['prompts'], list):
        converted = []
        for item in cfg['prompts']:
            if isinstance(item, dict):
                converted.append(f"{item['name']}:{item['id']}")
            else:
                converted.append(str(item))
        cfg['prompts'] = converted
    return cfg


# ─── 인자 파싱 ─────────────────────────────────────────────────────────────

def parse_args():
    # 1단계: --config 만 먼저 파싱
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None, help='JSON 설정 파일 경로')
    pre_args, remaining = pre.parse_known_args()

    cfg_defaults: Dict = {}
    if pre_args.config:
        cfg_defaults = _load_config(pre_args.config)

    def _req(key: str) -> bool:
        return key not in cfg_defaults

    parser = argparse.ArgumentParser(
        description='LocateAnything 기반 YOLOv7 자동 라벨 생성 도구',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        parents=[pre],
    )

    # 경로
    parser.add_argument('--image_dir',      required=_req('image_dir'),
                        help='JPEGImages 폴더 경로')
    parser.add_argument('--output_dir',     required=_req('output_dir'),
                        help='출력 루트 경로 (output_dir/labels/ 에 라벨 저장)')
    parser.add_argument('--model_path',     required=_req('model_path'),
                        help='로컬 모델 폴더 경로 또는 HuggingFace ID '
                             '(예: ServiceNow-AI/LocateAnything-3B)')
    parser.add_argument('--model_save_dir', default='./models',
                        help='모델 다운로드 저장 경로 (기본: ./models)')

    # 클래스
    parser.add_argument('--prompts', nargs='+', required=_req('prompts'),
                        help='클래스 정의 "클래스명:클래스ID" 형식 '
                             '(예: "person:0" "car:1" "bicycle:2")')

    # 이미지 입력 크기
    parser.add_argument('--max_input_size', type=int, default=1024,
                        help='추론 전 이미지 장변 최대 크기 px (OOM 방지, 0=리사이즈 없음, 기본: 1024)')

    # NMS
    parser.add_argument('--iou_threshold',  type=float, default=0.45,
                        help='NMS IoU 임계값 (기본: 0.45)')
    parser.add_argument('--nms_mode',       choices=['unified', 'per_class'],
                        default='per_class',
                        help='NMS 방식: unified=전체통합, per_class=클래스별 (기본: per_class)')

    # 디버그
    parser.add_argument('--debug_interval',   type=int, default=0,
                        help='N장마다 디버그 이미지 저장 (0=비활성, 예: 1000)')
    parser.add_argument('--debug_output_dir', default='./debug_images',
                        help='디버그 이미지 저장 경로 (기본: ./debug_images)')

    # 실행 옵션
    parser.add_argument('--gpu',    default='0',
                        help='사용할 GPU ID (예: "0", "0,1") (기본: 0)')
    parser.add_argument('--resume', action='store_true',
                        help='이미 라벨 파일이 존재하는 이미지 스킵')
    parser.add_argument('--expandable_segments', action='store_true',
                        help='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 설정 (메모리 단편화 완화)')

    # 추론 파라미터
    parser.add_argument('--generation_mode', default='hybrid',
                        choices=['hybrid', 'ar', 'nar'],
                        help='LocateAnything 생성 모드 (기본: hybrid)')
    parser.add_argument('--temperature',  type=float, default=0.7,
                        help='샘플링 온도 (기본: 0.7)')
    parser.add_argument('--max_new_tokens', type=int, default=8192,
                        help='최대 생성 토큰 수 (기본: 8192)')

    parser.set_defaults(**cfg_defaults)
    return parser.parse_args(remaining)


# ─── 메인 ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s %(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    log = logging.getLogger(__name__)

    # 환경변수는 _apply_env_early() 에서 torch import 전에 이미 적용됨
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f"디바이스: {device}  (GPU: {args.gpu})")
    if args.expandable_segments:
        log.info("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 적용됨")
    if args.max_input_size > 0:
        log.info(f"이미지 입력 크기 제한: 장변 최대 {args.max_input_size}px")

    # 클래스 맵 파싱: {cls_id: class_name}
    class_map: Dict[int, str] = {}
    for p in args.prompts:
        parts = p.rsplit(':', 1)
        if len(parts) != 2:
            log.error(f"프롬프트 형식 오류 '{p}' → 'class_name:class_id' 형식으로 입력하세요.")
            sys.exit(1)
        name, cid = parts[0].strip(), int(parts[1].strip())
        class_map[cid] = name

    # 클래스 ID 순서대로 정렬된 리스트 (detect 호출용)
    sorted_classes = sorted(class_map.items())  # [(cls_id, name), ...]
    category_names = [name for _, name in sorted_classes]
    category_ids   = [cid  for cid,  _ in sorted_classes]

    log.info(f"클래스: {class_map}")
    log.info(f"NMS 모드: {args.nms_mode}  IoU<{args.iou_threshold}")

    # 모델 로드
    hf_token = getattr(args, 'hf_token', '') or ''
    model_path = resolve_model_path(args.model_path, args.model_save_dir, log, hf_token)
    log.info(f"LocateAnything 모델 로딩: {model_path}")
    worker = LocateAnythingWorker(model_path=model_path, device=device)
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

    total_labeled = 0
    total_boxes   = 0
    skipped       = 0

    for idx, img_path in enumerate(tqdm(image_files, desc='라벨 생성')):
        label_path = os.path.join(labels_dir, img_path.stem + '.txt')

        if args.resume and os.path.exists(label_path):
            skipped += 1
            continue

        try:
            original_image = Image.open(img_path).convert('RGB')
        except Exception as e:
            log.warning(f"이미지 열기 실패 {img_path.name}: {e}")
            continue

        # 추론용 리사이즈 (라벨 좌표는 0~1 정규화라서 원본 크기 무관)
        infer_image = resize_for_inference(original_image, args.max_input_size)
        if infer_image.size != original_image.size:
            log.debug(f"  리사이즈: {original_image.size} → {infer_image.size}")

        # 클래스별 개별 detect 호출 → cls_id 정확히 매핑
        # (LocateAnything 다중 카테고리 응답은 박스-클래스 순서를 보장하지 않음)
        all_boxes: List[Tuple] = []

        for cls_id, class_name in class_map.items():
            try:
                r = worker.detect(
                    infer_image,
                    [class_name],
                    generation_mode=args.generation_mode,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    verbose=False,
                )
                raw = parse_boxes_normalized(r.get('answer', ''))
                for (x1, y1, x2, y2) in raw:
                    all_boxes.append((x1, y1, x2, y2, 1.0, cls_id))
            except torch.cuda.OutOfMemoryError:
                log.warning(
                    f"OOM 발생: {img_path.name} [{class_name}] "
                    f"(max_input_size={args.max_input_size}). "
                    "--max_input_size 를 줄이거나 --expandable_segments 를 추가하세요."
                )
                torch.cuda.empty_cache()
            except Exception as e:
                log.warning(f"추론 실패 {img_path.name} [{class_name}]: {e}")

        # NMS 적용 (LocateAnything은 confidence 없으므로 conf=1.0 고정, IoU만 사용)
        final_boxes = apply_nms(all_boxes, args.nms_mode, args.iou_threshold)
        total_boxes += len(final_boxes)

        # YOLO 라벨 저장
        with open(label_path, 'w') as f:
            for (x1, y1, x2, y2, _conf, cls_id) in final_boxes:
                cx, cy, w, h = xyxy_to_yolo(x1, y1, x2, y2)
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        total_labeled += 1

        # 디버그 시각화 (원본 이미지에 박스 드로잉)
        if args.debug_interval > 0 and (idx + 1) % args.debug_interval == 0:
            debug_path = os.path.join(
                args.debug_output_dir,
                f"debug_{idx+1:06d}_{img_path.stem}.jpg",
            )
            save_debug_image(str(img_path), final_boxes, class_map, debug_path)
            log.info(
                f"[DEBUG {idx+1}/{len(image_files)}] {img_path.name} "
                f"({original_image.size[0]}×{original_image.size[1]}) "
                f"→ {len(final_boxes)}개 박스 | {debug_path}"
            )

    log.info("=" * 60)
    log.info(f"완료: {total_labeled}장 라벨 생성, {total_boxes}개 박스, {skipped}장 스킵")
    log.info(f"라벨 저장 경로: {labels_dir}")
    if args.debug_interval > 0:
        log.info(f"디버그 이미지: {args.debug_output_dir}")


if __name__ == '__main__':
    main()
