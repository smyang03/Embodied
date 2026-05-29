#!/usr/bin/env python3
"""
Model Manager - HuggingFace 모델 다운로드 & 로컬 관리 도구

사용법:
  python model_manager.py                        # 대화형 메뉴
  python model_manager.py list                   # 추천 모델 목록 출력
  python model_manager.py local                  # 로컬 저장된 모델 목록
  python model_manager.py download <model_id>    # 모델 직접 다운로드
  python model_manager.py download <model_id> --save_dir ./models
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Optional

# ─── 추천 모델 목록 ────────────────────────────────────────────────────────
# 새 모델 추가 시 이 리스트에만 항목을 넣으면 됩니다.

RECOMMENDED_MODELS = [
    {
        "id":          "ServiceNow-AI/LocateAnything-3B",
        "name":        "LocateAnything-3B",
        "category":    "Grounding / Detection",
        "size":        "~7 GB",
        "description": "오픈소스 시각적 그라운딩 모델. <box> 태그로 bbox 직접 출력. "
                       "label_generator.py 기본 백엔드.",
    },
    {
        "id":          "NVEagle/Eagle-X5-13B-Chat",
        "name":        "Eagle-X5-13B-Chat",
        "category":    "VLM (Vision-Language)",
        "size":        "~26 GB",
        "description": "NVIDIA Eagle 13B 채팅 모델. 고해상도 이미지 이해.",
    },
    {
        "id":          "NVEagle/Eagle-X5-7B-Chat",
        "name":        "Eagle-X5-7B-Chat",
        "category":    "VLM (Vision-Language)",
        "size":        "~14 GB",
        "description": "NVIDIA Eagle 7B 경량 버전.",
    },
    {
        "id":          "NVEagle/Eagle2-9B",
        "name":        "Eagle2-9B",
        "category":    "VLM (Vision-Language)",
        "size":        "~18 GB",
        "description": "Eagle2 시리즈 9B. 멀티이미지 지원.",
    },
    {
        "id":          "NVEagle/Eagle2-8B",
        "name":        "Eagle2-8B",
        "category":    "VLM (Vision-Language)",
        "size":        "~16 GB",
        "description": "Eagle2 시리즈 8B 경량 버전.",
    },
    {
        "id":          "Qwen/Qwen2.5-VL-7B-Instruct",
        "name":        "Qwen2.5-VL-7B-Instruct",
        "category":    "VLM (Vision-Language)",
        "size":        "~15 GB",
        "description": "Qwen2.5 비전-언어 7B. 객체 탐지/그라운딩 지원.",
    },
    {
        "id":          "Qwen/Qwen2.5-VL-3B-Instruct",
        "name":        "Qwen2.5-VL-3B-Instruct",
        "category":    "VLM (Vision-Language)",
        "size":        "~6 GB",
        "description": "Qwen2.5 비전-언어 3B 경량 버전.",
    },
    {
        "id":          "microsoft/Florence-2-large",
        "name":        "Florence-2-large",
        "category":    "Grounding / Detection",
        "size":        "~1.5 GB",
        "description": "Microsoft Florence-2. 객체 탐지·캡션·OCR 등 다양한 태스크 지원. 초경량.",
    },
    {
        "id":          "microsoft/Florence-2-base",
        "name":        "Florence-2-base",
        "category":    "Grounding / Detection",
        "size":        "~0.9 GB",
        "description": "Florence-2 base 버전. 매우 빠른 추론.",
    },
]

# ─── 유틸 ──────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"


def _c(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def _dir_size_gb(path: str) -> float:
    total = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
    return total / (1024 ** 3)


def _id_to_dirname(model_id: str) -> str:
    return model_id.replace('/', '__')


def _dirname_to_id(dirname: str) -> str:
    # ServiceNow-AI__LocateAnything-3B → ServiceNow-AI/LocateAnything-3B
    # 첫 번째 __ 만 / 로 복원 (모델명 내 __ 은 유지)
    parts = dirname.split('__', 1)
    return '/'.join(parts) if len(parts) == 2 else dirname


# ─── 기능 함수 ─────────────────────────────────────────────────────────────

def print_recommended(highlight: Optional[int] = None) -> None:
    """추천 모델 목록 출력."""
    print()
    print(_c("  추천 모델 목록", BOLD + CYAN))
    print("  " + "─" * 72)
    print(f"  {'#':<3} {'이름':<30} {'카테고리':<22} {'크기':<8} 설명")
    print("  " + "─" * 72)

    for i, m in enumerate(RECOMMENDED_MODELS, 1):
        marker = _c(f"{i:>2}.", YELLOW + BOLD) if i == highlight else f"{i:>2}."
        name   = _c(m['name'], GREEN) if i == highlight else m['name']
        print(f"  {marker} {name:<30} {m['category']:<22} {m['size']:<8} {_c(m['description'], DIM)}")
    print()


def list_local(save_dir: str) -> list:
    """로컬에 저장된 모델 목록 반환 및 출력."""
    base = Path(save_dir)
    if not base.exists():
        print(_c(f"\n  저장 경로 없음: {save_dir}", YELLOW))
        return []

    dirs = [d for d in sorted(base.iterdir()) if d.is_dir()]
    if not dirs:
        print(_c(f"\n  로컬 모델 없음: {save_dir}", YELLOW))
        return []

    print()
    print(_c(f"  로컬 저장 모델  ({save_dir})", BOLD + CYAN))
    print("  " + "─" * 60)
    local_ids = []
    for i, d in enumerate(dirs, 1):
        model_id = _dirname_to_id(d.name)
        try:
            size_gb = _dir_size_gb(str(d))
            size_str = f"{size_gb:.1f} GB"
        except Exception:
            size_str = "?"
        print(f"  {i:>2}. {_c(model_id, GREEN):<40} {size_str}")
        local_ids.append(model_id)
    print()
    return local_ids


def download_model(model_id: str, save_dir: str) -> str:
    """HuggingFace에서 모델 다운로드. 이미 있으면 스킵."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(_c("\n  [오류] huggingface_hub 가 설치되지 않았습니다.", RED))
        print("  pip install huggingface-hub")
        sys.exit(1)

    dest = os.path.join(save_dir, _id_to_dirname(model_id))

    if os.path.isdir(dest) and any(Path(dest).iterdir()):
        print(_c(f"\n  이미 로컬에 존재: {dest}", YELLOW))
        ans = input("  덮어쓰기(재다운로드)할까요? [y/N] ").strip().lower()
        if ans != 'y':
            print("  스킵.")
            return dest
        shutil.rmtree(dest)

    os.makedirs(dest, exist_ok=True)
    print(_c(f"\n  다운로드 시작: {model_id}", CYAN))
    print(f"  저장 경로: {dest}\n")

    try:
        snapshot_download(repo_id=model_id, local_dir=dest)
        size_gb = _dir_size_gb(dest)
        print(_c(f"\n  완료: {dest}  ({size_gb:.1f} GB)", GREEN))
        return dest
    except Exception as e:
        print(_c(f"\n  [오류] 다운로드 실패: {e}", RED))
        sys.exit(1)


def delete_model(save_dir: str) -> None:
    """로컬 모델 삭제 (대화형)."""
    local = list_local(save_dir)
    if not local:
        return

    raw = input("  삭제할 번호 입력 (취소: Enter): ").strip()
    if not raw:
        return
    try:
        idx = int(raw) - 1
        model_id = local[idx]
    except (ValueError, IndexError):
        print(_c("  잘못된 번호.", RED))
        return

    dest = os.path.join(save_dir, _id_to_dirname(model_id))
    ans = input(f"  {_c(model_id, YELLOW)} 을(를) 삭제할까요? [y/N] ").strip().lower()
    if ans == 'y':
        shutil.rmtree(dest)
        print(_c(f"  삭제 완료: {dest}", GREEN))


def interactive_menu(save_dir: str) -> None:
    """대화형 메인 메뉴."""
    while True:
        print()
        print(_c("  ╔══════════════════════════════╗", CYAN))
        print(_c("  ║     Model Manager            ║", CYAN + BOLD))
        print(_c("  ╚══════════════════════════════╝", CYAN))
        print(f"  저장 경로: {_c(save_dir, DIM)}")
        print()
        print("  1. 추천 모델 목록 보기 & 다운로드")
        print("  2. 직접 모델 ID 입력 & 다운로드")
        print("  3. 로컬 모델 목록 보기")
        print("  4. 로컬 모델 삭제")
        print("  0. 종료")
        print()

        choice = input("  선택: ").strip()

        if choice == '1':
            print_recommended()
            raw = input("  다운로드할 번호 입력 (취소: Enter): ").strip()
            if not raw:
                continue
            try:
                idx = int(raw) - 1
                model = RECOMMENDED_MODELS[idx]
            except (ValueError, IndexError):
                print(_c("  잘못된 번호.", RED))
                continue
            download_model(model['id'], save_dir)

        elif choice == '2':
            print()
            model_id = input("  HuggingFace 모델 ID (예: ServiceNow-AI/LocateAnything-3B): ").strip()
            if not model_id:
                continue
            download_model(model_id, save_dir)

        elif choice == '3':
            list_local(save_dir)

        elif choice == '4':
            delete_model(save_dir)

        elif choice == '0':
            print("  종료.")
            break
        else:
            print(_c("  잘못된 선택.", RED))


# ─── 서브커맨드 핸들러 ──────────────────────────────────────────────────────

def cmd_list(_args) -> None:
    print_recommended()


def cmd_local(args) -> None:
    list_local(args.save_dir)


def cmd_download(args) -> None:
    download_model(args.model_id, args.save_dir)


def cmd_interactive(args) -> None:
    interactive_menu(args.save_dir)


# ─── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='HuggingFace 모델 다운로드 & 로컬 관리 도구',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--save_dir', default='./models',
        help='모델 저장 루트 경로 (기본: ./models)',
    )

    sub = parser.add_subparsers(dest='command')

    sub.add_parser('list',  help='추천 모델 목록 출력')
    sub.add_parser('local', help='로컬 저장 모델 목록 출력')

    dl = sub.add_parser('download', help='모델 다운로드')
    dl.add_argument('model_id', help='HuggingFace 모델 ID (예: ServiceNow-AI/LocateAnything-3B)')

    args = parser.parse_args()

    dispatch = {
        'list':     cmd_list,
        'local':    cmd_local,
        'download': cmd_download,
        None:       cmd_interactive,
    }
    dispatch[args.command](args)


if __name__ == '__main__':
    main()
