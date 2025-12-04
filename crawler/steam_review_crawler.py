import os
import csv
import time
from typing import List, Dict

import requests


def fetch_reviews_for_app(
    app_id: int,
    max_reviews: int = 5000,
    language: str = "koreana",
    delay: float = 1.0,
) -> List[Dict]:
    """
    Steam 특정 게임(app_id)에 대해 한글 리뷰를 최대 max_reviews개까지 가져오는 함수.
    - Steam 리뷰 API 엔드포인트 사용
    - language='koreana'로 설정해서 한국어 리뷰만 수집
    - voted_up (추천 여부)를 label(1/0)로 변환
    """
    url = f"https://store.steampowered.com/appreviews/{app_id}"
    cursor = "*"
    reviews: List[Dict] = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    }

    while len(reviews) < max_reviews:
        params = {
            "json": 1,
            "filter": "recent",       # 최근 리뷰 기준
            "language": language,     # 한국어 리뷰만
            "review_type": "all",     # 긍/부정 모두
            "purchase_type": "all",   # 모든 구매 유형
            "num_per_page": 100,      # 한 번에 100개 요청
            "cursor": cursor,
        }

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
        except Exception as e:
            print(f"[ERROR] 앱 {app_id} 요청 중 에러 발생: {e}")
            break

        if resp.status_code != 200:
            print(f"[ERROR] 앱 {app_id} HTTP 상태 코드: {resp.status_code}")
            break

        data = resp.json()
        fetched = data.get("reviews", [])
        if not fetched:
            print(f"[INFO] 앱 {app_id} 더 이상 가져올 리뷰가 없습니다.")
            break

        for r in fetched:
            # 혹시 language 필드가 있을 경우, 한 번 더 한국어인지 체크 (안전용)
            if r.get("language") not in (None, "", language):
                continue

            text = (r.get("review") or "").strip()
            if not text:
                continue

            voted_up = bool(r.get("voted_up", False))
            label = 1 if voted_up else 0  # 1: 추천(긍정), 0: 비추천(부정)

            review_data = {
                "app_id": app_id,
                "recommend": voted_up,
                "review": text,
                "timestamp_created": r.get("timestamp_created", None),
                "label": label,
            }
            reviews.append(review_data)

            if len(reviews) >= max_reviews:
                break

        cursor = data.get("cursor")
        if not cursor:
            print(f"[INFO] 앱 {app_id} cursor가 더 이상 없습니다. 종료.")
            break

        print(f"[INFO] 앱 {app_id} 현재 수집 리뷰 수: {len(reviews)}")
        time.sleep(delay)

    print(f"[INFO] 앱 {app_id} 최종 수집 리뷰 수: {len(reviews)}")
    return reviews


def crawl_multiple_games(
    app_ids: List[int],
    max_reviews_per_game: int,
    output_path: str,
) -> None:
    """
    여러 개의 Steam 게임(app_ids)에 대해 리뷰를 수집하고
    하나의 CSV 파일(output_path)에 저장하는 함수.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = ["id", "app_id", "recommend", "review", "timestamp_created", "label"]

    global_id = 0
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for app_id in app_ids:
            print(f"\n[INFO] 앱 {app_id} 크롤링 시작")
            game_reviews = fetch_reviews_for_app(
                app_id=app_id,
                max_reviews=max_reviews_per_game,
                language="koreana",
            )

            for r in game_reviews:
                row = {
                    "id": global_id,
                    "app_id": r["app_id"],
                    "recommend": r["recommend"],
                    "review": r["review"],
                    "timestamp_created": r["timestamp_created"],
                    "label": r["label"],  # 1: 추천(긍정), 0: 비추천(부정)
                }
                writer.writerow(row)
                global_id += 1

    print(f"\n[INFO] 전체 크롤링 완료! 총 수집 리뷰 수: {global_id}")
    print(f"[INFO] CSV 저장 경로: {output_path}")


if __name__ == "__main__":
    # ✅ 인기 있는 스팀 게임들 appid (한국어 리뷰도 많은 편)
    APP_IDS = [
        570,      # Dota 2
        730,      # Counter-Strike 2
        578080,   # PUBG: BATTLEGROUNDS
        1599340,  # LOST ARK
        582660,   # Black Desert
        1172470,  # Apex Legends
    ]

    # 게임당 최대 리뷰 수 (예: 5000개씩 × 6게임 = 최대 3만 개)
    MAX_REVIEWS_PER_GAME = 5000

    # 프로젝트 루트 기준: data/raw/steam_reviews_raw.csv
    OUTPUT_PATH = os.path.join("data", "raw", "steam_reviews_raw.csv")

    crawl_multiple_games(APP_IDS, MAX_REVIEWS_PER_GAME, OUTPUT_PATH)
