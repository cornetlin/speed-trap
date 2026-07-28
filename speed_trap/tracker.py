"""Vehicle track bookkeeping and best-frame selection.

每台車在畫面裡會出現數十幀,最後只有一幀會送去 OCR。挑哪一幀決定了整個
系統讀不讀得出車牌,所以挑選規則寫在這裡,並且分成兩層:

1. 硬性排除:bbox 碰到畫面邊界的幀直接不列入候選。車頭一旦出框,車牌就
   被裁掉一半,OCR 只會拿到一兩個字元。這種幀不能靠權重稀釋 —— 面積權重
   反而會偏好它們(車子快出框時投影面積最大),必須直接排除。
2. 加權評分:剩下的候選幀之間,才比面積、置中程度與清晰度。

保護機制:低幀率時可能整條 track 每一幀都碰到邊界。這種情況不放棄,退而
取「最不靠近邊界」的那一幀送 OCR,並標記為需人工複核。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from speed_trap.config import StationConfig

STALE_TIMEOUT_NS: int = 5 * 1_000_000_000

_MAX_CENTER_DIST: float = math.hypot(0.5, 0.5)


@dataclass(frozen=True)
class Detection:
    track_id: int
    label: str
    bbox: tuple[float, float, float, float]
    confidence: float
    frame_ns: int
    frame_jpeg: bytes | None


@dataclass(frozen=True)
class ScoringParams:
    """最佳幀挑選的參數。預設值等同改版前的行為,實際值由 config 提供。"""

    area_weight: float = 0.6
    center_weight: float = 0.3
    sharpness_weight: float = 0.1
    edge_margin: float = 0.02

    @classmethod
    def from_config(cls, config: StationConfig) -> ScoringParams:
        return cls(
            area_weight=config.score_area_weight,
            center_weight=config.score_center_weight,
            sharpness_weight=config.score_sharpness_weight,
            edge_margin=config.edge_margin,
        )


DEFAULT_SCORING = ScoringParams()


def edge_distance(bbox: tuple[float, float, float, float]) -> float:
    """bbox 離最近的畫面邊界有多遠(正規化單位)。

    0 表示剛好貼齊邊界,負值表示已經超出畫面。四個邊取最小值,所以只要有
    任何一邊接近邊界,回傳值就會小。
    """
    x1, y1, x2, y2 = bbox
    return min(x1, y1, 1.0 - x2, 1.0 - y2)


def touches_edge(
    bbox: tuple[float, float, float, float], margin: float = DEFAULT_SCORING.edge_margin
) -> bool:
    return edge_distance(bbox) < margin


def score_detection(
    det: Detection, params: ScoringParams = DEFAULT_SCORING
) -> float:
    """候選幀的加權分數。只在通過邊界排除之後才有意義。"""
    x1, y1, x2, y2 = det.bbox
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    dist = math.hypot(cx - 0.5, cy - 0.5)
    center_closeness = max(0.0, 1.0 - dist / _MAX_CENTER_DIST)

    sharpness_placeholder = 0.0

    return (
        params.area_weight * area
        + params.center_weight * center_closeness
        + params.sharpness_weight * sharpness_placeholder
    )


@dataclass(frozen=True)
class BestFrame:
    """最後選中要送 OCR 的那一幀,連同挑它的理由。"""

    detection: Detection
    frame_index: int          # 這台車的第幾幀(1-based)
    score: float
    edge_distance: float
    from_edge_fallback: bool  # True = 沒有任何乾淨的幀,這是退而求其次的結果


@dataclass
class VehicleTrack:
    track_id: int
    label: str
    first_seen_ns: int
    last_seen_ns: int
    frame_count: int = 0
    edge_frame_count: int = 0
    # 通過邊界排除的最佳幀
    best_detection: Detection | None = None
    best_score: float = 0.0
    best_frame_index: int = -1
    # 保護機制:所有幀都碰到邊界時,留下最不靠邊的那一幀
    fallback_detection: Detection | None = None
    fallback_edge_distance: float = -math.inf
    fallback_frame_index: int = -1
    emitted: bool = False

    def best_frame(self) -> BestFrame | None:
        """要送 OCR 的那一幀。整條 track 都碰邊界時回傳 fallback。"""
        if self.best_detection is not None:
            return BestFrame(
                detection=self.best_detection,
                frame_index=self.best_frame_index,
                score=self.best_score,
                edge_distance=edge_distance(self.best_detection.bbox),
                from_edge_fallback=False,
            )
        if self.fallback_detection is not None:
            return BestFrame(
                detection=self.fallback_detection,
                frame_index=self.fallback_frame_index,
                score=0.0,
                edge_distance=self.fallback_edge_distance,
                from_edge_fallback=True,
            )
        return None


class VehicleTracker:
    def __init__(
        self,
        stale_timeout_ns: int = STALE_TIMEOUT_NS,
        scoring: ScoringParams = DEFAULT_SCORING,
    ) -> None:
        self._stale_timeout_ns = stale_timeout_ns
        self._scoring = scoring
        self._tracks: dict[int, VehicleTrack] = {}

    @property
    def scoring(self) -> ScoringParams:
        return self._scoring

    def update(self, detections: list[Detection]) -> None:
        for det in detections:
            track = self._tracks.get(det.track_id)
            if track is None:
                track = VehicleTrack(
                    track_id=det.track_id,
                    label=det.label,
                    first_seen_ns=det.frame_ns,
                    last_seen_ns=det.frame_ns,
                )
                self._tracks[det.track_id] = track

            track.last_seen_ns = det.frame_ns
            track.frame_count += 1
            index = track.frame_count

            distance = edge_distance(det.bbox)
            if distance < self._scoring.edge_margin:
                # 碰到邊界 —— 不列入候選,但留著當保護機制的後備,
                # 越不靠邊的越優先。
                track.edge_frame_count += 1
                if distance > track.fallback_edge_distance:
                    track.fallback_detection = det
                    track.fallback_edge_distance = distance
                    track.fallback_frame_index = index
                continue

            score = score_detection(det, self._scoring)
            if track.best_detection is None or score > track.best_score:
                track.best_detection = det
                track.best_score = score
                track.best_frame_index = index

    def get_track(self, track_id: int) -> VehicleTrack | None:
        return self._tracks.get(track_id)

    def prune_stale(self, now_ns: int) -> list[VehicleTrack]:
        """移除並回傳已經離開畫面的 track,供呼叫端結算。"""
        stale = [
            track
            for track in self._tracks.values()
            if now_ns - track.last_seen_ns > self._stale_timeout_ns
        ]
        for track in stale:
            del self._tracks[track.track_id]
        return stale

    def drain(self) -> list[VehicleTrack]:
        """收工時把還在畫面裡的 track 全部倒出來結算。"""
        tracks = list(self._tracks.values())
        self._tracks.clear()
        return tracks

    def all_active(self) -> list[VehicleTrack]:
        return list(self._tracks.values())
