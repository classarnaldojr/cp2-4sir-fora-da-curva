"""
Monitor de atenção / fadiga (estudo ou direção simulada) — ficheiro único.
OpenCV + MediaPipe Face Landmarker (tasks, modo VIDEO).

Teclas: [c] calibrar olhando à tela (~1,5 s) | [q] sair

Ideia do pipeline (visão geral):
- entrada: frame da webcam (OpenCV / BGR)
- pré-processamento: converter para RGB e empacotar como mp.Image
- processamento: Face Landmarker → landmarks + (opcional) blendshapes
- métricas simples (explicáveis):
  - EAR (Eye Aspect Ratio): indica olho aberto/fechado
  - MAR (Mouth Aspect Ratio): indica boca aberta (bocejo)
  - pose (pitch/yaw/roll): direção da cabeça (olhando para baixo / para o lado)
- decisão por tempo: não “um frame”, mas segundos acumulados em condição
- saída: HUD (nível + score) + malha (opcional) + alarme no vermelho
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np

from facemesh_conexoes import (
    FACEMESH_FACE_OVAL,
    FACEMESH_LEFT_EYE,
    FACEMESH_LEFT_IRIS,
    FACEMESH_LIPS,
    FACEMESH_NOSE,
    FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_IRIS,
)

# =============================================================================
# Divisão sugerida para apresentação (3 alunos) — ver README.md
# -----------------------------------------------------------------------------
# Lucas: captura (webcam), modelo .task, BGR→RGB, Face Landmarker, detect_for_video
# Rafael: métricas EAR/MAR/pose, calibração [c], oclusão, timers, níveis e score
# Pedro: malha no rosto, HUD, alarme, janela OpenCV, argumentos (--no-mesh, etc.)  (parte mais fácil)
# =============================================================================

# ---------------------------------------------------------------------------
# Rafael — Métricas e regras (EAR, MAR, pose da cabeça, oclusão)
# ---------------------------------------------------------------------------

RIGHT_EYE_IDX = (33, 160, 158, 133, 153, 144)
LEFT_EYE_IDX = (362, 385, 387, 263, 373, 380)
MOUTH_VERTICAL = (13, 14)
MOUTH_WIDTH = (61, 291)
POSE_NOSE_TIP = 1
POSE_CHIN = 152
POSE_LEFT_EYE = 263
POSE_RIGHT_EYE = 33
POSE_LEFT_MOUTH = 61
POSE_RIGHT_MOUTH = 291

MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)


def landmarks_para_pixels(landmarks: Any, width: int, height: int) -> np.ndarray:
    """Converte landmarks normalizados (0..1) para coordenadas em pixels."""
    pts = np.zeros((len(landmarks), 2), dtype=np.float64)
    for i, lm in enumerate(landmarks):
        pts[i, 0] = lm.x * width
        pts[i, 1] = lm.y * height
    return pts


def _euclid(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def razao_aspecto_olho(pts: np.ndarray, indices: tuple[int, ...]) -> float:
    """
    EAR (Eye Aspect Ratio).

    Intuição: mede a “abertura” do olho pela razão entre duas distâncias verticais
    (pálpebra superior ↔ inferior) e a distância horizontal do olho.
    - Olho aberto → EAR maior
    - Olho fechado → EAR menor
    """
    a = _euclid(pts[indices[1]], pts[indices[5]])
    b = _euclid(pts[indices[2]], pts[indices[4]])
    c = _euclid(pts[indices[0]], pts[indices[3]])
    if c < 1e-6:
        return 0.0
    return (a + b) / (2.0 * c)


def razao_aspecto_boca(pts: np.ndarray) -> float:
    """
    MAR (Mouth Aspect Ratio).

    Intuição: boca “abre” quando a distância vertical entre lábios aumenta
    em relação à largura da boca.
    """
    iu, il_ = MOUTH_VERTICAL
    ilf, irf = MOUTH_WIDTH
    vert = _euclid(pts[iu], pts[il_])
    horiz = _euclid(pts[ilf], pts[irf])
    if horiz < 1e-6:
        return 0.0
    return vert / horiz


def _squash_roll_deg(roll: float) -> float:
    r = float(roll)
    for _ in range(5):
        if r > 90.0:
            r -= 180.0
        elif r < -90.0:
            r += 180.0
        else:
            break
    return r


def pose_cabeca_graus(pts: np.ndarray, frame_w: int, frame_h: int) -> tuple[float, float, float]:
    """
    Estima pose da cabeça (pitch/yaw/roll) em graus usando solvePnP.

    O “pulo do gato” aqui é escolher alguns pontos 2D do rosto (nariz, queixo,
    cantos dos olhos e boca) e relacionar com um modelo 3D simplificado.
    Isso dá uma orientação aproximada suficiente para o nosso objetivo:
    detectar se a pessoa está olhando para fora (yaw) ou cabeça baixa (pitch).
    """
    h, w = frame_h, frame_w
    image_points = np.array(
        [
            pts[POSE_NOSE_TIP],
            pts[POSE_CHIN],
            pts[POSE_LEFT_EYE],
            pts[POSE_RIGHT_EYE],
            pts[POSE_LEFT_MOUTH],
            pts[POSE_RIGHT_MOUTH],
        ],
        dtype=np.float64,
    )
    focal = float(max(w, h))
    center = (w / 2.0, h / 2.0)
    cam_matrix = np.array(
        [[focal, 0.0, center[0]], [0.0, focal, center[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points.reshape(6, 1, 2),
        cam_matrix,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = float(angles[0]), float(angles[1]), float(angles[2])
    roll = _squash_roll_deg(roll)
    return pitch, yaw, roll


def ema(prev: Optional[float], value: float, alpha: float) -> float:
    if prev is None or np.isnan(prev):
        return value
    return alpha * value + (1.0 - alpha) * prev


def visibilidade_media_landmark(landmarks: Any, indices: tuple[int, ...]) -> Optional[float]:
    """Média de 'visibility' (quando disponível) para avaliar confiabilidade."""
    vals: list[float] = []
    for i in indices:
        v = getattr(landmarks[i], "visibility", None)
        if v is None:
            continue
        vals.append(float(v))
    if not vals:
        return None
    return float(np.mean(vals))


def olhos_nao_confiaveis_oclusao(
    ear_left: float,
    ear_right: float,
    vis_left: Optional[float],
    vis_right: Optional[float],
    blink_left: Optional[float],
    blink_right: Optional[float],
    *,
    vis_thresh: float = 0.52,
    ear_diff_thresh: float = 0.12,
    ear_ratio_thresh: float = 1.52,
    blink_diff_thresh: float = 0.38,
) -> tuple[bool, str]:
    """
    Heurística para identificar olho “não confiável”.

    Por que isso existe? Em webcam real, às vezes o Landmarker retorna geometria
    estranha por oclusão (mão no rosto, óculos com reflexo, rosto cortado, etc.).
    Se a gente confiar cegamente no EAR, dá falso “olho fechado”.

    Então usamos sinais de:
    - visibility baixa (quando existe)
    - assimetria forte de EAR (um olho “some”)
    - assimetria forte de blink (blendshapes)
    """
    flags: list[str] = []
    if vis_left is not None and vis_left < vis_thresh:
        flags.append("vis_esq_baixa")
    if vis_right is not None and vis_right < vis_thresh:
        flags.append("vis_dir_baixa")
    lo = min(ear_left, ear_right)
    hi = max(ear_left, ear_right)
    ear_diff_hit = hi > 0.13 and (hi - lo) > ear_diff_thresh
    ear_ratio_hit = lo > 1e-6 and hi / lo > ear_ratio_thresh and lo > 0.12
    vis_any = vis_left is not None or vis_right is not None
    if vis_any:
        if ear_diff_hit:
            flags.append("ear_assimetrico")
        if ear_ratio_hit:
            flags.append("ear_ratio")
    else:
        if ear_diff_hit and ear_ratio_hit:
            flags.append("ear_oclusao_prob")
    if blink_left is not None and blink_right is not None:
        bm = max(blink_left, blink_right)
        if bm > 0.22 and abs(blink_left - blink_right) > blink_diff_thresh:
            flags.append("blink_assimetrico")
    if flags:
        return True, "+".join(flags)
    return False, ""


# ---------------------------------------------------------------------------
# Pedro — Malha facial (overlay no vídeo); arestas em facemesh_conexoes.py
# ---------------------------------------------------------------------------


def _lm_pt(lm: Any, idx: int, w: int, h: int, mirror: bool) -> tuple[int, int]:
    x = float(lm[idx].x) * w
    y = float(lm[idx].y) * h
    if mirror:
        x = w - 1 - x
    return int(round(x)), int(round(y))


def _draw_edges(
    frame_bgr: np.ndarray,
    lm: Any,
    connections: frozenset,
    color: tuple[int, int, int],
    w: int,
    h: int,
    mirror: bool,
    thickness: int = 2,
) -> None:
    n = len(lm)
    for i, j in connections:
        i, j = int(i), int(j)
        if i >= n or j >= n:
            continue
        a = _lm_pt(lm, i, w, h, mirror)
        b = _lm_pt(lm, j, w, h, mirror)
        cv2.line(frame_bgr, a, b, color, thickness, cv2.LINE_AA)


def desenhar_malha_rosto(frame_bgr: np.ndarray, lm: Any, *, mirror: bool = True) -> None:
    """Pedro: desenha olhos, boca, nariz e contorno do rosto sobre o frame."""
    h, w = frame_bgr.shape[:2]
    if len(lm) < 468:
        return
    _draw_edges(frame_bgr, lm, FACEMESH_FACE_OVAL, (80, 80, 80), w, h, mirror, 2)
    _draw_edges(frame_bgr, lm, FACEMESH_NOSE, (0, 165, 255), w, h, mirror, 3)
    _draw_edges(frame_bgr, lm, FACEMESH_LEFT_EYE, (0, 255, 255), w, h, mirror, 3)
    _draw_edges(frame_bgr, lm, FACEMESH_RIGHT_EYE, (0, 255, 0), w, h, mirror, 3)
    _draw_edges(frame_bgr, lm, FACEMESH_LIPS, (0, 0, 255), w, h, mirror, 3)
    if len(lm) >= 478:
        _draw_edges(frame_bgr, lm, FACEMESH_LEFT_IRIS, (255, 0, 255), w, h, mirror, 2)
        _draw_edges(frame_bgr, lm, FACEMESH_RIGHT_IRIS, (255, 0, 255), w, h, mirror, 2)
    for idx, label in ((1, ""), (234, "L"), (454, "R")):
        if idx < len(lm):
            p = _lm_pt(lm, idx, w, h, mirror)
            cv2.circle(frame_bgr, p, 5, (0, 255, 255), -1, cv2.LINE_AA)
            if label:
                cv2.putText(
                    frame_bgr,
                    label,
                    (p[0] + 8, p[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )


def piscar_medio_do_resultado(result: Any, face_index: int = 0) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Extrai blendshape de blink (se o modelo fornecer)."""
    shapes = getattr(result, "face_blendshapes", None)
    if not shapes or face_index >= len(shapes):
        return None, None, None
    block = shapes[face_index]
    cats = getattr(block, "categories", None)
    if cats is None:
        cats = block if isinstance(block, (list, tuple)) else []
    left = right = None
    for c in cats:
        raw = getattr(c, "category_name", None) or getattr(c, "display_name", "") or ""
        name = raw.strip().lower()
        norm = name.replace(" ", "").replace("_", "").replace("-", "")
        score = float(getattr(c, "score", 0.0) or 0.0)
        if norm in ("eyeblinkleft", "blinkleft") or "eyeblinkleft" in norm:
            left = score
        elif norm in ("eyeblinkright", "blinkright") or "eyeblinkright" in norm:
            right = score
    if left is None or right is None:
        return None, None, None
    return left, right, (left + right) / 2.0


# ---------------------------------------------------------------------------
# Pedro — Alarme sonoro (nível vermelho)
# ---------------------------------------------------------------------------


def _synth_alarm() -> np.ndarray:
    fs = 44_100
    chunks: list[np.ndarray] = []
    pattern = (
        (1040.0, 0.22),
        (520.0, 0.18),
        (1040.0, 0.22),
        (780.0, 0.2),
        (1200.0, 0.35),
    )
    for freq, dur in pattern:
        n = int(fs * dur)
        t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float64)
        s = 0.62 * np.sin(2.0 * np.pi * freq * t)
        s += 0.28 * np.sign(np.sin(2.0 * np.pi * freq * 2.1 * t))
        fade = np.ones(n, dtype=np.float64)
        fade[: int(0.02 * fs)] = np.linspace(0.0, 1.0, int(0.02 * fs), endpoint=True)
        fade[-int(0.03 * fs) :] = np.linspace(1.0, 0.0, int(0.03 * fs), endpoint=True)
        chunks.append(np.clip(s * fade, -0.95, 0.95))
    return np.concatenate(chunks).astype(np.float32)


def _play_alarm_blocking() -> None:
    try:
        import sounddevice as sd

        sd.play(_synth_alarm(), 44_100, blocking=True)
    except Exception:
        try:
            import winsound

            for hz, ms in ((1200, 220), (800, 200), (1200, 220), (900, 280)):
                winsound.Beep(hz, ms)
        except Exception:
            pass


def play_red_alert_async() -> None:
    threading.Thread(target=_play_alarm_blocking, daemon=True).start()


# ---------------------------------------------------------------------------
# Lucas — MediaPipe Tasks: opções do Face Landmarker e ficheiro .task
# (A função executar() mais abaixo fecha o loop: webcam → RGB → deteção.)
# ---------------------------------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
RunningMode = mp.tasks.vision.RunningMode

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_FILENAME = "face_landmarker.task"


class Level(Enum):
    ATENTO = "verde"
    DISTRAIDO = "amarelo"
    FATIGADO = "vermelho"


@dataclass
class Thresholds:
    """Rafael: limiares ajustáveis (métricas e tempos em segundos)."""

    ear_fechado: float = 0.22
    blink_fechado: float = 0.38
    mar_bocejo: float = 0.45
    yaw_fora_graus: float = 22.0
    pitch_cabeca_baixa_graus: float = 15.0
    olhos_fechados_seg: float = 1.2
    bocejo_seg: float = 0.9
    olhar_fora_seg: float = 2.0
    cabeca_baixa_seg: float = 2.5


def garantir_modelo(path: Path) -> None:
    """Lucas: garante que o modelo existe localmente (download na 1.ª execução)."""
    if path.is_file() and path.stat().st_size > 1_000_000:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Baixando modelo MediaPipe para {path} ...")
    urlretrieve(MODEL_URL, path)
    print("Download concluído.")


def desenhar_hud(frame: np.ndarray, level: Level, score: float, lines: list[str]) -> None:
    """Pedro: barra de score, texto de métricas e instruções na imagem."""
    h, w = frame.shape[:2]
    if level == Level.ATENTO:
        color = (80, 220, 80)
    elif level == Level.DISTRAIDO:
        color = (0, 220, 255)
    else:
        color = (60, 60, 255)
    bar_w = int(w * 0.45)
    bar_x = 20
    bar_y = h - 55
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 18), (40, 40, 40), -1)
    fill = int(bar_w * float(np.clip(score / 100.0, 0.0, 1.0)))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + 18), color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 18), (200, 200, 200), 1)
    label = f"Nivel: {level.value.upper()} | Score: {score:.0f}/100"
    cv2.putText(
        frame,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        color,
        2,
        cv2.LINE_AA,
    )
    y0 = 78
    for i, t in enumerate(lines[:8]):
        cv2.putText(
            frame,
            t,
            (20, y0 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        "[c] calibrar olhando na tela | [q] sair",
        (20, h - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


def executar(args: argparse.Namespace) -> None:
    # --- Lucas: entrada (webcam) + carregar modelo + criar Face Landmarker ---
    root = Path(__file__).resolve().parent
    model_path = root / MODEL_FILENAME
    garantir_modelo(model_path)
    # --- Rafael: limiares usados nas regras (EAR, MAR, pose, tempos em segundos) ---
    thresholds = Thresholds()
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    baseline_pitch: Optional[float] = None
    baseline_yaw: Optional[float] = None
    calib_samples_pitch: list[float] = []
    calib_samples_yaw: list[float] = []
    calibrating = False
    calib_until = 0.0
    t_eyes_low: Optional[float] = None
    t_mar_high: Optional[float] = None
    t_yaw_away: Optional[float] = None
    t_pitch_down: Optional[float] = None
    score_smooth: Optional[float] = None
    last_ts = time.perf_counter()
    video_ms = 0
    prev_level: Optional[Level] = None
    last_red_alert_ts = 0.0
    debug_last_print = 0.0

    with FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            now = time.perf_counter()
            dt = max(now - last_ts, 1e-6)
            last_ts = now

            # [Lucas] Pré-processamento: BGR (OpenCV) → RGB (MediaPipe)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            video_ms += int(dt * 1000)
            if video_ms < 1:
                video_ms = 1

            # [Lucas] deteção em vídeo (timestamp em ms deve aumentar)
            result: FaceLandmarkerResult = landmarker.detect_for_video(mp_image, video_ms)
            # [Pedro] espelho só para ficar mais natural na webcam
            display = cv2.flip(frame, 1)
            lines: list[str] = []
            level = Level.ATENTO
            raw_score = 88.0

            if not result.face_landmarks:
                lines.append("Rosto: nao detectado")
                raw_score = 42.0
                t_eyes_low = t_mar_high = t_yaw_away = t_pitch_down = None
            else:
                lm = result.face_landmarks[0]
                pts = landmarks_para_pixels(lm, w, h)

                # [Rafael] Métricas principais: EAR, MAR, pose; calibração [c]; timers; nível/score
                ear_l = razao_aspecto_olho(pts, LEFT_EYE_IDX)
                ear_r = razao_aspecto_olho(pts, RIGHT_EYE_IDX)
                ear_avg = (ear_l + ear_r) / 2.0
                mar = razao_aspecto_boca(pts)
                pitch, yaw, roll = pose_cabeca_graus(pts, w, h)

                # 4) Calibração (tecla [c]): define baseline de pitch/yaw olhando “reto”
                # A ideia é não usar valores absolutos de pose, mas sim desvio do baseline.
                if calibrating:
                    if now < calib_until:
                        calib_samples_pitch.append(pitch)
                        calib_samples_yaw.append(yaw)
                    else:
                        calibrating = False
                        if len(calib_samples_pitch) >= 8:
                            baseline_pitch = float(np.median(calib_samples_pitch))
                            baseline_yaw = float(np.median(calib_samples_yaw))
                        calib_samples_pitch.clear()
                        calib_samples_yaw.clear()
                blink_l, blink_r, blink_mean = piscar_medio_do_resultado(result, 0)
                vis_l = visibilidade_media_landmark(lm, LEFT_EYE_IDX)
                vis_r = visibilidade_media_landmark(lm, RIGHT_EYE_IDX)
                bad_eyes, eyes_bad_reason = olhos_nao_confiaveis_oclusao(
                    ear_l, ear_r, vis_l, vis_r, blink_l, blink_r
                )
                if bad_eyes:
                    eyes_low_geom = False
                    eyes_low_blend = False
                else:
                    eyes_low_geom = ear_avg < thresholds.ear_fechado
                    eyes_low_blend = (
                        blink_mean is not None and blink_mean > thresholds.blink_fechado
                    )
                eyes_low = eyes_low_geom or eyes_low_blend
                mar_high = mar > thresholds.mar_bocejo
                yaw_delta = abs(yaw - baseline_yaw) if baseline_yaw is not None else 0.0
                yaw_away = baseline_yaw is not None and yaw_delta > thresholds.yaw_fora_graus
                pitch_down = (
                    baseline_pitch is not None
                    and (pitch - baseline_pitch) > thresholds.pitch_cabeca_baixa_graus
                )

                # 5) Decisão por TEMPO:
                # ao invés de classificar por frame, acumulamos segundos enquanto
                # a condição está ativa. Isso reduz falsos positivos.
                if not calibrating:
                    if not bad_eyes:
                        if eyes_low:
                            t_eyes_low = (t_eyes_low or 0.0) + dt
                        else:
                            t_eyes_low = None
                    else:
                        t_eyes_low = None
                    if mar_high:
                        t_mar_high = (t_mar_high or 0.0) + dt
                    else:
                        t_mar_high = None
                    if yaw_away:
                        t_yaw_away = (t_yaw_away or 0.0) + dt
                    else:
                        t_yaw_away = None
                    if pitch_down:
                        t_pitch_down = (t_pitch_down or 0.0) + dt
                    else:
                        t_pitch_down = None
                te = t_eyes_low or 0.0
                tm = t_mar_high or 0.0
                ty = t_yaw_away or 0.0
                tp = t_pitch_down or 0.0
                fatigued = (
                    te > thresholds.olhos_fechados_seg
                    or tm > thresholds.bocejo_seg
                    or tp > thresholds.cabeca_baixa_seg
                )
                distracted = ty > thresholds.olhar_fora_seg or bad_eyes

                # 6) Nível + score:
                # - vermelho: fadiga (olhos fechados/bocejo/cabeça baixa por tempo)
                # - amarelo: distração (olhar fora por tempo OU olhos não confiáveis)
                # - verde: atento (score cai conforme “sinais” aparecem)
                if fatigued:
                    level = Level.FATIGADO
                    raw_score = 18.0
                elif distracted:
                    level = Level.DISTRAIDO
                    raw_score = 40.0 if bad_eyes else 48.0
                else:
                    level = Level.ATENTO
                    raw_score = 92.0 - 32.0 * min(te / max(thresholds.olhos_fechados_seg, 1e-6), 1.0)
                    raw_score -= 18.0 * min(ty / max(thresholds.olhar_fora_seg, 1e-6), 1.0)
                    raw_score -= 12.0 * min(tm / max(thresholds.bocejo_seg, 1e-6), 1.0)
                    raw_score = float(np.clip(raw_score, 35.0, 100.0))
                blink_txt = (
                    f"Blink L/R/med: {blink_l:.2f} {blink_r:.2f} {blink_mean:.2f}"
                    if blink_mean is not None
                    else "Blink: (sem blendshapes)"
                )
                olhos_txt = "INVALIDO (nao valida)" if bad_eyes else ("fechados" if eyes_low else "abertos")
                if vis_l is None and vis_r is None:
                    vis_txt = "vis esq/dir: (n/d)"
                else:
                    ls = f"{vis_l:.2f}" if vis_l is not None else "-"
                    rs = f"{vis_r:.2f}" if vis_r is not None else "-"
                    vis_txt = f"vis esq/dir: {ls} / {rs}"
                lines = [
                    f"EAR L/med/R: {ear_l:.3f} {ear_avg:.3f} {ear_r:.3f}  MAR: {mar:.3f}  Olhos: {olhos_txt}",
                    blink_txt,
                    vis_txt,
                    f"Pose deg - pitch:{pitch:.1f} yaw:{yaw:.1f} roll:{roll:.1f}",
                    f"Timers s - olhos:{te:.1f} bocejo:{tm:.1f} fora:{ty:.1f} cabeca:{tp:.1f}",
                ]
                if bad_eyes:
                    short = eyes_bad_reason[:42] + ("..." if len(eyes_bad_reason) > 42 else "")
                    lines.append(f"Alerta rosto: {short}")
                if baseline_pitch is None:
                    lines.append("Pressione [c] olhando para a tela (pose cabeca)")
                else:
                    lines.append(f"Baseline pitch/yaw: {baseline_pitch:.1f} / {baseline_yaw:.1f}")

                # [Pedro] --debug: números no terminal (~1×/s) para a banca acompanhar
                if args.debug and (now - debug_last_print) >= 1.0:
                    debug_last_print = now
                    print(
                        f"EAR={ear_avg:.3f} MAR={mar:.3f} "
                        f"pitch/yaw={pitch:.1f}/{yaw:.1f} "
                        f"t(olhos/bocejo/fora/cabeca)={te:.1f}/{tm:.1f}/{ty:.1f}/{tp:.1f} "
                        f"nivel={level.value}"
                    )

            # [Pedro] overlay, HUD, alarme e janela
            if result.face_landmarks and not args.no_mesh:
                desenhar_malha_rosto(display, result.face_landmarks[0], mirror=True)
            score_smooth = ema(score_smooth, raw_score, 0.2)
            desenhar_hud(display, level, float(score_smooth), lines)
            if not args.no_audio and level == Level.FATIGADO:
                if prev_level != Level.FATIGADO:
                    play_red_alert_async()
                    last_red_alert_ts = now
                elif now - last_red_alert_ts >= 8.0:
                    play_red_alert_async()
                    last_red_alert_ts = now
            prev_level = level
            cv2.imshow("Atencao / fadiga (MediaPipe)", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                calibrating = True
                calib_until = time.perf_counter() + 1.5
                calib_samples_pitch.clear()
                calib_samples_yaw.clear()
    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Pedro: opções de linha de comando (--no-mesh, --no-audio, --debug)."""
    p = argparse.ArgumentParser(description="Monitor de atencao e fadiga (webcam).")
    p.add_argument("--camera", type=int, default=0, help="Indice da webcam (default 0).")
    p.add_argument(
        "--no-mesh",
        action="store_true",
        help="Nao desenha malha (olhos/boca/nariz) por cima do video.",
    )
    p.add_argument(
        "--no-audio",
        action="store_true",
        help="Desliga o alarme sonoro ao entrar em vermelho.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Imprime no terminal (1x/s) as metricas e o nivel, para estudo/apresentacao.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Compatibilidade: aliases em inglês (caso outros ficheiros/notebooks usem)
# ---------------------------------------------------------------------------

landmarks_to_pixels = landmarks_para_pixels
eye_aspect_ratio = razao_aspecto_olho
mouth_aspect_ratio = razao_aspecto_boca
head_pose_degrees = pose_cabeca_graus
mean_landmark_visibility = visibilidade_media_landmark
eyes_unreliable_occlusion = olhos_nao_confiaveis_oclusao
draw_face_landmarks_overlay = desenhar_malha_rosto
mean_eye_blink_from_result = piscar_medio_do_resultado
ensure_model = garantir_modelo
draw_hud = desenhar_hud
run = executar


if __name__ == "__main__":
    executar(parse_args())
