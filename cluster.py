import os
import cv2
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.cluster import OPTICS
from insightface.app import FaceAnalysis
from collections import defaultdict

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def imread_safe(path: Path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def normalize_embedding(e: np.ndarray) -> np.ndarray:
    e = e.astype(np.float64)
    norm = np.linalg.norm(e)
    return e / norm if norm > 0 else e


def init_face_model(det_size=(640, 640), providers=("CPUExecutionProvider",)) -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l", providers=list(providers))
    ctx_id = -1 if "cpu" in str(providers).lower() else 0
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app


def extract_embeddings(app: FaceAnalysis, img_paths: List[Path], min_score: float = 0.5, progress=None) -> Tuple[List[np.ndarray], List[Path], List[Path], List[Path], Dict[Path, int]]:
    embeddings, owners, unreadable, no_faces = [], [], [], []
    img_face_count = {}
    total = len(img_paths)

    for i, p in enumerate(img_paths):
        if progress:
            percent = 10 + int((i + 1) / max(total, 1) * 70)
            progress(f"📷 Анализ изображений: {percent}% ({i+1}/{total}) - {p.name}", percent)
        
        img = imread_safe(p)
        if img is None:
            unreadable.append(p)
            continue
            
        faces = app.get(img)
        if not faces:
            no_faces.append(p)
            continue

        count = 0
        for f in faces:
            if getattr(f, "det_score", 1.0) < min_score:
                continue
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                continue
            emb = normalize_embedding(emb)
            if np.any(np.isnan(emb)) or np.any(np.isinf(emb)) or np.max(np.abs(emb)) < 1e-6:
                continue
            embeddings.append(emb)
            owners.append(p)
            count += 1
        if count:
            img_face_count[p] = count

    return embeddings, owners, unreadable, no_faces, img_face_count


def cluster_embeddings(embeddings: List[np.ndarray], progress=None) -> np.ndarray:
    if progress:
        progress("🔄 Кластеризация через OPTICS...", 80)

    if not embeddings:
        return np.array([])
    
    X = np.vstack(embeddings)
    if X.shape[0] > 50:
        sim_matrix = cosine_similarity(X)
        dist_matrix = 1 - sim_matrix
    else:
        dist_matrix = cosine_distances(X)

    model = OPTICS(metric='precomputed', min_samples=2, xi=0.05, min_cluster_size=2)
    labels = model.fit_predict(dist_matrix)
    return labels


def build_plan(input_dir: Path, providers=("CPUExecutionProvider",), progress=None):
    input_dir = Path(input_dir)
    print(f"📂 Сканируем папку: {input_dir}")
    excluded_names = ["общие", "общая", "common", "shared", "все", "all", "mixed", "смешанные"]
    img_paths = [p for p in input_dir.rglob("*") if is_image(p) and not any(ex in str(p).lower() for ex in excluded_names)]
    print(f"📷 Найдено изображений: {len(img_paths)}")

    if progress:
        progress(f"📂 Сканируется: {input_dir}, найдено изображений: {len(img_paths)}", 1)

    print("🤖 Инициализируем модель распознавания лиц...")
    app = init_face_model(providers=providers)
    print("✅ Модель загружена")

    if progress:
        progress("✅ Модель загружена, начинаем анализ изображений...", 10)

    print("🔍 Начинаем извлечение эмбеддингов...")
    embeddings, owners, unreadable, no_faces, face_count = extract_embeddings(app, img_paths, progress=progress)
    print(f"📊 Извлечено эмбеддингов: {len(embeddings)}")

    if not embeddings:
        if progress:
            progress("⚠️ Не найдено лиц для кластеризации", 100)
        return {"clusters": {}, "plan": [], "unreadable": [str(p) for p in unreadable], "no_faces": [str(p) for p in no_faces]}

    print("🔄 Начинаем кластеризацию...")
    labels = cluster_embeddings(embeddings, progress)
    print(f"📊 Результат кластеризации: {len(np.unique(labels))} кластеров")

    if labels.size > 0 and np.all(labels == -1):
        print("⚠️ Все точки помечены как шум, создаем отдельные кластеры")
        labels = np.arange(len(embeddings))

    cluster_map: Dict[int, Set[Path]] = defaultdict(set)
    cluster_by_img: Dict[Path, Set[int]] = defaultdict(set)

    for label, path in zip(labels, owners):
        if label == -1:
            continue
        cluster_map[label].add(path)
        cluster_by_img[path].add(label)
    
    plan = []
    for path in img_paths:
        clusters = cluster_by_img.get(path)
        if not clusters:
            continue
        plan.append({"path": str(path), "cluster": sorted(list(clusters)), "faces": face_count.get(path, 0)})

    if not plan and embeddings:
        fallback_cluster_id = 0
        img_with_faces = [p for p, cnt in face_count.items() if cnt > 0]
        for p in img_with_faces:
            plan.append({"path": str(p), "cluster": [fallback_cluster_id], "faces": face_count.get(p, 0)})

    if progress:
        progress(f"✅ Кластеризация завершена! Найдено {len(cluster_map)} кластеров, обработано {len(plan)} изображений", 100)

    return {
        "clusters": {int(k): [str(p) for p in sorted(v, key=lambda x: str(x))] for k, v in cluster_map.items()},
        "plan": plan,
        "unreadable": [str(p) for p in unreadable],
        "no_faces": [str(p) for p in no_faces],
    }


def distribute_to_folders(plan_result, input_dir: Path, progress_callback=None):
    """Распределяет изображения по папкам на основе плана кластеризации."""
    if progress_callback:
        progress_callback("📁 Создание папок и распределение файлов...", 90)
    
    clusters = plan_result.get("clusters", {})
    plan = plan_result.get("plan", [])
    
    if not clusters:
        return 0, 0, 0
    
    input_dir = Path(input_dir)
    output_dir = input_dir / "clustered"
    output_dir.mkdir(exist_ok=True)
    
    moved = 0
    copied = 0
    next_cluster_id = max(clusters.keys(), default=0) + 1
    
    for cluster_id, image_paths in clusters.items():
        if not image_paths:
            continue
            
        cluster_dir = output_dir / f"person_{cluster_id}"
        cluster_dir.mkdir(exist_ok=True)
        
        for img_path in image_paths:
            src = Path(img_path)
        if not src.exists():
            continue

            dst = cluster_dir / src.name
            if dst.exists():
                # Если файл уже существует, добавляем суффикс
                stem = src.stem
                suffix = src.suffix
                counter = 1
                while dst.exists():
                    dst = cluster_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
            
            try:
                shutil.copy2(src, dst)
                copied += 1
            except Exception as e:
                print(f"Ошибка копирования {src} -> {dst}: {e}")
    
    return moved, copied, next_cluster_id


def process_group_folder(input_dir: Path, progress_callback=None):
    """Обрабатывает папку с уже сгруппированными изображениями."""
    if progress_callback:
        progress_callback("🔄 Обработка групповой папки...", 10)
    
    # Простая реализация - просто сканируем папку
    input_dir = Path(input_dir)
    if not input_dir.exists():
        return
    
    # Сканируем все изображения в папке
    all_images = [p for p in input_dir.rglob("*") if is_image(p)]
    
    if progress_callback:
        progress_callback(f"📂 Найдено {len(all_images)} изображений в групповой папке", 50)
    
    # Здесь можно добавить дополнительную логику обработки
    # Например, переименование файлов, создание превью и т.д.
    
    if progress_callback:
        progress_callback("✅ Групповая папка обработана", 100)


if __name__ == "__main__":
    def dummy_progress(msg, percent):
        print(f"[{percent}%] {msg}")

    result = build_plan(
        input_dir=Path("C:/Users/denis/Desktop/t/9"),
        providers=("CPUExecutionProvider",),
        progress=dummy_progress
    )

    print("\n📋 План:")
    for item in result["plan"]:
        print(item)
