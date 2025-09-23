import os
import cv2
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from sklearn.metrics.pairwise import cosine_distances
from insightface.app import FaceAnalysis
import hdbscan
from collections import defaultdict

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def _win_long(path: Path) -> str:
    p = str(path.resolve())
    if os.name == "nt":
        return "\\\\?\\" + p if not p.startswith("\\\\?\\") else p
    return p

def imread_safe(path: Path):
    try:
        data = np.fromfile(_win_long(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

def merge_clusters_by_centroid(
    embeddings: List[np.ndarray],
    owners: List[Path],
    raw_labels: np.ndarray,
    threshold: Optional[float] = None,
    auto_threshold: bool = False,
    margin: float = 0.05,
    min_threshold: float = 0.2,
    max_threshold: float = 0.4,
    progress_callback=None
) -> Tuple[Dict[int, Set[Path]], Dict[Path, Set[int]]]:

    if progress_callback:
        progress_callback("🔄 Объединение близких кластеров...", 92)

    cluster_embeddings: Dict[int, List[np.ndarray]] = defaultdict(list)
    cluster_paths: Dict[int, List[Path]] = defaultdict(list)

    for label, emb, path in zip(raw_labels, embeddings, owners):
        if label == -1:
            continue
        cluster_embeddings[label].append(emb)
        cluster_paths[label].append(path)

    centroids = {label: np.mean(embs, axis=0) for label, embs in cluster_embeddings.items()}
    labels = list(centroids.keys())

    if auto_threshold and threshold is None:
        pairwise = [cosine_distances([centroids[a]], [centroids[b]])[0][0]
                    for i, a in enumerate(labels) for b in labels[i+1:]]
        if pairwise:
            mean_dist = np.mean(pairwise)
            threshold = max(min_threshold, min(mean_dist - margin, max_threshold))
        else:
            threshold = min_threshold

        if progress_callback:
            progress_callback(f"📏 Авто-порог объединения: {threshold:.3f}", 93)
    elif threshold is None:
        threshold = 0.3

    next_cluster_id = 0
    label_to_group = {}
    total = len(labels)

    for i, label_i in enumerate(labels):
        if progress_callback:
            percent = 93 + int((i + 1) / max(total, 1) * 2)
            progress_callback(f"🔁 Слияние кластеров: {percent}% ({i+1}/{total})", percent)

        if label_i in label_to_group:
            continue
        group = [label_i]
        for j in range(i + 1, len(labels)):
            label_j = labels[j]
            if label_j in label_to_group:
                continue
            dist = cosine_distances([centroids[label_i]], [centroids[label_j]])[0][0]
            if dist < threshold:
                group.append(label_j)

        for l in group:
            label_to_group[l] = next_cluster_id
        next_cluster_id += 1

    merged_clusters: Dict[int, Set[Path]] = defaultdict(set)
    cluster_by_img: Dict[Path, Set[int]] = defaultdict(set)

    for label, path in zip(raw_labels, owners):
        if label == -1:
            continue
        new_label = label_to_group[label]
        merged_clusters[new_label].add(path)
        cluster_by_img[path].add(new_label)

    return merged_clusters, cluster_by_img

def build_plan_live(
    input_dir: Path,
    det_size=(640, 640),
    min_score: float = 0.5,
    min_cluster_size: int = 2,
    min_samples: int = 1,
    providers: List[str] = ("CPUExecutionProvider",),
    progress_callback=None,
):
    input_dir = Path(input_dir)
    all_images = [p for p in input_dir.rglob("*") if is_image(p) and "общие" not in str(p.parent).lower()]

    if progress_callback:
        progress_callback(f"📂 Сканируется: {input_dir}, найдено изображений: {len(all_images)}", 1)

    app = FaceAnalysis(name="buffalo_l", providers=list(providers))
    ctx_id = -1 if "cpu" in str(providers).lower() else 0
    app.prepare(ctx_id=ctx_id, det_size=det_size)

    if progress_callback:
        progress_callback("✅ Модель загружена, начинаем анализ изображений...", 10)

    embeddings = []
    owners = []
    img_face_count = {}
    unreadable = []
    no_faces = []

    total = len(all_images)
    processed_faces = 0
    
    for i, p in enumerate(all_images):
        # Обновляем прогресс для каждого изображения
        if progress_callback:
            percent = 10 + int((i + 1) / max(total, 1) * 70)  # 10-80% для анализа изображений
            progress_callback(f"📷 Анализ изображений: {percent}% ({i+1}/{total}) - {p.name}", percent)
        
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
            emb = emb.astype(np.float64)  # HDBSCAN expects double
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            embeddings.append(emb)
            owners.append(p)
            count += 1
            processed_faces += 1

        if count > 0:
            img_face_count[p] = count

    if not embeddings:
        if progress_callback:
            progress_callback("⚠️ Не найдено лиц для кластеризации", 100)
        print(f"⚠️ Нет эмбеддингов: {input_dir}")
        return {
            "clusters": {},
            "plan": [],
            "unreadable": [str(p) for p in unreadable],
            "no_faces": [str(p) for p in no_faces],
        }

    # Этап 2: Кластеризация
    if progress_callback:
        progress_callback(f"🔄 Кластеризация {len(embeddings)} лиц...", 80)
    
    X = np.vstack(embeddings)
    distance_matrix = cosine_distances(X)

    if progress_callback:
        progress_callback("🔄 Вычисление матрицы расстояний...", 85)

    model = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=min_cluster_size, min_samples=min_samples)
    raw_labels = model.fit_predict(distance_matrix)

    cluster_map, cluster_by_img = merge_clusters_by_centroid(
        embeddings=embeddings,
        owners=owners,
        raw_labels=raw_labels,
        auto_threshold=True,
        margin=0.05,
        min_threshold=0.2,
        max_threshold=0.4,
        progress_callback=progress_callback
    )

    # Этап 3: Формирование плана распределения
    if progress_callback:
        progress_callback("🔄 Формирование плана распределения...", 95)
    
    plan = []
    for path in all_images:
        clusters = cluster_by_img.get(path)
        if not clusters:
            continue
        plan.append({
            "path": str(path),
            "cluster": sorted(list(clusters)),
            "faces": img_face_count.get(path, 0)
        })

    # Завершение
    if progress_callback:
        progress_callback(f"✅ Кластеризация завершена! Найдено {len(cluster_map)} кластеров, обработано {len(plan)} изображений", 100)

    print(f"✅ Кластеризация завершена: {input_dir} → кластеров: {len(cluster_map)}, изображений: {len(plan)}")

    return {
        "clusters": {
            int(k): [str(p) for p in sorted(v, key=lambda x: str(x))]
            for k, v in cluster_map.items()
        },
        "plan": plan,
        "unreadable": [str(p) for p in unreadable],
        "no_faces": [str(p) for p in no_faces],
    }

def distribute_to_folders(plan: dict, base_dir: Path, cluster_start: int = 1, progress_callback=None) -> Tuple[int, int, int]:
    moved, copied = 0, 0
    moved_paths = set()

    used_clusters = sorted({c for item in plan.get("plan", []) for c in item["cluster"]})
    cluster_id_map = {old: cluster_start + idx for idx, old in enumerate(used_clusters)}

    plan_items = plan.get("plan", [])
    total_items = len(plan_items)
    
    if progress_callback:
        progress_callback(f"🔄 Распределение {total_items} файлов по папкам...", 0)

    for i, item in enumerate(plan_items):
        if progress_callback:
            percent = int((i + 1) / max(total_items, 1) * 100)
            progress_callback(f"📁 Распределение файлов: {percent}% ({i+1}/{total_items})", percent)
            
        src = Path(item["path"])
        clusters = [cluster_id_map[c] for c in item["cluster"]]
        if not src.exists():
            continue

        if len(clusters) == 1:
            cluster_id = clusters[0]
            dst = base_dir / f"{cluster_id}" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(src), str(dst))
                moved += 1
                moved_paths.add(src.parent)
            except Exception as e:
                print(f"❌ Ошибка перемещения {src} → {dst}: {e}")
        else:
            for cluster_id in clusters:
                dst = base_dir / f"{cluster_id}" / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(str(src), str(dst))
                    copied += 1
                except Exception as e:
                    print(f"❌ Ошибка копирования {src} → {dst}: {e}")
            try:
                src.unlink()  # удаляем оригинал после копирования в несколько папок
            except Exception as e:
                print(f"❌ Ошибка удаления {src}: {e}")

    # Очистка пустых папок
    if progress_callback:
        progress_callback("🧹 Очистка пустых папок...", 100)

    for p in sorted(moved_paths, key=lambda x: len(str(x)), reverse=True):
        try:
            if p.exists() and not any(p.iterdir()):
                p.rmdir()
        except Exception:
            pass

    print(f"📦 Перемещено: {moved}, скопировано: {copied}")
    return moved, copied, cluster_start + len(used_clusters)

def process_group_folder(group_dir: Path, progress_callback=None):
    cluster_counter = 1
    subfolders = [f for f in sorted(group_dir.iterdir()) if f.is_dir() and "общие" not in f.name.lower()]
    total_subfolders = len(subfolders)
    
    for i, subfolder in enumerate(subfolders):
        if progress_callback:
            percent = 10 + int((i + 1) / max(total_subfolders, 1) * 80)
            progress_callback(f"🔍 Обрабатывается подпапка: {subfolder.name} ({i+1}/{total_subfolders})", percent)
            
        print(f"🔍 Обрабатывается подпапка: {subfolder}")
        plan = build_plan_live(subfolder)
        print(f"📊 Кластеров: {len(plan.get('clusters', {}))}, файлов: {len(plan.get('plan', []))}")
        moved, copied, cluster_counter = distribute_to_folders(plan, subfolder, cluster_start=cluster_counter)



