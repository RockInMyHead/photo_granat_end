import os
import cv2
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import OPTICS, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from insightface.app import FaceAnalysis
import hdbscan
from collections import defaultdict

try:
    from PIL import Image, ImageOps
    _PIL_OK = True
except Exception:
    _PIL_OK = False

# =============================================================================
# Константы порогов кластеризации
# =============================================================================
DEFAULT_THRESHOLD = 0.27           # [1] Основной порог объединения кластеров — более строгий
FINAL_MERGE_THRESHOLD = 0.25      # [1] Порог финального объединения через merge_clusters_by_centroid
POSTPROCESS_THRESHOLD = 0.23      # [1] Порог в post_process_clusters
SMART_LARGE_THRESHOLD = 0.28      # [1] Порог для объединения маленьких кластеров с большими
SMART_SMALL_THRESHOLD = 0.25      # [1] Порог для объединения между маленькими кластерами
SUPER_AGGRESSIVE_THRESHOLD = 0.26 # [1] Порог для супер-агрессивного объединения
# =============================================================================

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def imread_exif_oriented(path: Path) -> Optional[np.ndarray]:
    """Читает изображение с учётом EXIF-ориентации. Возвращает BGR np.ndarray или None."""
    p = str(Path(path).resolve())
    if _PIL_OK:
        try:
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im)
                im = im.convert('RGB')
                arr = np.array(im)
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            pass
    try:
        data = np.fromfile(p, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def enhance_for_detection(img: np.ndarray) -> np.ndarray:
    """Лёгкое улучшение: CLAHE по L-каналу + мягкая гамма."""
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        gamma = 0.9
        table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)]).astype('uint8')
        return cv2.LUT(enhanced, table)
    except Exception:
        return img


def maybe_upscale_small(img: np.ndarray, min_side: int = 800) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m < min_side:
        scale = min_side / float(m)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return img


def init_dual_apps(providers: List[str], det_size_base=(640, 640), det_size_hi=(1280, 1280)):
    """Подготавливает две копии FaceAnalysis с разными детекторами."""
    ctx_id = -1 if "cpu" in str(providers).lower() else 0
    app_base = FaceAnalysis(name="buffalo_l", providers=list(providers))
    app_base.prepare(ctx_id=ctx_id, det_size=det_size_base)

    app_hi = FaceAnalysis(name="buffalo_l", providers=list(providers))
    app_hi.prepare(ctx_id=ctx_id, det_size=det_size_hi)
    return app_base, app_hi


def detect_faces_multi(
    app_base: FaceAnalysis,
    app_hi: FaceAnalysis,
    img_bgr: np.ndarray,
    min_score_main: float = 0.4,
    min_score_fallback: float = 0.35
) -> List:
    """Многоступенчатая детекция."""
    faces = app_base.get(img_bgr) or []
    faces = [f for f in faces if getattr(f, "det_score", 1.0) >= min_score_main]
    if faces:
        return faces

    img_enh = enhance_for_detection(img_bgr)
    faces = app_base.get(img_enh) or []
    faces = [f for f in faces if getattr(f, "det_score", 1.0) >= min_score_fallback]
    if faces:
        return faces

    img_hi = maybe_upscale_small(img_enh, min_side=900)
    faces = app_hi.get(img_hi) or []
    faces = [f for f in faces if getattr(f, "det_score", 1.0) >= min_score_fallback]
    if faces:
        return faces

    img_rot = cv2.rotate(img_hi, cv2.ROTATE_90_CLOCKWISE)
    faces = app_hi.get(img_rot) or []
    faces = [f for f in faces if getattr(f, "det_score", 1.0) >= min_score_fallback]
    if faces:
        return faces

    img_rot2 = cv2.rotate(img_hi, cv2.ROTATE_90_COUNTERCLOCKWISE)
    faces = app_hi.get(img_rot2) or []
    faces = [f for f in faces if getattr(f, "det_score", 1.0) >= min_score_fallback]
    return faces


# ------------------------------------------------------------
# HYBRID CLUSTERING HELPERS
# ------------------------------------------------------------


def _safe_silhouette(X: np.ndarray, labels: np.ndarray, metric: str = "cosine") -> Optional[float]:
    try:
        valid = labels[labels != -1]
        if len(np.unique(valid)) < 2:
            return None
        return silhouette_score(X, labels, metric=metric)
    except Exception:
        return None


def _quality(X: np.ndarray, labels: np.ndarray) -> Tuple[int, float, Optional[float]]:
    n = len(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = float(np.sum(labels == -1)) / max(n, 1)
    sil = _safe_silhouette(X, labels, metric="cosine")
    return n_clusters, noise_ratio, sil


def cluster_with_optics_hybrid(X: np.ndarray, progress_callback=None) -> np.ndarray:
    """OPTICS → quality check → HDBSCAN fallback."""
    if progress_callback:
        progress_callback("🔄 Кластеризация: OPTICS→проверка→HDBSCAN fallback", 80)

    labels = None
    try:
        optics = OPTICS(metric="cosine", min_samples=2, cluster_method="xi", xi=0.05)
        optics.fit(X)
        labels = optics.labels_.copy()
        if progress_callback:
            n_optics = len(set(labels)) - (1 if -1 in labels else 0)
            progress_callback(f"✅ OPTICS(xi) метки получены (кластеры: {n_optics})", 82)
    except Exception as e:
        if progress_callback:
            progress_callback(f"⚠️ OPTICS ошибка: {e}. Переходим к HDBSCAN.", 82)
        labels = None

    use_fallback = labels is None
    if labels is not None:
        n_clusters, noise_ratio, sil = _quality(X, labels)
        if n_clusters == 0 or (n_clusters == 1 and noise_ratio > 0.2) or noise_ratio > 0.75 or (sil is not None and sil < 0.12):
            use_fallback = True

    if use_fallback:
        if progress_callback:
            progress_callback("🔁 Fallback: HDBSCAN (авто min_cluster_size)", 83)
        min_cluster_size = max(2, int(np.sqrt(len(X))))
        try:
            hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1, metric="euclidean")
            labels_fb = hdb.fit_predict(X)
            labels = labels_fb
            if progress_callback:
                n_fb, noise_fb, sil_fb = _quality(X, labels_fb)
                sil_text = f"{sil_fb:.3f}" if sil_fb is not None else "n/a"
                progress_callback(f"✅ HDBSCAN fallback: кластеры={n_fb}, шум={noise_fb:.0%}, silhouette={sil_text}", 85)
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ HDBSCAN fallback не сработал: {e}. Все в один кластер.", 85)
            labels = np.zeros(len(X), dtype=int)

    if labels.size > 0 and np.all(labels == -1):
        labels = np.arange(len(X), dtype=int)
    return labels


def refine_overwide_clusters(
    cluster_map: Dict[int, Set[Path]],
    X: np.ndarray,
    owners: List[Path],
    max_cosine_dist: float = 0.38
) -> Dict[int, Set[Path]]:
    path2idx = {p: i for i, p in enumerate(owners)}
    new_map: Dict[int, Set[Path]] = {}
    next_id = 0

    for cid, paths in cluster_map.items():
        path_list = list(paths)
        filtered_paths = [p for p in path_list if p in path2idx]
        idxs = [path2idx[p] for p in filtered_paths]
        if len(idxs) <= 2:
            new_map[next_id] = set(filtered_paths)
            next_id += 1
            continue

        sub_X = X[idxs]
        D = cosine_distances(sub_X)
        tri = np.triu_indices_from(D, 1)
        maxd = float(np.max(D[tri])) if tri[0].size > 0 else 0.0

        if maxd <= max_cosine_dist:
            new_map[next_id] = set(filtered_paths)
            next_id += 1
            continue

        try:
            agg = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=max_cosine_dist,
                metric="cosine",
                linkage="average"
            )
            sub_labels = agg.fit_predict(sub_X)
            idxs_by_label = defaultdict(list)
            for local_idx, lbl in enumerate(sub_labels):
                idxs_by_label[lbl].append(local_idx)
            for sub_id, local_indices in idxs_by_label.items():
                sub_paths = {filtered_paths[idx] for idx in local_indices}
                if sub_paths:
                    new_map[next_id] = set(sub_paths)
                    next_id += 1
        except Exception:
            new_map[next_id] = set(filtered_paths)
            next_id += 1

    return new_map

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def _win_long(path: Path) -> str:
    p = str(path.resolve())
    if os.name == "nt":
        return "\\\\?\\" + p if not p.startswith("\\\\?\\") else p
    return p

def imread_safe(path: Path):
    return imread_exif_oriented(path)

def merge_clusters_by_centroid(
    embeddings: List[np.ndarray],
    owners: List[Path],
    raw_labels: np.ndarray,
    threshold: Optional[float] = DEFAULT_THRESHOLD,
    auto_threshold: bool = False,
    margin: float = 0.10,  # Более агрессивное значение для лучшего объединения
    min_threshold: float = 0.18,  # Более мягкий минимальный порог
    max_threshold: float = 0.45,  # Более высокий максимальный порог
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
            # Более консервативное объединение для точности
            threshold = max(min_threshold, min(mean_dist - margin, max_threshold))
        else:
            threshold = min_threshold

        if progress_callback:
            progress_callback(f"📏 Авто-порог объединения: {threshold:.3f}", 93)
    elif threshold is None:
        # Используем глобальный порог по умолчанию
        threshold = DEFAULT_THRESHOLD

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

    # Дополнительное объединение на основе максимального расстояния внутри кластеров
    if progress_callback:
        progress_callback("🔗 Дополнительное объединение похожих кластеров...", 94)
    
    # Вычисляем максимальное расстояние внутри каждого кластера
    cluster_max_distances = {}
    for label, embs in cluster_embeddings.items():
        if len(embs) > 1:
            distances = []
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    dist = cosine_distances([embs[i]], [embs[j]])[0][0]
                    distances.append(dist)
            cluster_max_distances[label] = max(distances) if distances else 0
        else:
            cluster_max_distances[label] = 0
    
    # Объединяем кластеры, если расстояние между их центрами меньше максимального расстояния внутри любого из них
    additional_merges = {}
    for i, label_i in enumerate(labels):
        if label_i in additional_merges:
            continue
        for j, label_j in enumerate(labels[i+1:], i+1):
            if label_j in additional_merges:
                continue
            dist = cosine_distances([centroids[label_i]], [centroids[label_j]])[0][0]
            max_internal_dist = max(cluster_max_distances[label_i], cluster_max_distances[label_j])
            
            # Более агрессивное объединение для лучшего распознавания
            if dist < max_internal_dist * 1.5:  # Увеличиваем буфер для более агрессивного объединения
                additional_merges[label_j] = label_i
            # Дополнительная проверка: для маленьких кластеров еще более мягкие условия
            elif (len(cluster_embeddings[label_i]) <= 4 and len(cluster_embeddings[label_j]) <= 4 and 
                  dist < SMART_SMALL_THRESHOLD):
                additional_merges[label_j] = label_i
    
    # Применяем дополнительные объединения
    for label_j, label_i in additional_merges.items():
        if label_i in label_to_group:
            label_to_group[label_j] = label_to_group[label_i]
        else:
            label_to_group[label_j] = label_to_group.get(label_i, next_cluster_id)
            if label_i not in label_to_group:
                label_to_group[label_i] = next_cluster_id
                next_cluster_id += 1
    
    # Итеративное объединение: повторяем процесс для оставшихся кластеров
    if progress_callback:
        progress_callback("🔄 Итеративное объединение оставшихся кластеров...", 95)
    
    # Создаем новые центроиды после первого объединения
    merged_centroids = {}
    for label, group_id in label_to_group.items():
        if group_id not in merged_centroids:
            merged_centroids[group_id] = centroids[label]
        else:
            # Усредняем центроиды объединенных кластеров
            merged_centroids[group_id] = (merged_centroids[group_id] + centroids[label]) / 2
    
    # Повторное объединение с новыми центроидами
    final_merges = {}
    merged_labels = list(merged_centroids.keys())
    for i, label_i in enumerate(merged_labels):
        if label_i in final_merges:
            continue
        for j, label_j in enumerate(merged_labels[i+1:], i+1):
            if label_j in final_merges:
                continue
            dist = cosine_distances([merged_centroids[label_i]], [merged_centroids[label_j]])[0][0]
            # Более агрессивный порог для финального объединения
            if dist < FINAL_MERGE_THRESHOLD:  # Более мягкий порог для лучшего объединения
                final_merges[label_j] = label_i
    
    # Применяем финальные объединения
    for label_j, label_i in final_merges.items():
        # Обновляем все кластеры, которые были связаны с label_j
        for original_label, group_id in label_to_group.items():
            if group_id == label_j:
                label_to_group[original_label] = label_to_group.get(label_i, label_i)

    merged_clusters: Dict[int, Set[Path]] = defaultdict(set)
    cluster_by_img: Dict[Path, Set[int]] = defaultdict(set)

    for label, path in zip(raw_labels, owners):
        if label == -1:
            continue
        new_label = label_to_group[label]
        merged_clusters[new_label].add(path)
        cluster_by_img[path].add(new_label)

    return merged_clusters, cluster_by_img

def validate_cluster_quality(embeddings_list: List[np.ndarray], threshold: float = 0.4) -> bool:
    """
    Проверяет качество кластера - все ли лица в нем достаточно похожи
    """
    if len(embeddings_list) <= 1:
        return True
    
    # Вычисляем все попарные расстояния внутри кластера
    distances = []
    for i in range(len(embeddings_list)):
        for j in range(i + 1, len(embeddings_list)):
            dist = cosine_distances([embeddings_list[i]], [embeddings_list[j]])[0][0]
            distances.append(dist)
    
    # Если максимальное расстояние больше порога, кластер плохой
    max_distance = max(distances) if distances else 0
    return max_distance < threshold

def post_process_clusters(
    cluster_map: Dict[int, Set[Path]], 
    embeddings: List[np.ndarray], 
    owners: List[Path],
    progress_callback=None
) -> Dict[int, Set[Path]]:
    """
    Дополнительная постобработка кластеров для объединения очень похожих лиц
    """
    if progress_callback:
        progress_callback("🔍 Постобработка кластеров...", 96)
    
    # Создаем маппинг путь -> эмбеддинг
    path_to_embedding = {}
    for emb, path in zip(embeddings, owners):
        path_to_embedding[path] = emb
    
    # Находим кластеры для объединения
    clusters_to_merge = []
    cluster_ids = list(cluster_map.keys())
    
    for i, cluster_id_i in enumerate(cluster_ids):
        if cluster_id_i in clusters_to_merge:
            continue
            
        paths_i = cluster_map[cluster_id_i]
        if len(paths_i) == 0:
            continue
            
        # Вычисляем центроид первого кластера
        embeddings_i = [path_to_embedding[p] for p in paths_i if p in path_to_embedding]
        if not embeddings_i:
            continue
        centroid_i = np.mean(embeddings_i, axis=0)
        
        for j, cluster_id_j in enumerate(cluster_ids[i+1:], i+1):
            if cluster_id_j in clusters_to_merge:
                continue
                
            paths_j = cluster_map[cluster_id_j]
            if len(paths_j) == 0:
                continue
                
            # Вычисляем центроид второго кластера
            embeddings_j = [path_to_embedding[p] for p in paths_j if p in path_to_embedding]
            if not embeddings_j:
                continue
            centroid_j = np.mean(embeddings_j, axis=0)
            
            # Проверяем расстояние между центроидами
            dist = cosine_distances([centroid_i], [centroid_j])[0][0]
            
            # Более агрессивный анализ для постобработки
            if dist < POSTPROCESS_THRESHOLD:
                # Дополнительная проверка: валидируем качество объединенного кластера
                combined_embeddings = embeddings_i + embeddings_j
                
                # Более мягкие пороги валидации для лучшего объединения
                validation_threshold = 0.35
                
                if validate_cluster_quality(combined_embeddings, threshold=validation_threshold):
                    clusters_to_merge.append((cluster_id_i, cluster_id_j))
                    print(f"✅ Объединяем кластеры {cluster_id_i} и {cluster_id_j} (расстояние: {dist:.3f})")
                else:
                    print(f"⚠️ Отклонено объединение кластеров {cluster_id_i} и {cluster_id_j} - низкое качество (расстояние: {dist:.3f})")
    
    # Объединяем найденные кластеры
    if clusters_to_merge:
        if progress_callback:
            progress_callback(f"🔗 Объединяем {len(clusters_to_merge)} пар кластеров...", 97)
        
        # Создаем граф объединений
        merge_groups = {}
        next_group_id = 0
        
        for cluster_a, cluster_b in clusters_to_merge:
            group_a = None
            group_b = None
            
            # Ищем существующие группы
            for group_id, clusters in merge_groups.items():
                if cluster_a in clusters:
                    group_a = group_id
                if cluster_b in clusters:
                    group_b = group_id
            
            if group_a is not None and group_b is not None:
                # Обе группы существуют - объединяем их
                if group_a != group_b:
                    merge_groups[group_a].extend(merge_groups[group_b])
                    del merge_groups[group_b]
            elif group_a is not None:
                # Добавляем cluster_b к существующей группе
                merge_groups[group_a].append(cluster_b)
            elif group_b is not None:
                # Добавляем cluster_a к существующей группе
                merge_groups[group_b].append(cluster_a)
            else:
                # Создаем новую группу
                merge_groups[next_group_id] = [cluster_a, cluster_b]
                next_group_id += 1
        
        # Применяем объединения
        final_cluster_map = {}
        used_clusters = set()
        
        # Сначала обрабатываем объединенные группы
        for group_clusters in merge_groups.values():
            if not group_clusters:
                continue
                
            # Выбираем первый кластер как основной
            main_cluster = group_clusters[0]
            final_cluster_map[main_cluster] = set()
            
            for cluster_id in group_clusters:
                if cluster_id in cluster_map:
                    final_cluster_map[main_cluster].update(cluster_map[cluster_id])
                    used_clusters.add(cluster_id)
        
        # Добавляем необъединенные кластеры
        for cluster_id, paths in cluster_map.items():
            if cluster_id not in used_clusters:
                final_cluster_map[cluster_id] = paths
        
        return final_cluster_map
    
    return cluster_map

def smart_final_merge(
    cluster_map: Dict[int, Set[Path]], 
    embeddings: List[np.ndarray], 
    owners: List[Path],
    progress_callback=None
) -> Dict[int, Set[Path]]:
    """
    Финальное умное объединение для случаев, когда один человек попал в разные кластеры
    """
    if progress_callback:
        progress_callback("🧠 Умное финальное объединение...", 98)
    
    # Создаем маппинг путь -> эмбеддинг
    path_to_embedding = {}
    for emb, path in zip(embeddings, owners):
        path_to_embedding[path] = emb
    
    # Анализируем кластеры по размеру - маленькие кластеры чаще всего нужно объединять
    small_clusters = []
    large_clusters = []
    
    for cluster_id, paths in cluster_map.items():
        if len(paths) <= 3:  # Маленькие кластеры
            small_clusters.append(cluster_id)
        else:
            large_clusters.append(cluster_id)
    
    print(f"🔍 Анализ кластеров: маленьких={len(small_clusters)}, больших={len(large_clusters)}")
    
    # Пытаемся объединить маленькие кластеры с большими или между собой
    merges_to_apply = []
    
    for small_id in small_clusters:
        small_paths = cluster_map[small_id]
        small_embeddings = [path_to_embedding[p] for p in small_paths if p in path_to_embedding]
        if not small_embeddings:
            continue
        small_centroid = np.mean(small_embeddings, axis=0)
        
        best_match = None
        best_distance = float('inf')
        
        # Сначала проверяем большие кластеры
        for large_id in large_clusters:
            if large_id == small_id:
                continue
            large_paths = cluster_map[large_id]
            large_embeddings = [path_to_embedding[p] for p in large_paths if p in path_to_embedding]
            if not large_embeddings:
                continue
            large_centroid = np.mean(large_embeddings, axis=0)
            
            dist = cosine_distances([small_centroid], [large_centroid])[0][0]
            if dist < SMART_LARGE_THRESHOLD and dist < best_distance:
                best_distance = dist
                best_match = large_id
        
        # Если не нашли подходящий большой кластер, проверяем другие маленькие
        if best_match is None:
            for other_small_id in small_clusters:
                if other_small_id == small_id or other_small_id in [m[1] for m in merges_to_apply]:
                    continue
                other_paths = cluster_map[other_small_id]
                other_embeddings = [path_to_embedding[p] for p in other_paths if p in path_to_embedding]
                if not other_embeddings:
                    continue
                other_centroid = np.mean(other_embeddings, axis=0)
                
                dist = cosine_distances([small_centroid], [other_centroid])[0][0]
                if dist < SMART_SMALL_THRESHOLD and dist < best_distance:
                    best_distance = dist
                    best_match = other_small_id
        
        if best_match is not None:
            merges_to_apply.append((small_id, best_match))
            print(f"🔗 Планируем объединить кластер {small_id} с {best_match} (расстояние: {best_distance:.3f})")
    
    # Применяем объединения
    if merges_to_apply:
        print(f"🔄 Применяем {len(merges_to_apply)} умных объединений...")
        final_cluster_map = cluster_map.copy()
        
        for source_id, target_id in merges_to_apply:
            if source_id in final_cluster_map and target_id in final_cluster_map:
                # Объединяем кластеры
                final_cluster_map[target_id].update(final_cluster_map[source_id])
                del final_cluster_map[source_id]
                print(f"✅ Объединили кластер {source_id} с {target_id}")
        
        return final_cluster_map
    
    return cluster_map

def super_aggressive_merge(
    cluster_map: Dict[int, Set[Path]], 
    embeddings: List[np.ndarray], 
    owners: List[Path],
    progress_callback=None
) -> Dict[int, Set[Path]]:
    """
    Супер-агрессивное объединение для максимального слияния похожих лиц
    """
    if progress_callback:
        progress_callback("🔥 Супер-агрессивное объединение...", 99)
    
    # Создаем маппинг путь -> эмбеддинг
    path_to_embedding = {}
    for emb, path in zip(embeddings, owners):
        path_to_embedding[path] = emb
    
    # Находим все пары кластеров для потенциального объединения
    cluster_ids = list(cluster_map.keys())
    merges_to_apply = []
    
    for i, cluster_id_i in enumerate(cluster_ids):
        paths_i = cluster_map[cluster_id_i]
        if len(paths_i) == 0:
            continue
            
        embeddings_i = [path_to_embedding[p] for p in paths_i if p in path_to_embedding]
        if not embeddings_i:
            continue
        centroid_i = np.mean(embeddings_i, axis=0)
        
        for j, cluster_id_j in enumerate(cluster_ids[i+1:], i+1):
            paths_j = cluster_map[cluster_id_j]
            if len(paths_j) == 0:
                continue
                
            embeddings_j = [path_to_embedding[p] for p in paths_j if p in path_to_embedding]
            if not embeddings_j:
                continue
            centroid_j = np.mean(embeddings_j, axis=0)
            
            # Проверяем расстояние между центроидами
            dist = cosine_distances([centroid_i], [centroid_j])[0][0]
            
            # Супер-агрессивный порог - объединяем практически все похожие лица
            if dist < SUPER_AGGRESSIVE_THRESHOLD:
                merges_to_apply.append((cluster_id_i, cluster_id_j))
                print(f"🔥 Супер-агрессивное объединение кластеров {cluster_id_i} и {cluster_id_j} (расстояние: {dist:.3f})")
    
    # Применяем объединения
    if merges_to_apply:
        print(f"🔥 Применяем {len(merges_to_apply)} супер-агрессивных объединений...")
        final_cluster_map = {cid: set(paths) for cid, paths in cluster_map.items()}

        parents = {cid: cid for cid in final_cluster_map.keys()}

        def find(x):
            if parents[x] != x:
                parents[x] = find(parents[x])
            return parents[x]

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            parents[rb] = ra
            if ra in final_cluster_map and rb in final_cluster_map:
                final_cluster_map[ra].update(final_cluster_map[rb])
                final_cluster_map.pop(rb, None)

        for cluster_a, cluster_b in merges_to_apply:
            if cluster_a in parents and cluster_b in parents:
                union(cluster_a, cluster_b)

        return final_cluster_map
    
    return cluster_map

def build_plan_live(
    input_dir: Path,
    det_size=(640, 640),
    min_score: float = 0.5,  # Сбалансированный порог для качества лиц
    min_cluster_size: int = 2,  # HDBSCAN требует минимум 2 элемента
    min_samples: int = 1,       # Минимальное количество образцов
    providers: List[str] = ("CPUExecutionProvider",),
    progress_callback=None,
):
    input_dir = Path(input_dir)
    # Собираем все изображения, исключая те, что находятся в папках с нежелательными именами
    excluded_names = ["общие", "общая", "common", "shared", "все", "all", "mixed", "смешанные"]
    all_images = [
        p for p in input_dir.rglob("*")
        if is_image(p)
        and not any(ex in str(p).lower() for ex in excluded_names)
    ]

    if progress_callback:
        progress_callback(f"📂 Сканируется: {input_dir}, найдено изображений: {len(all_images)}", 1)

    app_base, app_hi = init_dual_apps(
        providers,
        det_size_base=det_size,
        det_size_hi=(max(960, det_size[0] * 2), max(960, det_size[1] * 2))
    )

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
        
        print(f"🔍 Обрабатываем изображение {i+1}/{total}: {p.name}")
        
        img = imread_exif_oriented(p)
        if img is None:
            print(f"❌ Не удалось прочитать изображение: {p.name}")
            unreadable.append(p)
            continue
            
        print(f"✅ Изображение загружено, размер: {img.shape}")
        faces = detect_faces_multi(
            app_base,
            app_hi,
            img,
            min_score_main=min_score,
            min_score_fallback=min(0.35, max(0.3, min_score * 0.8))
        )
        print(f"🔍 Найдено лиц: {len(faces) if faces else 0}")
        
        if not faces:
            print(f"⚠️ Лица не найдены в: {p.name}")
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
            
            # Улучшенная нормализация эмбеддингов
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
                # Дополнительная проверка качества эмбеддинга
                if np.any(np.isnan(emb)) or np.any(np.isinf(emb)):
                    continue
                # Проверяем, что эмбеддинг не слишком близок к нулю
                if np.max(np.abs(emb)) < 1e-6:
                    continue
                    
            embeddings.append(emb)
            owners.append(p)
            count += 1
            processed_faces += 1

        if count > 0:
            img_face_count[p] = count
            print(f"✅ Обработано лиц в {p.name}: {count}")

    print(f"📊 Итого обработано изображений: {len(all_images)}")
    print(f"📊 Найдено эмбеддингов: {len(embeddings)}")
    print(f"📊 Нечитаемых файлов: {len(unreadable)}")
    print(f"📊 Файлов без лиц: {len(no_faces)}")

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
    print(f"🔄 Начинаем гибридную кластеризацию {len(embeddings)} лиц...")
    if progress_callback:
        progress_callback(f"🔄 Кластеризация {len(embeddings)} лиц (OPTICS→HDBSCAN)", 80)

    X = np.vstack(embeddings).astype(np.float64)
    raw_labels = cluster_with_optics_hybrid(X, progress_callback=progress_callback)

    cluster_map, cluster_by_img = merge_clusters_by_centroid(
        embeddings=embeddings,
        owners=owners,
        raw_labels=raw_labels,
        auto_threshold=True,
        margin=0.10,  # Более агрессивное значение для лучшего объединения
        min_threshold=0.18,  # Более мягкий минимальный порог
        max_threshold=0.45,  # Более высокий максимальный порог
        progress_callback=progress_callback
    )
    
    # Дополнительная постобработка для объединения очень похожих кластеров
    cluster_map = post_process_clusters(
        cluster_map=cluster_map,
        embeddings=embeddings,
        owners=owners,
        progress_callback=progress_callback
    )
    
    # Финальное умное объединение для решения проблемы разделения одного человека
    cluster_map = smart_final_merge(
        cluster_map=cluster_map,
        embeddings=embeddings,
        owners=owners,
        progress_callback=progress_callback
    )
    
    # Супер-агрессивное объединение как последний этап (можно отключить при необходимости)
    cluster_map = super_aggressive_merge(
        cluster_map=cluster_map,
        embeddings=embeddings,
        owners=owners,
        progress_callback=progress_callback
    )

    # Дополнительная защита от "слишком широких" кластеров
    cluster_map = refine_overwide_clusters(cluster_map, X, owners)
    
    # Обновляем cluster_by_img после всех объединений
    cluster_by_img = defaultdict(set)
    for cluster_id, paths in cluster_map.items():
        for path in paths:
            cluster_by_img[path].add(cluster_id)

    # Этап 3: Формирование плана распределения
    print(f"🔄 Формируем план распределения...")
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
    
    print(f"📋 План распределения: {len(plan)} файлов")

    # Если по какой-то причине план пуст, но эмбеддинги были — переносим все изображения с лицами в один кластер
    if not plan and embeddings:
        if progress_callback:
            progress_callback("⚠️ План пуст. Переносим все изображения с лицами в один кластер (резервный режим)", 96)
        fallback_cluster_id = 0
        img_with_faces = [p for p, cnt in img_face_count.items() if cnt > 0]
        for p in img_with_faces:
            plan.append({
                "path": str(p),
                "cluster": [fallback_cluster_id],
                "faces": img_face_count.get(p, 0)
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
        # Логируем прогресс обработки подпапок
        if progress_callback:
            percent = 10 + int((i + 1) / max(total_subfolders, 1) * 80)
            progress_callback(f"🔍 Обрабатывается подпапка: {subfolder.name} ({i+1}/{total_subfolders})", percent)

        print(f"🔍 Обрабатывается подпапка: {subfolder}")
        # Кластеризация с передачей коллбэка для логов
        plan = build_plan_live(subfolder, progress_callback=progress_callback)
        # Логируем результат кластеризации
        clusters_count = len(plan.get('clusters', {}))
        items_count = len(plan.get('plan', []))
        print(f"📊 Подпапка: {subfolder.name} → кластеров: {clusters_count}, файлов: {items_count}")
        if progress_callback:
            progress_callback(f"📊 Результат: кластеров={clusters_count}, файлов={items_count}", percent=percent + 1)

        # Распределение по папкам с передачей коллбэка
        moved, copied, cluster_counter = distribute_to_folders(
            plan,
            subfolder,
            cluster_start=cluster_counter,
            progress_callback=progress_callback
        )
        # Логирование переноса
        print(f"📦 Перенесено: {moved}, скопировано: {copied} в подпапке {subfolder.name}")
        if progress_callback:
            progress_callback(f"📦 Перенесено={moved}, скопировано={copied}", percent=90)



