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
    margin: float = 0.07,  # Компромиссное значение между агрессивным и консервативным
    min_threshold: float = 0.20,  # Умеренный минимальный порог
    max_threshold: float = 0.40,  # Умеренный максимальный порог
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
        # Сбалансированный порог по умолчанию
        threshold = 0.32

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
            
            # Сбалансированное объединение
            if dist < max_internal_dist * 1.3:  # Увеличиваем буфер для лучшего объединения
                additional_merges[label_j] = label_i
            # Дополнительная проверка: для маленьких кластеров более мягкие условия
            elif (len(cluster_embeddings[label_i]) <= 3 and len(cluster_embeddings[label_j]) <= 3 and 
                  dist < 0.32):  # Более мягкий порог для маленьких кластеров
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
            # Сбалансированный порог для финального объединения
            if dist < 0.28:  # Компромиссный порог
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
            
            # Умный анализ для постобработки
            if dist < 0.25:  # Более мягкий порог для начальной проверки
                # Дополнительная проверка: валидируем качество объединенного кластера
                combined_embeddings = embeddings_i + embeddings_j
                
                # Адаптивный порог валидации в зависимости от размера кластеров
                validation_threshold = 0.4 if (len(embeddings_i) <= 2 or len(embeddings_j) <= 2) else 0.35
                
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
            if dist < 0.35 and dist < best_distance:  # Более мягкий порог для объединения с большими кластерами
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
                if dist < 0.30 and dist < best_distance:  # Строже для объединения маленьких кластеров
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

def build_plan_live(
    input_dir: Path,
    det_size=(640, 640),
    min_score: float = 0.5,  # Сбалансированный порог для качества лиц
    min_cluster_size: int = 1,  # Разрешаем кластеры из 1 элемента
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
        
        print(f"🔍 Обрабатываем изображение {i+1}/{total}: {p.name}")
        
        img = imread_safe(p)
        if img is None:
            print(f"❌ Не удалось прочитать изображение: {p.name}")
            unreadable.append(p)
            continue
            
        print(f"✅ Изображение загружено, размер: {img.shape}")
        faces = app.get(img)
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
    print(f"🔄 Начинаем кластеризацию {len(embeddings)} лиц...")
    if progress_callback:
        progress_callback(f"🔄 Кластеризация {len(embeddings)} лиц...", 80)
    
    X = np.vstack(embeddings)
    print(f"📐 Создаем матрицу расстояний для {X.shape[0]} эмбеддингов...")
    
    # Оптимизированное создание матрицы расстояний
    if X.shape[0] > 50:
        print("⚠️ Большое количество эмбеддингов, используем оптимизированный алгоритм...")
        # Для больших наборов используем более эффективный подход
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(X)
        distance_matrix = 1 - similarity_matrix
    else:
        distance_matrix = cosine_distances(X)
    
    print(f"✅ Матрица расстояний создана: {distance_matrix.shape}")

    if progress_callback:
        progress_callback("🔄 Вычисление матрицы расстояний...", 85)

    print("🔄 Запускаем HDBSCAN...")
    try:
        # Добавляем таймаут для HDBSCAN
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("HDBSCAN timeout")
        
        # Устанавливаем таймаут в 5 минут
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(300)  # 5 минут
        
        model = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=min_cluster_size, min_samples=min_samples)
        raw_labels = model.fit_predict(distance_matrix)
        
        signal.alarm(0)  # Отключаем таймаут
        print(f"✅ HDBSCAN завершен. Уникальные метки: {np.unique(raw_labels)}")
        
    except TimeoutError:
        print("⚠️ HDBSCAN timeout! Используем альтернативную кластеризацию...")
        # Fallback: используем KMeans как альтернативу
        try:
            from sklearn.cluster import KMeans
            # Определяем количество кластеров (примерно 1 кластер на 3-5 лиц)
            n_clusters = max(1, min(len(embeddings) // 3, 20))
            print(f"🔄 KMeans с {n_clusters} кластерами...")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            raw_labels = kmeans.fit_predict(X)
            print(f"✅ KMeans завершен. Уникальные метки: {np.unique(raw_labels)}")
        except Exception as e2:
            print(f"❌ Ошибка KMeans: {e2}. Используем простые кластеры...")
            raw_labels = np.arange(len(embeddings), dtype=int)
    except Exception as e:
        print(f"❌ Ошибка HDBSCAN: {e}. Используем альтернативную кластеризацию...")
        try:
            from sklearn.cluster import KMeans
            n_clusters = max(1, min(len(embeddings) // 3, 20))
            print(f"🔄 KMeans с {n_clusters} кластерами...")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            raw_labels = kmeans.fit_predict(X)
            print(f"✅ KMeans завершен. Уникальные метки: {np.unique(raw_labels)}")
        except Exception as e2:
            print(f"❌ Ошибка KMeans: {e2}. Используем простые кластеры...")
            raw_labels = np.arange(len(embeddings), dtype=int)

    # Fallback: если HDBSCAN пометил все точки как шум, используем уникальные кластеры,
    # которые затем будут слиты нашими этапами объединения
    if raw_labels.size > 0 and np.all(raw_labels == -1):
        if progress_callback:
            progress_callback("⚠️ Все точки помечены как шум HDBSCAN. Включаем резервный режим кластеризации.", 82)
        raw_labels = np.arange(len(embeddings), dtype=int)

    cluster_map, cluster_by_img = merge_clusters_by_centroid(
        embeddings=embeddings,
        owners=owners,
        raw_labels=raw_labels,
        auto_threshold=True,
        margin=0.07,  # Сбалансированное значение
        min_threshold=0.20,  # Умеренный минимальный порог
        max_threshold=0.40,  # Умеренный максимальный порог
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



