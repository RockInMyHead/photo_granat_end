import os
import cv2
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Tuple
from sklearn.metrics.pairwise import cosine_distances
from insightface.app import FaceAnalysis
import hdbscan

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

    print(f"📂 Сканируется: {input_dir}, найдено изображений: {len(all_images)}")

    app = FaceAnalysis(name="buffalo_l", providers=list(providers))
    ctx_id = -1 if "cpu" in str(providers).lower() else 0
    app.prepare(ctx_id=ctx_id, det_size=det_size)

    embeddings = []
    owners = []
    img_face_count = {}
    unreadable = []
    no_faces = []

    total = len(all_images)
    for i, p in enumerate(all_images):
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

        if count > 0:
            img_face_count[p] = count

        if progress_callback:
            percent = int((i + 1) / max(total, 1) * 100)
            bar = int(percent / 2) * "█"
            progress_callback.text(f"📷 Scanning: {percent}%|{bar:<50}| {i+1}/{total}")

    if not embeddings:
        print(f"⚠️ Нет эмбеддингов: {input_dir}")
        return {
            "clusters": {},
            "plan": [],
            "unreadable": [str(p) for p in unreadable],
            "no_faces": [str(p) for p in no_faces],
        }

    X = np.vstack(embeddings)
    distance_matrix = cosine_distances(X)

    model = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=min_cluster_size, min_samples=min_samples)
    raw_labels = model.fit_predict(distance_matrix)

    label_map = {label: idx + 1 for idx, label in enumerate(sorted(set(raw_labels) - {-1}))}  # start from 1

    cluster_map: Dict[int, Set[Path]] = {}
    cluster_by_img: Dict[Path, Set[int]] = {}

    for lbl, path in zip(raw_labels, owners):
        if lbl == -1:
            continue
        new_lbl = label_map[lbl]
        cluster_map.setdefault(new_lbl, set()).add(path)
        cluster_by_img.setdefault(path, set()).add(new_lbl)

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

def distribute_to_folders(plan: dict, base_dir: Path, cluster_start: int = 1) -> Tuple[int, int, int]:
    moved, copied = 0, 0
    moved_paths = set()

    used_clusters = sorted({c for item in plan.get("plan", []) for c in item["cluster"]})
    cluster_id_map = {old: cluster_start + idx for idx, old in enumerate(used_clusters)}

    for item in plan.get("plan", []):
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

    for p in sorted(moved_paths, key=lambda x: len(str(x)), reverse=True):
        try:
            if p.exists() and not any(p.iterdir()):
                p.rmdir()
        except Exception:
            pass

    print(f"📦 Перемещено: {moved}, скопировано: {copied}")
    return moved, copied, cluster_start + len(used_clusters)

def process_group_folder(group_dir: Path):
    cluster_counter = 1
    for subfolder in sorted(group_dir.iterdir()):
        if not subfolder.is_dir():
            continue
        if "общие" in subfolder.name.lower():
            continue

        print(f"🔍 Обрабатывается подпапка: {subfolder}")
        plan = build_plan_live(subfolder)
        print(f"📊 Кластеров: {len(plan.get('clusters', {}))}, файлов: {len(plan.get('plan', []))}")
        moved, copied, cluster_counter = distribute_to_folders(plan, subfolder, cluster_start=cluster_counter)



