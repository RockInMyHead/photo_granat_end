from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import zipfile
import shutil
import asyncio
from pathlib import Path
import psutil
from PIL import Image
import uuid
import time
import tempfile

from cluster import build_plan_live, distribute_to_folders, process_group_folder, IMG_EXTS

app = FastAPI(title="Кластеризация лиц", description="API для кластеризации лиц и распределения по группам")

# CORS middleware для поддержки фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Состояние приложения (в продакшене стоит использовать Redis/Database)
app_state = {
    "queue": [],
    "current_tasks": {},
    "task_history": []
}

# Модели данных
class FolderInfo(BaseModel):
    path: str
    name: str
    is_directory: bool
    size: Optional[int] = None
    image_count: Optional[int] = None

class QueueItem(BaseModel):
    path: str

class TaskProgress(BaseModel):
    task_id: str
    status: str  # "pending", "running", "completed", "error"
    progress: int
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ProcessingResult(BaseModel):
    moved: int
    copied: int
    clusters_count: int
    unreadable_count: int
    no_faces_count: int
    unreadable_files: List[str]
    no_faces_files: List[str]

class MoveItem(BaseModel):
    src: str
    dest: str

# Утилиты
def get_logical_drives():
    """Получить список логических дисков"""
    return [Path(p.mountpoint) for p in psutil.disk_partitions(all=False) if Path(p.mountpoint).exists()]

def get_special_dirs():
    """Получить специальные директории"""
    home = Path.home()
    return {
        "💼 Рабочий стол": home / "Desktop",
        "📄 Документы": home / "Documents", 
        "📥 Загрузки": home / "Downloads",
        "🖼 Изображения": home / "Pictures",
    }

def count_images_in_dir(path: Path) -> int:
    """Подсчитать количество изображений в директории"""
    try:
        return len([f for f in path.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS])
    except:
        return 0

def get_folder_contents(path: Path) -> List[FolderInfo]:
    """Получить содержимое папки"""
    try:
        contents = []
        
        # Добавляем родительскую папку если не корень
        if path.parent != path:
            contents.append(FolderInfo(
                path=str(path.parent),
                name="⬅️ Назад",
                is_directory=True
            ))
        
        # Добавляем подпапки
        for item in sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if item.is_dir():
                image_count = count_images_in_dir(item)
                contents.append(FolderInfo(
                    path=str(item),
                    name=f"📂 {item.name}",
                    is_directory=True,
                    image_count=image_count
                ))
            elif item.suffix.lower() in IMG_EXTS:
                try:
                    size = item.stat().st_size
                    contents.append(FolderInfo(
                        path=str(item),
                        name=f"🖼 {item.name}",
                        is_directory=False,
                        size=size
                    ))
                except:
                    pass
        
        return contents
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Нет доступа к папке: {path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при чтении папки: {str(e)}")

async def process_folder_task(task_id: str, folder_path: str):
    """Фоновая задача обработки папки"""
    try:
        app_state["current_tasks"][task_id]["status"] = "running"
        app_state["current_tasks"][task_id]["message"] = "Начинаем обработку..."
        
        path = Path(folder_path)
        if not path.exists():
            raise Exception("Путь не существует")
        
        # Определяем тип обработки
        has_subdirs = any(p.is_dir() and "общие" not in str(p).lower() for p in path.iterdir())
        
        if has_subdirs:
            # Групповая обработка
            app_state["current_tasks"][task_id]["message"] = "Групповая обработка папок..."
            process_group_folder(path)
            result = ProcessingResult(
                moved=0, copied=0, clusters_count=0,
                unreadable_count=0, no_faces_count=0,
                unreadable_files=[], no_faces_files=[]
            )
        else:
            # Обычная кластеризация
            def progress_callback(progress_text: str):
                if task_id in app_state["current_tasks"]:
                    app_state["current_tasks"][task_id]["message"] = progress_text
                    # Попытка извлечь процент из текста
                    try:
                        if "%" in progress_text:
                            percent_str = progress_text.split("%")[0].split()[-1]
                            app_state["current_tasks"][task_id]["progress"] = int(percent_str)
                    except:
                        pass
            
            app_state["current_tasks"][task_id]["message"] = "Кластеризация лиц..."
            plan = build_plan_live(path, progress_callback=lambda text: progress_callback(text))
            
            app_state["current_tasks"][task_id]["message"] = "Распределение по папкам..."
            app_state["current_tasks"][task_id]["progress"] = 90
            
            moved, copied, _ = distribute_to_folders(plan, path)
            
            result = ProcessingResult(
                moved=moved,
                copied=copied, 
                clusters_count=len(plan.get("clusters", {})),
                unreadable_count=len(plan.get("unreadable", [])),
                no_faces_count=len(plan.get("no_faces", [])),
                unreadable_files=plan.get("unreadable", [])[:30],
                no_faces_files=plan.get("no_faces", [])[:30]
            )
        
        app_state["current_tasks"][task_id]["status"] = "completed"
        app_state["current_tasks"][task_id]["progress"] = 100
        app_state["current_tasks"][task_id]["message"] = "Обработка завершена"
        app_state["current_tasks"][task_id]["result"] = result.dict()
        
    except Exception as e:
        app_state["current_tasks"][task_id]["status"] = "error"
        app_state["current_tasks"][task_id]["error"] = str(e)
        app_state["current_tasks"][task_id]["message"] = f"Ошибка: {str(e)}"

# API endpoints
@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Главная страница"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/api/drives")
async def get_drives():
    """Получить список дисков и специальных папок"""
    drives = []
    
    # Логические диски
    for drive in get_logical_drives():
        drives.append({
            "path": str(drive),
            "name": f"📍 {drive}",
            "type": "drive"
        })
    
    # Специальные папки
    for name, path in get_special_dirs().items():
        if path.exists():
            drives.append({
                "path": str(path),
                "name": name,
                "type": "special"
            })
    
    return drives

@app.get("/api/folder")
async def get_folder_info(path: str):
    """Получить содержимое папки"""
    folder_path = Path(path)
    if not folder_path.exists():
        raise HTTPException(status_code=404, detail="Папка не найдена")
    
    contents = get_folder_contents(folder_path)
    image_count = count_images_in_dir(folder_path)
    
    return {
        "path": str(folder_path),
        "contents": contents,
        "image_count": image_count
    }

@app.post("/api/upload")
async def upload_files(
    path: str,
    files: List[UploadFile] = File(...)
):
    """Загрузить файлы в указанную папку"""
    target_dir = Path(path)
    if not target_dir.exists():
        raise HTTPException(status_code=404, detail="Целевая папка не найдена")
    
    results = []
    
    for file in files:
        try:
            if file.filename.endswith(".zip"):
                # Обработка ZIP архива
                temp_zip = target_dir / f"temp_{uuid.uuid4().hex}.zip"
                with open(temp_zip, "wb") as f:
                    content = await file.read()
                    f.write(content)
                
                with zipfile.ZipFile(temp_zip) as archive:
                    archive.extractall(target_dir)
                
                temp_zip.unlink()
                results.append({"filename": file.filename, "status": "extracted"})
            else:
                # Обычный файл
                file_path = target_dir / file.filename
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                results.append({"filename": file.filename, "status": "uploaded"})
                
        except Exception as e:
            results.append({"filename": file.filename, "status": "error", "error": str(e)})
    
    return {"results": results}

@app.get("/api/queue")
async def get_queue():
    """Получить текущую очередь обработки"""
    return {"queue": app_state["queue"]}

@app.post("/api/queue/add")
async def add_to_queue(item: QueueItem):
    """Добавить папку в очередь"""
    if item.path not in app_state["queue"]:
        app_state["queue"].append(item.path)
        return {"message": f"Папка добавлена в очередь: {item.path}"}
    else:
        return {"message": "Папка уже в очереди"}

@app.delete("/api/queue")
async def clear_queue():
    """Очистить очередь"""
    app_state["queue"].clear()
    return {"message": "Очередь очищена"}

@app.post("/api/process")
async def process_queue(background_tasks: BackgroundTasks):
    """Запустить обработку очереди"""
    if not app_state["queue"]:
        raise HTTPException(status_code=400, detail="Очередь пуста")
    
    task_ids = []
    
    for folder_path in app_state["queue"]:
        task_id = str(uuid.uuid4())
        app_state["current_tasks"][task_id] = {
            "task_id": task_id,
            "status": "pending",
            "progress": 0,
            "message": "В очереди...",
            "folder_path": folder_path,
            "created_at": time.time()
        }
        
        background_tasks.add_task(process_folder_task, task_id, folder_path)
        task_ids.append(task_id)
    
    app_state["queue"].clear()
    return {"message": "Обработка запущена", "task_ids": task_ids}

@app.get("/api/tasks")
async def get_tasks():
    """Получить статус всех задач"""
    return {"tasks": list(app_state["current_tasks"].values())}

@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    """Получить статус конкретной задачи"""
    if task_id not in app_state["current_tasks"]:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    return app_state["current_tasks"][task_id]

@app.get("/api/image/preview")
async def get_image_preview(path: str, size: int = 100):
    """Получить превью изображения"""
    img_path = Path(path)
    if not img_path.exists() or img_path.suffix.lower() not in IMG_EXTS:
        raise HTTPException(status_code=404, detail="Изображение не найдено")
    
    try:
        # Создаем временное превью
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img.thumbnail((size, size), Image.Resampling.LANCZOS)
            
            # Сохраняем во временный файл
            preview_path = Path(f"/tmp/preview_{uuid.uuid4().hex}.jpg")
            img.save(preview_path, "JPEG", quality=85)
            
            # Читаем и возвращаем файл
            with open(preview_path, "rb") as f:
                content = f.read()
            
            preview_path.unlink()  # Удаляем временный файл
            
            from fastapi.responses import Response
            return Response(content=content, media_type="image/jpeg")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка создания превью: {str(e)}")

@app.get("/api/zip")
async def zip_folder(path: str):
    """Создает ZIP архивацию указанной папки и возвращает файл"""
    folder = Path(path)
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=404, detail="Папка не найдена")
    # Создаем временный zip-файл
    tmp_dir = tempfile.gettempdir()
    zip_name = f"{uuid.uuid4()}.zip"
    zip_path = Path(tmp_dir) / zip_name
    # Делает архив
    shutil.make_archive(str(zip_path.with_suffix('')), 'zip', root_dir=folder)
    # Отдает файл для скачивания
    return FileResponse(str(zip_path), media_type="application/zip", filename=f"{folder.name}.zip")

# Статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/api/move")
async def move_item(item: MoveItem):
    """Переместить файл или папку"""
    src_path = Path(item.src)
    dest_path = Path(item.dest)
    if not src_path.exists():
        raise HTTPException(status_code=404, detail="Источник не найден")
    if not dest_path.exists():
        raise HTTPException(status_code=404, detail="Назначение не найдено")
    
    target = dest_path / src_path.name
    try:
        shutil.move(str(src_path), str(target))
        return {"message": "Успешно перемещено", "src": str(src_path), "dest": str(target)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка перемещения: {e}")

@app.get("/favicon.ico")
async def favicon():
    """Возвращает простой favicon чтобы избежать 404 ошибок"""
    return Response(content="", media_type="image/x-icon")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
