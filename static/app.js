class PhotoClusterApp {
    constructor() {
        this.currentPath = '';
        this.queue = [];
        this.lastTasksStr = '';
        
        this.initializeElements();
        this.setupEventListeners();
        this.loadInitialData();
        this.startTaskPolling();
    }

    initializeElements() {
        this.driveButtons = document.getElementById('driveButtons');
        this.currentPathEl = document.getElementById('currentPath');
        this.folderContents = document.getElementById('folderContents');
        this.uploadZone = document.getElementById('uploadZone');
        this.fileInput = document.getElementById('fileInput');
        this.queueList = document.getElementById('queueList');
        this.processBtn = document.getElementById('processBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.addQueueBtn = document.getElementById('addQueueBtn');
        this.tasksList = document.getElementById('tasksList');
    }

    setupEventListeners() {
        // Разрешить drop в очередь
        this.queueList.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.queueList.classList.add('drag-over');
        });
        this.queueList.addEventListener('dragleave', (e) => {
            e.preventDefault();
            this.queueList.classList.remove('drag-over');
        });
        this.queueList.addEventListener('drop', (e) => {
            e.preventDefault();
            this.queueList.classList.remove('drag-over');
            const path = e.dataTransfer.getData('text/plain');
            if (path) this.addToQueue(path);
        });
        // Кнопки обработки очереди
        this.processBtn.addEventListener('click', () => this.processQueue());
        this.clearBtn.addEventListener('click', () => this.clearQueue());
        // Кнопка добавить в очередь
        this.addQueueBtn.addEventListener('click', () => this.addToQueue(this.currentPath));

        // Загрузка файлов
        this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e.target.files));

        // Drag & Drop
        this.uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadZone.classList.add('drag-over');
        });

        this.uploadZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            this.uploadZone.classList.remove('drag-over');
        });

        this.uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadZone.classList.remove('drag-over');
            this.handleFileUpload(e.dataTransfer.files);
        });
    }

    async loadInitialData() {
        await this.loadDrives();
        await this.loadQueue();
    }

    async loadDrives() {
        try {
            const response = await fetch('/api/drives');
            const drives = await response.json();
            
            this.driveButtons.innerHTML = '';
            drives.forEach(drive => {
                const button = document.createElement('button');
                button.className = 'drive-btn';
                button.textContent = drive.name;
                button.addEventListener('click', () => this.navigateToFolder(drive.path));
                this.driveButtons.appendChild(button);
            });
        } catch (error) {
            this.showNotification('Ошибка загрузки дисков: ' + error.message, 'error');
        }
    }

    async navigateToFolder(path) {
        try {
            this.currentPath = path;
            const response = await fetch(`/api/folder?path=${encodeURIComponent(path)}`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            this.currentPathEl.innerHTML = `<strong>Текущая папка:</strong> ${path}`;
            await this.displayFolderContents(data.contents);
            
        } catch (error) {
            this.showNotification('Ошибка доступа к папке: ' + error.message, 'error');
        }
    }

    async displayFolderContents(contents) {
        this.folderContents.innerHTML = '';
        
        if (contents.length === 0) {
            this.folderContents.innerHTML = `
                <p style="text-align: center; color: #666; padding: 40px 0;">
                    Папка пуста
                </p>
            `;
            return;
        }

        for (const item of contents) {
            // Навигационная кнопка Назад
            if (item.name.includes('⬅️')) {
                const button = document.createElement('button');
                button.className = 'folder-btn back';
                button.setAttribute('draggable', 'true');
                button.addEventListener('dragstart', (e) => {
                    e.dataTransfer.setData('text/plain', item.path);
                    e.dataTransfer.effectAllowed = 'move';
                });
                button.textContent = item.name;
                if (item.is_directory) button.addEventListener('click', () => this.navigateToFolder(item.path));
                this.folderContents.appendChild(button);
                continue;
            }
            if (item.is_directory) {
                // Папка: если есть изображения, показываем превью, иначе кнопка
                let imgs = [];
                try {
                    const res = await fetch(`/api/folder?path=${encodeURIComponent(item.path)}`);
                    const folderData = await res.json();
                    imgs = folderData.contents.filter(c => !c.is_directory);
                } catch {}
                if (imgs.length > 0) {
                    // Превью папки
                    const div = document.createElement('div');
                    div.className = 'thumbnail';
                    div.setAttribute('draggable','true');
                    div.addEventListener('click', () => this.navigateToFolder(item.path));
                    
                    // Drag & Drop для папки
                    div.addEventListener('dragstart', e => {
                        e.dataTransfer.setData('text/plain', item.path);
                        e.dataTransfer.effectAllowed = 'move';
                    });
                    div.addEventListener('dragover', e => {
                        e.preventDefault();
                        div.classList.add('drag-over');
                    });
                    div.addEventListener('dragleave', e => {
                        e.preventDefault();
                        div.classList.remove('drag-over');
                    });
                    div.addEventListener('drop', e => {
                        e.preventDefault();
                        div.classList.remove('drag-over');
                        const src = e.dataTransfer.getData('text/plain');
                        this.moveItem(src, item.path);
                    });
                    
                    const img = document.createElement('img');
                    img.src = `/api/image/preview?path=${encodeURIComponent(imgs[0].path)}&size=150`;
                    img.alt = item.name.replace('📂 ', '');
                    div.appendChild(img);
                    
                    // Добавляем подпись с названием папки
                    const caption = document.createElement('div');
                    caption.className = 'thumbnail-caption';
                    caption.textContent = item.name.replace('📂 ', '');
                    div.appendChild(caption);
                    
                    this.folderContents.appendChild(div);
                } else {
                    // Обычная папка без превью
                    const button = document.createElement('button');
                    button.className = 'folder-btn';
                    button.textContent = item.name.replace('📂 ', '');
                    button.addEventListener('click', () => this.navigateToFolder(item.path));
                    
                    // Drag & Drop для обычной папки
                    button.setAttribute('draggable', 'true');
                    button.addEventListener('dragstart', e => {
                        e.dataTransfer.setData('text/plain', item.path);
                        e.dataTransfer.effectAllowed = 'move';
                    });
                    button.addEventListener('dragover', e => {
                        e.preventDefault();
                        button.classList.add('drag-over');
                    });
                    button.addEventListener('dragleave', e => {
                        e.preventDefault();
                        button.classList.remove('drag-over');
                    });
                    button.addEventListener('drop', e => {
                        e.preventDefault();
                        button.classList.remove('drag-over');
                        const src = e.dataTransfer.getData('text/plain');
                        this.moveItem(src, item.path);
                    });
                    
                    this.folderContents.appendChild(button);
                }
                continue;
            }
            // Изображение файла
            if (!item.is_directory && item.name.match(/\.(jpg|jpeg|png|bmp|tif|tiff|webp)$/i)) {
                const div = document.createElement('div');
                div.className = 'thumbnail';
                div.setAttribute('draggable', 'true');
                
                // Drag & Drop для изображения
                div.addEventListener('dragstart', e => {
                    e.dataTransfer.setData('text/plain', item.path);
                    e.dataTransfer.effectAllowed = 'move';
                });
                div.addEventListener('dragover', e => {
                    e.preventDefault();
                    div.classList.add('drag-over');
                });
                div.addEventListener('dragleave', e => {
                    e.preventDefault();
                    div.classList.remove('drag-over');
                });
                div.addEventListener('drop', e => {
                    e.preventDefault();
                    div.classList.remove('drag-over');
                    const src = e.dataTransfer.getData('text/plain');
                    this.moveItem(src, item.path);
                });
                
                const img = document.createElement('img');
                img.src = `/api/image/preview?path=${encodeURIComponent(item.path)}&size=150`;
                img.alt = item.name.replace('🖼 ', '');
                div.appendChild(img);
                
                // Добавляем подпись с названием файла
                const caption = document.createElement('div');
                caption.className = 'thumbnail-caption';
                caption.textContent = item.name.replace('🖼 ', '');
                div.appendChild(caption);
                
                this.folderContents.appendChild(div);
                continue;
            }
            // Другие файлы: просто кнопка
            const button = document.createElement('button');
            button.className = 'folder-btn';
            button.textContent = item.name;
            this.folderContents.appendChild(button);
        }

        // Добавляем кнопку "Добавить в очередь" если это не навигационная кнопка
        if (!contents.some(item => item.name.includes('⬅️'))) {
            const addButton = document.createElement('button');
            addButton.className = 'action-btn';
            addButton.style.marginTop = '15px';
            addButton.textContent = '📌 Добавить в очередь';
            addButton.addEventListener('click', () => this.addToQueue(this.currentPath));
            this.folderContents.appendChild(addButton);
        }
    }

    formatFileSize(bytes) {
        const sizes = ['Б', 'КБ', 'МБ', 'ГБ'];
        if (bytes === 0) return '0 Б';
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
    }

    async handleFileUpload(files) {
        if (!this.currentPath) {
            this.showNotification('Выберите папку для загрузки файлов', 'error');
            return;
        }

        const formData = new FormData();
        for (let file of files) {
            formData.append('files', file);
        }

        try {
            const response = await fetch(`/api/upload?path=${encodeURIComponent(this.currentPath)}`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            let successCount = 0;
            let errorCount = 0;
            
            result.results.forEach(item => {
                if (item.status === 'uploaded' || item.status === 'extracted') {
                    successCount++;
                } else {
                    errorCount++;
                }
            });

            if (successCount > 0) {
                this.showNotification(`Загружено файлов: ${successCount}`, 'success');
                // Обновляем содержимое папки
                this.navigateToFolder(this.currentPath);
            }
            
            if (errorCount > 0) {
                this.showNotification(`Ошибок при загрузке: ${errorCount}`, 'error');
            }

        } catch (error) {
            this.showNotification('Ошибка загрузки файлов: ' + error.message, 'error');
        }

        // Очищаем input
        this.fileInput.value = '';
    }

    async addToQueue(path) {
        try {
            const response = await fetch('/api/queue/add', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ path: path })
            });

            const result = await response.json();
            this.showNotification(result.message, 'success');
            await this.loadQueue();

        } catch (error) {
            this.showNotification('Ошибка добавления в очередь: ' + error.message, 'error');
        }
    }

    async loadQueue() {
        try {
            const response = await fetch('/api/queue');
            const data = await response.json();
            this.queue = data.queue;
            this.displayQueue();
        } catch (error) {
            console.error('Ошибка загрузки очереди:', error);
        }
    }

    displayQueue() {
        if (this.queue.length === 0) {
            this.queueList.innerHTML = `
                <p style="text-align: center; color: #666; padding: 20px 0;">
                    Очередь пуста
                </p>
            `;
            this.processBtn.disabled = true;
            this.clearBtn.disabled = true;
            this.addQueueBtn.disabled = false;
        } else {
            this.queueList.innerHTML = '';
            this.queue.forEach((path, index) => {
                const item = document.createElement('div');
                item.className = 'queue-item';
                item.innerHTML = `
                    <span>${index + 1}. ${path}</span>
                `;
                this.queueList.appendChild(item);
            });
            this.processBtn.disabled = false;
            this.clearBtn.disabled = false;
            this.addQueueBtn.disabled = false;
        }
    }

    async processQueue() {
        try {
            this.processBtn.disabled = true;
            this.processBtn.innerHTML = '<div class="loading"></div> Запуск...';

            const response = await fetch('/api/process', {
                method: 'POST'
            });

            const result = await response.json();
            this.showNotification(result.message, 'success');
            
            await this.loadQueue();
            
        } catch (error) {
            this.showNotification('Ошибка запуска обработки: ' + error.message, 'error');
        } finally {
            this.processBtn.disabled = false;
            this.processBtn.innerHTML = '🚀 Обработать очередь';
        }
    }

    async clearQueue() {
        try {
            const response = await fetch('/api/queue', {
                method: 'DELETE'
            });

            const result = await response.json();
            this.showNotification(result.message, 'success');
            await this.loadQueue();

        } catch (error) {
            this.showNotification('Ошибка очистки очереди: ' + error.message, 'error');
        }
    }

    async loadTasks() {
        try {
            const response = await fetch('/api/tasks');
            const data = await response.json();
            
            // Обновляем только если есть изменения
            const newTasksStr = JSON.stringify(data.tasks);
            if (this.lastTasksStr !== newTasksStr) {
                this.lastTasksStr = newTasksStr;
                this.displayTasks(data.tasks);
            }
            
        } catch (error) {
            console.error('Ошибка загрузки задач:', error);
        }
    }

    displayTasks(tasks) {
        if (tasks.length === 0) {
            this.tasksList.innerHTML = `
                <p style="text-align: center; color: #666; padding: 40px 0;">
                    Задач пока нет
                </p>
            `;
            return;
        }

        this.tasksList.innerHTML = '';
        
        // Сортируем задачи - сначала активные, потом завершенные
        tasks.sort((a, b) => {
            const order = { 'running': 0, 'pending': 1, 'completed': 2, 'error': 3 };
            return order[a.status] - order[b.status];
        });

        tasks.forEach(task => {
            const taskEl = document.createElement('div');
            taskEl.className = `task-item ${task.status}`;
            
            const statusEmoji = {
                'pending': '⏳',
                'running': '⚡',
                'completed': '✅',
                'error': '❌'
            };

            let resultHtml = '';
            if (task.status === 'completed' && task.result) {
                resultHtml = `
                    <div class="result-stats">
                        <div class="stat-item">
                            <div class="stat-value moved">${task.result.moved}</div>
                            <div class="stat-label">Перемещено</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value copied">${task.result.copied}</div>
                            <div class="stat-label">Скопировано</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value clusters">${task.result.clusters_count}</div>
                            <div class="stat-label">Кластеров</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value no-faces">${task.result.no_faces_count}</div>
                            <div class="stat-label">Без лиц</div>
                        </div>
                    </div>
                `;
            }

            let progressHtml = '';
            if (task.status === 'running' || task.status === 'pending') {
                const progress = task.progress || 0;
                progressHtml = `
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${progress}%"></div>
                    </div>
                    <div class="progress-text">${progress}%</div>
                    <div class="progress-details">${task.message || 'Подготовка...'}</div>
                `;
            }

            taskEl.innerHTML = `
                <div class="task-header">
                    <strong>${task.folder_path}</strong>
                    <span class="task-status">${statusEmoji[task.status]} ${task.status}</span>
                </div>
                ${progressHtml}
                <div class="task-message">${task.message}</div>
                ${resultHtml}
            `;

            this.tasksList.appendChild(taskEl);
        });
    }

    startTaskPolling() {
        // Автоматическое обновление задач каждые 2 секунды
        setInterval(() => {
            this.loadTasks();
        }, 2000);
    }

    showNotification(message, type = 'success') {
        // Удаляем предыдущие уведомления
        const existing = document.querySelector('.notification');
        if (existing) {
            existing.remove();
        }

        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Показываем уведомление
        setTimeout(() => {
            notification.classList.add('show');
        }, 100);
        
        // Скрываем через 4 секунды
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 4000);
    }

    async moveItem(src, dest) {
        try {
            const response = await fetch('/api/move', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ src: src, dest: dest })
            });
            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.detail || 'Unknown error');
            }
            this.showNotification(`Перемещено: ${result.src} → ${result.dest}`, 'success');
            // Обновляем текущее содержимое папки
            this.navigateToFolder(this.currentPath);
        } catch (error) {
            this.showNotification('Ошибка перемещения: ' + error.message, 'error');
        }
    }
}

// Инициализация приложения
document.addEventListener('DOMContentLoaded', () => {
    new PhotoClusterApp();
});
