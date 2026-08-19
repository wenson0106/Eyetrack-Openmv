# Eyetrack OpenMV 教學系統

> Head-motion robustness experiments: see [HUMAN_TESTING.md](HUMAN_TESTING.md). Every experiment branch provides a standalone quantitative runner at `http://127.0.0.1:8000/benchmark`; it does not require OpenMV hardware.

此 Repo 包含：

- 電腦端眼動控制前端與 FastAPI 後端
- UniGaze 個人化校準所需的 Python 程式
- OpenMV 小車教學程式
- 原生安裝與 Docker 安裝方式

> UniGaze 預訓練權重採用 `MG-NC-RAI-2.0` 非商業授權。本專案的預建 image 定位為非商業教學與研究使用，發布或改作其他用途前請閱讀 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 兩種安裝方案

| 方案 | 學生需要準備 |
| --- | --- |
| 原生安裝 | Git、Python 3.11、約 5 GB 可用空間 |
| Docker | Docker Desktop 或 Docker Engine、約 6 GB 可用空間 |

## 共通準備

1. OpenMV 小車與執行伺服器的電腦連上同一個區域網路。
2. 開啟 `day4_wifi.py`，將 `WIFI_SSID` 與 `WIFI_PASSWORD` 改成課堂使用的 Wi-Fi 資訊。
3. 開發板端只需要本 Repo 的 `day4_wifi.py` 與必要依賴 `servos.py`。
4. 使用 OpenMV IDE 直接執行 `day4_wifi.py`。若要讓開發板通電後自動執行，請在開發板儲存空間將同一份程式另存為 `main.py`，不需要在 Repo 維護另一個版本。
5. 確認執行後 OpenMV IDE 會印出小車 IP。
6. Chrome、Edge、Firefox 或 Safari 允許使用 webcam。

---

## 方案一：原生安裝

### 支援範圍

| 作業系統 | 狀態 | 備註 |
| --- | --- | --- |
| Windows 10/11 64-bit | 支援 | 建議使用 Python.org 的 Python 3.11 |
| Ubuntu / Debian x86_64 | 支援 | 需要 `python3-venv` |
| macOS 11 以上，Intel | 支援 | 使用 MediaPipe 0.10.14 universal wheel |
| macOS 11 以上，Apple Silicon | 支援 | 建議原生 arm64 Python 3.11 |
| Linux ARM64 | 實驗性 | PyTorch wheel 供應情況依發行版而異 |

統一建議使用 Python 3.11。請使用 64-bit Python，不要使用 Microsoft Store 的舊版 Python。

### Windows PowerShell

```powershell
git clone https://github.com/wenson0106/Eyetrack-Openmv.git
cd Eyetrack-Openmv

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\download_models.py
.\.venv\Scripts\python.exe eye_server.py
```


### Ubuntu / Debian Linux

```bash
sudo apt update
sudo apt install -y git python3 python3-venv libgl1 libglib2.0-0 libgomp1

git clone https://github.com/wenson0106/Eyetrack-Openmv.git
cd Eyetrack-Openmv

python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip setuptools wheel
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python scripts/download_models.py
./.venv/bin/python eye_server.py
```

### macOS

先安裝 [Homebrew](https://brew.sh/)，再執行：

```bash
brew install git python@3.11

git clone https://github.com/wenson0106/Eyetrack-Openmv.git
cd Eyetrack-Openmv

python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip setuptools wheel
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python scripts/download_models.py
./.venv/bin/python eye_server.py
```

### 開啟系統

伺服器啟動後，在同一台電腦開啟：

```text
http://127.0.0.1:8000
```

第一次執行 `scripts/download_models.py` 會從 Hugging Face 下載約 350 MB 的 UniGaze 模型。之後會使用本機快取，不必重複下載。

### 原生安裝常見問題

- `No module named ...`：確認執行的是 `.venv` 裡的 Python，不是系統 Python。
- macOS webcam 無法使用：到「系統設定 > 隱私權與安全性 > 相機」允許瀏覽器。
- Windows 防火牆詢問：允許私人網路；若學校網路被標為公用網路，請依課堂指示處理。

---

## 方案二：Docker

### Docker image 支援範圍

提供的 image 是 `linux/amd64` CPU 版本：

| 主機作業系統 | 狀態 | 執行方式 |
| --- | --- | --- |
| Docker Desktop 官方支援的 Windows 10/11 x86_64 | 支援 | Docker Desktop 使用 Linux containers |
| Linux x86_64 | 支援 | Docker Engine 與 Compose v2 |
| macOS Intel | 支援 | Docker Desktop 原生執行 amd64 image |
| macOS Apple Silicon | 支援但較慢 | Docker Desktop 透過 amd64 模擬執行 |
| Linux ARM64 | 不提供預建 image | 建議使用方案一，或自行驗證 arm64 build |
| Windows Server | 不支援此教學 image | 改用 Linux 主機或支援的 Docker Desktop |

Docker Desktop 官方安裝文件：[Windows](https://docs.docker.com/desktop/setup/install/windows-install/)、[macOS](https://docs.docker.com/desktop/setup/install/mac-install/)。Linux 請參考 [Docker Engine](https://docs.docker.com/engine/install/)。

### 使用離線 image（建議）

1. 取得 `eyetrack-openmv-amd64.tar`。(https://drive.google.com/file/d/1JpYYIzWrw44gpnn3_kKKV_hrprmINH4K/view?usp=sharing)
2. 開啟 PowerShell 或 Terminal，進入 `.tar` 所在資料夾。
3. 此安裝方式不需要下載或 clone Repo。

Windows PowerShell 可先確認檔案是否完整：

```powershell
(Get-FileHash .\eyetrack-openmv-amd64.tar -Algorithm SHA256).Hash
```

Linux / macOS：

```bash
sha256sum eyetrack-openmv-amd64.tar
```

正確的 SHA256 應為：

```text
E9D4673EAB97597EDB6A66375D4FA229A730151D6124229963780531D15054E6
```

載入 image：

```bash
docker load -i eyetrack-openmv-amd64.tar
```

Windows PowerShell 啟動指令：

```powershell
docker run -d --name eyetrack-openmv -p 8000:8000 -v eyetrack-openmv-data:/app/data -v eyetrack-openmv-runs:/app/runs ghcr.io/wenson0106/eyetrack-openmv:latest
```

Linux / macOS：

```bash
docker run -d \
  --name eyetrack-openmv \
  -p 8000:8000 \
  -v eyetrack-openmv-data:/app/data \
  -v eyetrack-openmv-runs:/app/runs \
  ghcr.io/wenson0106/eyetrack-openmv:latest
```

`eyetrack-openmv-data` 與 `eyetrack-openmv-runs` 是 Docker volumes，可讓眼動校準資料在停止或重新啟動容器後繼續保留。

啟動後開啟：

```text
http://127.0.0.1:8000
```

停止服務：

```bash
docker stop eyetrack-openmv
```

之後開啟只需要執行以下指令，不必再次 `docker load`：

```bash
docker start eyetrack-openmv
```

查看執行訊息：

```bash
docker logs -f eyetrack-openmv
```

### 從 Repo 自行建置

沒有離線 image 時，也可以直接從 Repo 建置：

```bash
git clone https://github.com/wenson0106/Eyetrack-Openmv.git
cd Eyetrack-Openmv
docker compose up --build -d
```

建置會下載 Python 套件與約 350 MB 的 UniGaze 模型，可能需要 10 到 30 分鐘。完成後模型會包含在 image 中，正常啟動不需要再次下載。
