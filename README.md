# RVC Inference
使用 RVC（Retrieval-based Voice Conversion）模型進行語音轉換，支援 CPU 或 GPU 執行環境。  
可將一段原始語音轉換為指定目標聲音。

---

## Installation
在 Windows 上安裝透過 GitHub 安裝 fairseq，過程中需要建立 symbolic link 或複製某些目錄，預設權限不足會導致安裝失敗。  
因此請使用「系統管理員身分」啟動你的終端機工具。
### 1. 建立 Conda 環境並安裝 PyTorch (CPU 版本)
```bash
conda create -n rvcinfer python=3.10
conda activate rvcinfer
```

### 2. 安裝 PyTorch 
CPU 版本：
```bash
pip install torch==2.9.0+cpu torchaudio==2.9.0+cpu torchvision==0.24.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

### 3. 安裝 Microsoft Visual C++ Build Tools（C++ 編譯器）
`fairseq` 安裝時需要編譯 C++ 模組，**Windows 用戶必須安裝 C++ 編譯器**。

請依下列方式安裝：

1. 前往 [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. 下載安裝程式
3. 勾選「C++ build tools」工作負載
4. 勾選「Windows 10 SDK」或「Windows 11 SDK」
5. 安裝並重新啟動電腦（視情況）

> macOS 或 Linux 系統通常已內建 C++ 編譯器，不需要執行這步驟。

### 4. Clone the repository 使用 requirements.txt 一次安裝所有套件
```bash
git clone https://github.com/wenting2110/ml-rvc-inference rvc_inference
cd rvc_inference
pip install -r requirements.txt
```
這會自動安裝以下內容：
* librosa, soundfile, scikit-learn, faiss-cpu 等常見套件
* 從 GitHub 安裝的 fairseq（需要 C++ 編譯）
* inferrvc 套件（.whl 格式）


### 5. 修改 inferrvc 原始碼（讓 CPU 模式正常運作）
如果你想在沒有 NVIDIA GPU 的電腦上使用 RVC 推論，請依下列方式修改 inferrvc 原始碼，讓其正確使用 CPU：

* 開啟下列檔案：
```bash
<你的 Conda 環境路徑>\Lib\site-packages\inferrvc\pipeline.py
```

* 找到以下這一行（大約第 31 行）：
```bash
bh, ah = torch.from_numpy(bh).to(_gpu, non_blocking=True), torch.from_numpy(ah).to(_gpu, non_blocking=True)
```

* 修改成以下內容，根據是否有 GPU 自動選擇裝置：
```py
device = "cuda" if torch.cuda.is_available() else "cpu"
bh, ah = torch.from_numpy(bh).to(device), torch.from_numpy(ah).to(device)
```

---

## Usage
### 1. 模型下載與放置方式
請至本專案的 [Releases 頁面](https://github.com/wenting2110/ml-rvc-inference/releases) 下載下列兩個檔案：

- `Teacher_infer.pth`：模型權重
- `Teacher_infer.index`：聲音索引

下載後請將檔案放置於以下資料夾：
```perl
rvc_inference/
├── infer_CPU.py             # 主推論腳本
├── package.py               # 將 WebUI 模型轉換為 infer 專用格式
├── index.py                 # 訓練 index (.index)
├── model/                   # 儲存 .pth 模型檔
│   └── Teacher_infer.pth
├── index/                   # 儲存 .index 索引檔
│   └── Teacher_infer.index
├── input/                   # 放待轉換的輸入音檔（.wav）
│   └── sample.wav
├── Teacher_infer.wav        # 推論後的音檔（.wav）
├── requirements.txt         # Python 套件需求
└── README.md
```


### 2. 執行推論程式碼 `infer_CPU.py`
```bash
cd rvc_inference
python infer_CPU.py
```


### 3. 推論完成後，音訊檔會儲存在專案根目錄，例如：
```perl
rvc_inference/Teacher_infer.wav
```
---

### 4. Optimization：修改 `infer_CPU.py` 中的 `f0_up_key` value
`f0_up_key` 是 RVC 推論時控制音高（pitch）的參數，用來設定「**將輸入聲音升高或降低幾個音階**」，影響轉換出來的聲音是否像目標聲音。


* `f0_up_key` 的基本說明：
| 數值 | 效果 |
| --- | --- |
| `0` | 不改變原始音高 |
| `> 0` | 提高音高（聲音變尖） |
| `< 0` | 降低音高（聲音變低） |


* 如何選擇 `f0_up_key` 值？
| 狀況 | 建議值範圍 |
| --- | --- |
| 男聲 → 女聲 | `+5 ~ +12` |
| 女聲 → 男聲 | `-5 ~ -12` |
| 男聲 → 另一種男聲（偏高） | `+2 ~ +5` |
| 女聲 → 另一種女聲（偏低） | `-2 ~ -5` |
| 原始音高已很接近目標聲音 | `0` |


---


