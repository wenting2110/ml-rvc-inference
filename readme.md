# RVC Inference
使用 RVC（Retrieval-based Voice Conversion）模型進行語音轉換，支援 CPU 或 GPU 執行環境。可將一段原始語音轉換為指定目標聲音。


## Installation
1. 建立 Conda 環境並安裝 PyTorch (CPU 版本)
```bash
conda create -n rvcinfer python=3.10
conda activate rvcinfer
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
---

2. 安裝 Microsoft Visual C++ Build Tools（C++ 編譯器）

3. Clone the repository 使用 requirements.txt 一次安裝所有套件
```bash
git clone <repository-url> rvc_inference
cd rvc_inference
pip install -r requirements.txt
```
這會自動安裝以下內容：
* torch, torchaudio, torchvision（預設為 CPU 版）
* librosa, soundfile, scikit-learn, faiss-cpu 等常見套件
* 從 GitHub 安裝的 fairseq（需要 C++ 編譯）
* inferrvc 套件（.whl 格式）

---

4. 修改 inferrvc 原始碼（讓 CPU 模式正常運作）
如果你想在沒有 NVIDIA GPU 的電腦上使用 RVC 推論，請依下列方式修改 inferrvc 原始碼，讓其正確使用 CPU：

(1) 開啟下列檔案：
```bash
<你的 Conda 環境路徑>\Lib\site-packages\inferrvc\pipeline.py
```

(2) 找到以下這一行（大約第 31 行）：
```bash
bh, ah = torch.from_numpy(bh).to(_gpu, non_blocking=True), torch.from_numpy(ah).to(_gpu, non_blocking=True)
```

(3) 修改成以下內容：
```py
device = "cuda" if torch.cuda.is_available() else "cpu"
bh, ah = torch.from_numpy(bh).to(device), torch.from_numpy(ah).to(device)
```

---

## Usage
1. 資料夾結構如下：
```perl
RVC Inference/
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

---

2. 設定模型與索引資料夾 
```bash
cd rvc_inference
python infer_CPU.py
```

---

### 推論範例程式碼（節錄自 infer_CPU.py）

```python
converted = model(
    audio,
    f0_up_key=-5,           # 調整音高（建議值參考下方表格）
    output_device="cpu",    # 強制使用 CPU 模式
    output_volume=RVC.MATCH_ORIGINAL,
    index_rate=0.5          # 控制index參與比例（0 = 不用 index，1 = 完全依賴 index）
)
```

---

3. 推論完成後，音訊檔會儲存在專案根目錄，例如：
```perl
rvc_inference/Teacher_infer.wav
```
---

### Optimization
1. 修改 `infer_CPU.py` 中的 `f0_up_key` 值
`f0_up_key` 是 RVC 推論時控制音高（pitch）的參數，用來設定「**將輸入聲音升高或降低幾個音階（semitones）**」。這會直接影響你轉出來的聲音是否像你目標聲音。

---

## `f0_up_key` 的基本說明：
| 數值 | 效果 |
| --- | --- |
| `0` | 不改變原始音高 |
| `> 0` | 提高音高（聲音變尖） |
| `< 0` | 降低音高（聲音變低） |

---

## 📌 如何選擇 `f0_up_key` 值？
| 狀況 | 建議值範圍 |
| --- | --- |
| ♂ 男聲 → ♀ 女聲 | `+5 ~ +12` |
| ♀ 女聲 → ♂ 男聲 | `-5 ~ -12` |
| ♂ 男聲 → 另一種男聲（偏高） | `+2 ~ +5` |
| ♀ 女聲 → 另一種女聲（偏低） | `-2 ~ -5` |
| 原始音高已很接近目標聲音 | `0` |
| 想模仿卡通角色、高音、機器音 | `+12 ~ +20` |
| 想模仿低沉、地獄聲音、變怪物 | `-12 ~ -24` |

---


