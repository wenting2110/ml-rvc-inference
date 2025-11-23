import faiss
import numpy as np
import os

# 資料夾路徑：包含所有 .npy 特徵檔案
npy_dir = r"D:\RVC_inference\Teacher_2\3_feature768"

# 載入所有 .npy，合併成一個大的矩陣
npy_files = [os.path.join(npy_dir, f) for f in os.listdir(npy_dir) if f.endswith('.npy')]
npy_files.sort()  # 確保順序一致

features = [np.load(f) for f in npy_files]
all_features = np.concatenate(features, axis=0).astype('float32')  # FAISS 需要 float32

print(f"Loaded {len(features)} files with shape {all_features.shape}")

# 建立 FAISS index
index = faiss.IndexFlatL2(all_features.shape[1])  # L2距離（也就是歐式距離）
index.add(all_features)

# 儲存 index
faiss.write_index(index, r"D:\RVC_inference\model\Teacher_infer.index")
print("Index saved to Teacher_infer.index")
