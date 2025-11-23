import torch
import json

# 載入原始模型權重
raw = torch.load("C:/Users/user/Desktop/ML_proj/RVC Inference/model/G_45600.pth", map_location="cpu")

# 載入 WebUI 提供的 config.json
with open("C:/Users/user/Desktop/ML_proj/RVC Inference/model/config.json", "r") as f:
    config_dict = json.load(f)

# 為 inferrvc 格式設置 config 成 list
model_config = config_dict["model"]  # 只取出模型參數區塊

# 推論設定
sr = config_dict["data"]["sampling_rate"]  # 取樣率 48000
f0 = True  # 是否使用音高(pitch)資訊來輔助聲音轉換

config_list = [
    1025,                               # spec_channels
    32,                                 # segment_size
    192,                                # inter_channels
    192,                                # hidden_channels
    768,                                # filter_channels
    2,                                  # n_heads
    6,                                  # n_layers
    3,                                  # kernel_size
    0,                                  # p_dropout
    "1",                                # resblock
    [3, 7, 11],                         # resblock_kernel_sizes
    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],  # resblock_dilation_sizes
    [12, 10, 2, 2],                     # upsample_rates
    512,                                # upsample_initial_channel
    [24, 20, 4, 4],                     # upsample_kernel_sizes
    109,                                # spk_embed_dim
    256,                                # gin_channels
    48000                               # sr 
]

# 打包成 inferrvc 格式
cpt = {
    "weight": raw["model"],
    "config": config_list,     # list 形式
    "sr": sr,
    "f0": f0,
    "version": "v2",
    "info": "converted from G_45600.pth"
}

# 存成新的檔案
torch.save(cpt, "C:/Users/user/Desktop/ML_proj/RVC Inference/model/Teacher_infer.pth")
print(cpt.keys())
print('config_list: ', config_list)