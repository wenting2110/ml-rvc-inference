import os
from inferrvc import RVC, load_torchaudio
import soundfile as sf
import torch
import fairseq.data.dictionary
#import torch.serialization

# 允許 fairseq Dictionary 被安全載入
torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])


# 強制關閉 CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
torch.cuda.is_available = lambda: False
torch._C._cuda_isDriverSufficient = lambda: False
print("Using CPU mode only. CUDA disabled.")

# 設定模型與索引的資料夾
base_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['RVC_MODELDIR'] = os.path.join(base_dir, 'model')
os.environ['RVC_INDEXDIR'] = os.path.join(base_dir, 'index')
#os.environ['RVC_MODELDIR'] = r"C:\Users\user\Desktop\ML_proj\RVC Inference\model"
#os.environ['RVC_INDEXDIR'] = r"C:\Users\user\Desktop\ML_proj\RVC Inference\index"

# the audio output frequency, default is 44100.
os.environ['RVC_OUTPUTFREQ'] = '44100'
# If the output audio tensor should block until fully loaded, this can be ignored. 
# But if you want to run in a larger torch pipeline, setting to False will improve performance a little.
os.environ['RVC_RETURNBLOCKING'] = 'True'

# 載入模型
model = RVC('Teacher_infer.pth', index='Teacher_infer.index')

print(model.name)
print('Paths', model.model_path, model.index_path)


# 讀取要轉換的音檔
input_audio_path = os.path.join(base_dir, "input", "docs_audio_obama.wav")
audio, sr = load_torchaudio(input_audio_path)

# 語音轉換
converted = model(
    audio,
    f0_up_key=-5,           # 降低音高
    output_device="cpu",    # 使用 CPU
    output_volume=RVC.MATCH_ORIGINAL,
    index_rate=0.5
)

# 輸出轉換後的音訊
sf.write("Teacher_infer.wav", converted, 44100)
print("Done!")
