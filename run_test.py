import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import ChatTTS
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("test")

logger.info("Loading ChatTTS model from HuggingFace...")
chat = ChatTTS.Chat()
ok = chat.load(source="huggingface", compile=False)

if not ok:
    logger.error("Model load FAILED. Check internet connection.")
    sys.exit(1)

logger.info("Model loaded OK. Sampling speaker...")
spk = chat.sample_random_speaker()

logger.info("Running inference...")
wavs = chat.infer(
    ["Hello, this is ChatTTS. The model is working correctly."],
    params_infer_code=ChatTTS.Chat.InferCodeParams(spk_emb=spk),
)

logger.info("Inference done. Saving MP3...")
from tools.audio.pcm import pcm_arr_to_mp3_view

out_path = os.path.join(os.path.dirname(__file__), "output_audio_0.mp3")
data = pcm_arr_to_mp3_view(wavs[0])
with open(out_path, "wb") as f:
    f.write(data)

logger.info("SUCCESS — saved to %s", out_path)
