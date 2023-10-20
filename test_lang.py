from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# url_model = "https://huggingface.co/TheBloke/Thespis-13B-v0.3-GGUF/blob/main/thespis-13b-v0.3.Q5_K_M.gguf"

model_path = "./models/thespis-13b-v0.3.Q5_K_M.gguf"

n_gpu_layers = 1
n_batch = 512
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=callback_manager,
    verbose=True,
)

llm("")
