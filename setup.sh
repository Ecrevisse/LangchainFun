mkdir models
cd models
wget https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q5_K_M.gguf
cd ..
pip install -U pip
pip install -r requirements.txt