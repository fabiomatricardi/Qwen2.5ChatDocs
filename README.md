# Qwen2.5ChatDocs
A gradio Chatbot with llamaCPP server and Qwen2.5 3B for your documents


<img src='https://i.ibb.co/ymVYsCFL/qwen25-side.png' width=800>


Download the model *qwen2.5-3b-instruct-q6_k.gguf* in GGUF format [from here](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/tree/main)

Download the [latest binaries of llama.cpp](https://github.com/ggml-org/llama.cpp/releases) in the same directory

### Dependencies
```
pip install gradio pypdf tiktoken openai
```

## Run the interface and the model
from the terminal run
```
python gr_Qwen2.5_doc_chat.py
```


### prompt guidelines
Read more in the [official blog post](https://qwenlm.github.io/blog/qwen2.5/)


### The running app


<img src='https://github.com/fabiomatricardi/Qwen2.5ChatDocs/raw/main/QWEN25_docsGradio_sm.gif' width=1000>
