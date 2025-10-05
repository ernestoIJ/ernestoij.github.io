---
title: "MiniGPT From Scratch"
excerpt: "Implemented a custom Byte Pair Encoding (BPE) tokenizer and built a Transformer-based Large Language Model (LLM) from scratch in PyTorch, replicating the core architecture of GPT-style models. <br/><img src='/images/miniGPT.png'>"
collection: portfolio
---

### Overview
Developed a **GPT-style Large Language Model (LLM)** entirely from scratch to understand and reproduce the architecture and training process of modern transformer-based language models. The project included implementing both the tokenizer and the model using only core PyTorch and Python libraries. The code to this project can be found [here.](https://github.com/ernestoIJ/LLM-From-Scratch)

### Technical Highlights
- Implemented a **custom Byte Pair Encoding (BPE) tokenizer** supporting vocabulary sizes up to **4K**, reducing token count by **~35%** compared to raw character encoding.  
- Built and trained a **Transformer-based LLM** from scratch in PyTorch, replicating the **core architecture of GPT models**.  
- Designed a **custom training and sampling pipeline**, achieving **stable convergence** (validation loss reduced from **15.5 to 9.4**) and coherent text generation on **CPU/MPS hardware**.

![MiniGPT Model Demo](/images/miniGPT.png)
