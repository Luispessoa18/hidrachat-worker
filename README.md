# ⚡ HidraChat Worker

Contribua com seu hardware para a rede de IA distribuída **HidraChat** e ganhe tokens HC.

🌐 **[hidrachat.cloud](https://hidrachat.cloud)**

---

## Como funciona

A HidraChat é uma rede distribuída de IA. Usuários enviam perguntas, a rede roteia para o melhor worker disponível, e o worker processa e retorna a resposta usando um modelo local (llama.cpp + GGUF).

**Quanto mais poder computacional você contribui → mais HC você ganha → maior seu tier de acesso.**

| Tier | HC Ganho | Prioridade | Max tokens |
|------|----------|-----------|------------|
| T1   | 0–99     | Baixa     | 512        |
| T2   | 100–499  | Normal    | 1024       |
| T3   | 500–1999 | Alta      | 2048       |
| T4   | 2000–9999| Muito alta| 3072       |
| T5   | 10000+   | Máxima    | 4096       |

---

## Pré-requisitos

- Python 3.10+
- Git
- CMake 3.14+ (para compilar o llama.cpp)
- 4 GB+ de RAM (8 GB+ recomendado)

---

## 1. Clonar este repositório

```bash
git clone https://github.com/SEU_USUARIO/hidrachat-worker.git
cd hidrachat-worker
```

---

## 2. Compilar o llama.cpp

Clone o llama.cpp dentro da pasta do worker:

```bash
git clone https://github.com/ggml-org/llama.cpp llamacpp
cd llamacpp
```

### Linux / macOS — CPU

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
cd ..
```

### Linux — CUDA (NVIDIA)

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
cd ..
```

### Linux — Vulkan (AMD / Intel)

```bash
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
cd ..
```

### macOS — Apple Silicon (Metal)

```bash
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
cd ..
```

### Windows — CPU (PowerShell)

```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $env:NUMBER_OF_PROCESSORS
cd ..
```

### Windows — CUDA (PowerShell)

```powershell
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $env:NUMBER_OF_PROCESSORS
cd ..
```

### Windows — Vulkan (PowerShell)

```powershell
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $env:NUMBER_OF_PROCESSORS
cd ..
```

> **Dica:** Se não quiser compilar, baixe os binários pré-compilados em [github.com/ggml-org/llama.cpp/releases](https://github.com/ggml-org/llama.cpp/releases) e coloque em `llamacpp/`.

---

## 3. Baixar um modelo GGUF

Instale o huggingface-cli:

```bash
pip install huggingface-hub
```

Baixe um modelo (exemplos):

```bash
# Llama 3.2 3B — leve, bom para CPU
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
  --include "Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
  --local-dir models/

# Mistral 7B — bom equilíbrio CPU/GPU
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
  --include "mistral-7b-instruct-v0.2.Q4_K_M.gguf" \
  --local-dir models/

# Llama 3.1 8B — melhor qualidade, requer GPU
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  --include "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" \
  --local-dir models/
```

---

## 4. Rodar o worker

### Linux / macOS

```bash
chmod +x start.sh
./start.sh
```

### Windows (PowerShell)

```powershell
.\start.ps1
```

### Direto com Python

```bash
python worker.py
```

O worker vai pedir seu **email do HidraChat** e escolher o modelo automaticamente.

---

## Variáveis de ambiente

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `HIDRACHAT_WORKER_EMAIL` | *(pergunta)* | Email da sua conta HidraChat |
| `HIDRACHAT_WORKER_NAME` | `my-worker` | Nome do seu worker na rede |
| `HIDRACHAT_MODEL_PATH` | *(auto)* | Caminho direto para o .gguf |
| `HIDRACHAT_N_GPU_LAYERS` | `0` | Camadas na GPU (0 = só CPU) |
| `HIDRACHAT_THREADS` | `4` | Threads de CPU |
| `HIDRACHAT_RAM_GB` | `4` | RAM disponível declarada |
| `HIDRACHAT_CTX_SIZE` | `4096` | Tamanho do contexto |
| `HIDRACHAT_REGION` | `local` | Região geográfica |

### Exemplo com GPU CUDA

```bash
export HIDRACHAT_WORKER_EMAIL="seu@email.com"
export HIDRACHAT_WORKER_NAME="rtx-4090"
export HIDRACHAT_N_GPU_LAYERS=35
export HIDRACHAT_RAM_GB=24
python worker.py
```

### Windows com GPU

```powershell
$env:HIDRACHAT_WORKER_EMAIL  = "seu@email.com"
$env:HIDRACHAT_WORKER_NAME   = "rtx-4090"
$env:HIDRACHAT_N_GPU_LAYERS  = "35"
$env:HIDRACHAT_RAM_GB        = "24"
python worker.py
```

---

## Sistema de recompensas

- **HC (HidraCoins)** — tokens ganhos por processar jobs na rede
- Workers mais rápidos (TK/s maior) ganham mais HC por job
- O HC acumulado define seu **tier** de acesso ao chat
- Não há limite de ganhos — quanto mais você rodar, mais você ganha

---

## Suporte

🌐 [hidrachat.cloud](https://hidrachat.cloud) · Abra uma issue aqui no GitHub
