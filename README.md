# vLLM DGX Spark NVFP4 Orchestration

A dockerized orchestration setup for running **Qwen3-Coder-Next-NVFP4** on NVIDIA DGX Spark systems using **vLLM** with NVFP4 (4-bit NVIDIA Floating Point) quantization.

> **Why a custom vLLM image?** NVFP4 quantization requires custom kernels that are not included in the official vLLM image. The `avarok/dgx-vllm-nvfp4-kernel` image provides:
>
> - NVFP4 GEMM backend (Marlin) for 4-bit inference
> - FlashInfer attention kernels optimized for DGX Spark
> - FP8 KV-cache support for reduced memory usage
>
> The official vLLM image does not support NVFP4 on GB10 (Grace Blackwell) GPUs out of the box—this custom build is required to unlock NVFP4 performance on DGX Spark systems.

## Overview

This project provides a production-ready configuration for deploying high-performance LLM inference on DGX Spark hardware. It leverages:

- **vLLM** with FlashInfer attention kernels
- **NVFP4 quantization** for memory-efficient inference
- **FlashAttention** with FP8 KV-cache
- **Chunked prefill** for optimal throughput

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA Docker runtime configured (`nvidia-container-toolkit`)
- Access to Hugging Face model `RedHatAI/Qwen3-Coder-Next-NVFP4`
- Hugging Face token with read access
- DGX Spark system with compatible GPU(s)

## Configuration

This project uses `direnv` to manage environment variables for Docker Compose.

### Using direnv (Recommended)

1. Install `direnv` if not already installed:

2. Copy `.envrc.example` to `.envrc` and customize:

   ```bash
   cp .envrc.example .envrc
   # Edit .envrc with your values
   ```

3. Allow direnv to load:

   ```bash
   direnv allow
   ```

The environment variables will now be automatically loaded when you enter the directory and injected into `docker-compose` runs.

### Manual Environment Setup

If not using `direnv`, export the variables manually or use `.env` file:

```bash
export IMAGE="avarok/dgx-vllm-nvfp4-kernel"
export VLLM_ADDR="0.0.0.0"
export VLLM_PORT=8000
export HF_TOKEN="your-huggingface-token"
export HF_MODEL_NAME="RedHatAI/Qwen3-Coder-Next-NVFP4"
export SERVED_MODEL_NAME="qwen3-coder-next"
# ... other variables from .envrc.example
```

Key environment variables:

| Variable                 | Description               | Default                           |
| ------------------------ | ------------------------- | --------------------------------- |
| `IMAGE`                  | Docker image name         | `avarok/dgx-vllm-nvfp4-kernel`    |
| `VLLM_ADDR`              | Bind address              | `0.0.0.0`                         |
| `VLLM_PORT`              | API port                  | `8000`                            |
| `HF_TOKEN`               | Hugging Face access token | [ REDACTED ]                      |
| `HF_MODEL_NAME`          | Model identifier          | `RedHatAI/Qwen3-Coder-Next-NVFP4` |
| `SERVED_MODEL_NAME`      | Alias for API             | `qwen3-coder-next`                |
| `GPU_MEM_UTIL`           | GPU memory utilization    | `0.9`                             |
| `CONTEXT_LEN`            | Max context length        | `131072` (2^17)                   |
| `MAX_NUM_BATCHED_TOKENS` | Max batched tokens        | `16384` (2^14)                    |
| `MAX_NUM_SEQS`           | Max concurrent sequences  | `4`                               |

## Quick Start

```bash
# Start the vLLM server in daemon mode
docker-compose up -d

# Check if the model loads and the server starts
docker-compose logs -f vllm

# Check server status
curl http://localhost:8000/v1/models

# Generate text (example)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder-next",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7
  }'
```

## Stopping the Service

```bash
docker-compose down
```

To also remove volumes (cache):

```bash
docker-compose down -v
```

## Architecture

```asciiart
┌─────────────────────────────────────────────────────────┐
│                    DGX Spark Host                       │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │              vLLM Container                       │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  Qwen3-Coder-Next-NVFP4 Model (NVFP4)       │  │  │
│  │  │  - FlashInfer MOE Backend                   │  │  │
│  │  │  - FP8 KV Cache                             │  │  │
│  │  │  - Marlin NVFP4 GEMM Backend                │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  REST API (Port 8000)                       │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## API Documentation

Once the server is running, visit:

- **Swagger UI**: `http://localhost:8000/docs`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

## Performance Tuning

Adjust these parameters based on your GPU memory and workload:

1. **`GPU_MEM_UTIL`**: Increase for higher throughput, decrease if OOM
2. **`MAX_NUM_SEQS`**: More concurrent sequences for higher throughput
3. **`MAX_NUM_BATCHED_TOKENS`**: Higher values improveprefill efficiency
4. **`CONTEXT_LEN`**: Adjust based on your longest required context

## Troubleshooting

### GPU Memory Issues

- Reduce `GPU_MEM_UTIL`
- Reduce `MAX_NUM_SEQS`
- Reduce `CONTEXT_LEN`

### Model Not Loading

- Verify HF token has access to the model
- Check model name matches Hugging Face identifier
- Ensure sufficient GPU memory for the model size

### Connection Refused

- Ensure no other service is using port 8000
- Check Docker container logs: `docker-compose logs -f`

---

Copyright (c) 2026 Lukasz P. Orlowski <lukasz@orlowski.io>
All rights granted under [MIT License](LICENSE)
