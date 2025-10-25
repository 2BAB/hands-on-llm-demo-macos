# Hands-on LLM Demo for macOS

Convert [samples](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models) of 'Hands on Large Language Models (LLM)' **to run on macOS**.

- You can find all scripts such as `ch01.py` in the root directory.
- For `/note` directory, it contains some Q&A that I was confused about which was answering by Gemini 2.5 Flash/Pro.
- For `/result` directory, it contains the result of running the script per chapter.

## Requirements

- macOS (Apple Silicon M series recommended)
- Python 3.12
- [uv](https://github.com/astral-sh/uv) package manager
- At least 20GB of free disk space (for model storage)

## Setup Instructions

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone or download this project

```bash
cd /path/to/project
```

### 3. Install dependencies

The project uses uv to manage dependencies, which will automatically install packages defined in `pyproject.toml`:

```bash
uv sync
```

### 4. Download the model locally

First-time setup requires downloading the Phi-3 model (~7.6GB):

```bash
uv run python download_model.py
```

This will download the model to the `./model/Phi-3-mini-4k-instruct/` directory.

**Note**: Download may take several minutes depending on your network speed.

## Usage

### Run the example script

After downloading the model, run the example script such as `ch01.py`:

```bash
uv run python ch01.py
```

This script will:
1. Load the Phi-3 model and tokenizer from local storage
2. Create a text generation pipeline
3. Generate a funny joke about chickens

### Customize the prompt

Edit the `messages` variable in `ch01.py` to change the input:

```python
messages = [
    {"role": "user", "content": "Your custom prompt here"}
]
```

## Project Structure

```
README.md              # Project documentation
pyproject.toml         # Project configuration and dependencies
download_model.py      # Model download script
ch01.py               # Example ch01: Basic text generation
note/                  # Q&A notes
result/                # Result of running the script per chapter
model/                # Model storage directory (created after first run)
  Phi-3-mini-4k-instruct/
  ...
.gitignore            # Git ignore configuration
```

## Configuration

### Apple Silicon (M1/M2/M3)

The script is configured by default to use MPS (Metal Performance Shaders):

```python
device_map="mps"
```

### Intel Mac

If you're using an Intel Mac, change the device configuration in `ch01.py` to:

```python
device_map="cpu"
```

## Troubleshooting

### 1. Model download fails

If the download fails, try:
- Check your network connection
- Use a proxy or VPN
- Re-run `download_model.py`

### 2. Out of memory

Phi-3-mini requires approximately 8GB RAM. If you encounter memory issues:
- Close other applications
- Consider using a smaller model

### 3. Slow performance

- Apple Silicon Mac: Ensure you're using `device_map="mps"`
- Intel Mac: CPU inference will be slower, which is expected

## Resources

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Phi-3 Model Page](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [uv Documentation](https://docs.astral.sh/uv/)

## License

This project is for learning and demonstration purposes only.
