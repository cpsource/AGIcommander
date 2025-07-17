# Commander.py Setup Guide

## Overview
Commander.py now supports multiple LLM providers through a modular architecture. You can easily switch between different AI models using the `-m` command line flag.

## Supported Models
- **gemini** (default) - Google's Gemini AI
- **claude** - Anthropic's Claude
- **chatgpt** - OpenAI's GPT models
- **xai** - xAI's Grok models
- **watsonx** - IBM's WatsonX AI

## Installation

### 1. Install Base Dependencies
```bash
pip install python-dotenv
```

### 2. Install Model-Specific Dependencies
Install only the dependencies for the models you plan to use:

#### For Gemini:
```bash
pip install langchain-google-genai langchain
```

#### For Claude:
```bash
pip install anthropic
```

#### For ChatGPT:
```bash
pip install openai
```

#### For xAI:
```bash
pip install openai  # xAI uses OpenAI-compatible API
```

#### For WatsonX:
```bash
pip install ibm-watsonx-ai
```

### 3. Set Up API Keys
Create a `~/.env` file with your API keys:

```bash
# For Gemini
GOOGLE_API_KEY=your_google_api_key_here

# For Claude
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# For ChatGPT
OPENAI_API_KEY=your_openai_api_key_here

# For xAI
XAI_API_KEY=your_xai_api_key_here

# For WatsonX
WATSONX_API_KEY=your_watsonx_api_key_here
WATSONX_PROJECT_ID=your_watsonx_project_id_here
```

## Usage Examples

### Basic Usage (Default Gemini)
```bash
python commander.py
```

### Using Different Models
```bash
# Use Claude
python commander.py -m claude

# Use ChatGPT
python commander.py -m chatgpt

# Use xAI Grok
python commander.py -m xai

# Use WatsonX
python commander.py -m watsonx
```

### Combined with Other Options
```bash
# Use Claude with recursive processing and specific extensions
python commander.py -m claude -r -x "py,js,md"

# Use ChatGPT with auto-confirmation
python commander.py -m chatgpt -y
```

## API Key Sources

### Gemini (Google)
- Get API key from: https://makersuite.google.com/app/apikey
- Set as: `GOOGLE_API_KEY`

### Claude (Anthropic)
- Get API key from: https://console.anthropic.com/
- Set as: `ANTHROPIC_API_KEY`

### ChatGPT (OpenAI)
- Get API key from: https://platform.openai.com/api-keys
- Set as: `OPENAI_API_KEY`

### xAI
- Get API key from: https://x.ai/
- Set as: `XAI_API_KEY`

### WatsonX (IBM)
- Get API key and project ID from: https://dataplatform.cloud.ibm.com/
- Set as: `WATSONX_API_KEY` and `WATSONX_PROJECT_ID`

## Model Configuration

Each model can be configured with additional parameters by modifying the respective files in the `comutl/` directory:

- `comutl/gemini.py` - Configure Gemini model and temperature
- `comutl/claude.py` - Configure Claude model and parameters
- `comutl/chatgpt.py` - Configure GPT model and parameters
- `comutl/xai.py` - Configure Grok model and parameters
- `comutl/watsonx.py` - Configure WatsonX model and parameters

## Architecture

The new modular architecture consists of:

1. **Base LLM Class** (`comutl/base_llm.py`) - Defines the interface all models must implement
2. **Model-Specific Classes** - Each model has its own implementation
3. **Registry System** (`comutl/__init__.py`) - Manages model selection and instantiation
4. **Updated Commander** - Uses the modular system instead of hardcoded Gemini

This design makes it easy to add new LLM providers by simply creating a new class that inherits from `BaseLLM` and adding it to the registry.

## Troubleshooting

### Import Errors
If you get import errors, make sure you've installed the dependencies for the model you're trying to use.

### API Key Errors
Ensure your API keys are correctly set in the `~/.env` file and that you're using the correct environment variable names.

### Model-Specific Issues
Each model may have different requirements or limitations. Check the respective provider's documentation for specific configuration details.

