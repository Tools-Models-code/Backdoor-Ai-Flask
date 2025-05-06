# Notebook Fixes for train_backdoor_b2d4g5.ipynb

Here are the three issues that need to be fixed in the notebook:

## 1. Update MODEL_NAME Path

In the "Model Architecture and Loading" section (around line 140), change:

```python
# Define model name
MODEL_NAME = "Backdoor-B2D4G5-Tool-Use"
```

to:

```python
# Define model name
MODEL_NAME = "kaggle/input/b2d4g5/pytorch/backdoor-b2d4g5-tool-use/1/Backdoor-B2D4G5-Tool-Use"
```

Also, in the same cell, update the reference from MODEL_PATH to MODEL_NAME:

```python
missing_files = [file for file in required_files if not os.path.exists(os.path.join(MODEL_NAME, file))]
```

And in the model loading cell (around line 196), change:

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    ...
)
```

to:

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    ...
)
```

## 2. Fix CUDA Issue for Quantization

In the "Configure quantization for efficient training" cell (around line 188), add this code before the BitsAndBytesConfig setup:

```python
# Check CUDA availability for bitsandbytes
import torch
cuda_available = torch.cuda.is_available()
if not cuda_available:
    print("CUDA is not available. Using CPU configuration for bitsandbytes instead.")
    # Use a configuration that works without CUDA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=False,  # Set to False when CUDA is not available
        bnb_4bit_compute_dtype=torch.float32  # Use float32 instead of bfloat16
    )
else:
    # Original CUDA configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
```

Replace the existing bnb_config definition with this code.

## 3. Fix NameError in LoRA Cell

In the "Configure LoRA for efficient fine-tuning" cell (around line 228), add this check before applying LoRA:

```python
# Make sure model is defined before applying LoRA
if "model" not in locals() or model is None:
    print("Model not found. Loading model first.")
    # Load the model (copy from earlier cell)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    model = prepare_model_for_kbit_training(model)
```

Add this code before the peft_config definition.
