import json
import re

# Path to notebook
notebook_path = "notebooks/train_backdoor_b2d4g5.ipynb"
output_path = "notebooks/train_backdoor_b2d4g5_fixed.ipynb"

# Read the notebook
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# 0. Rename the first MODEL_NAME variable to MODELS_DIR to avoid conflicts
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            # Find the first MODEL_NAME definition (the models directory)
            if 'MODEL_NAME = "/kaggle/input/MODELS"' in line:
                # Replace with MODELS_DIR
                cell['source'][i] = line.replace('MODEL_NAME = "/kaggle/input/MODELS"', 'MODELS_DIR = "/kaggle/input/MODELS"')
                
                # Also update any references to this variable in the same cell
                j = i + 1
                while j < len(cell['source']):
                    if 'MODEL_NAME' in cell['source'][j]:
                        cell['source'][j] = cell['source'][j].replace('MODEL_NAME', 'MODELS_DIR')
                    j += 1

# 1. Update MODEL_NAME to new path
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            # Update MODEL_NAME definition
            if '# Define model name' in line and i + 1 < len(cell['source']) and 'MODEL_NAME = "Backdoor-B2D4G5-Tool-Use"' in cell['source'][i + 1]:
                cell['source'][i + 1] = 'MODEL_NAME = "kaggle/input/b2d4g5/pytorch/backdoor-b2d4g5-tool-use/1/Backdoor-B2D4G5-Tool-Use"\n'
                
            # Update ALL MODEL_PATH references to MODEL_NAME
            if 'MODEL_PATH' in line:
                cell['source'][i] = line.replace('MODEL_PATH', 'MODEL_NAME')
            
            # Change MODEL_PATH to MODEL_NAME in from_pretrained call
            if 'model = AutoModelForCausalLM.from_pretrained(' in line and i + 1 < len(cell['source']) and 'MODEL_PATH,' in cell['source'][i + 1]:
                cell['source'][i + 1] = '    MODEL_NAME,\n'

# 2. Fix CUDA issue in quantization cell
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            if '# Configure quantization for efficient training' in line:
                # Replace the bnb_config setup with CUDA aware code
                cuda_check_code = [
                    '# Check CUDA availability for bitsandbytes\n',
                    'import torch\n',
                    'cuda_available = torch.cuda.is_available()\n',
                    'if not cuda_available:\n',
                    '    print("CUDA is not available. Using CPU configuration for bitsandbytes instead.")\n',
                    '    # Use a configuration that works without CUDA\n',
                    '    bnb_config = BitsAndBytesConfig(\n',
                    '        load_in_4bit=False,  # Set to False when CUDA is not available\n',
                    '        bnb_4bit_compute_dtype=torch.float32  # Use float32 instead of bfloat16\n',
                    '    )\n',
                    'else:\n',
                    '    # Original CUDA configuration\n',
                    '    bnb_config = BitsAndBytesConfig(\n',
                    '        load_in_4bit=True,\n',
                    '        bnb_4bit_quant_type="nf4",\n',
                    '        bnb_4bit_compute_dtype=torch.bfloat16,\n',
                    '        bnb_4bit_use_double_quant=True\n',
                    '    )\n'
                ]
                
                # Find the end of the bnb_config code block
                end_idx = i + 1
                while end_idx < len(cell['source']) and '))' not in cell['source'][end_idx]:
                    end_idx += 1
                if end_idx < len(cell['source']):
                    end_idx += 1  # Include the closing line
                
                # Replace the old bnb_config with the new cuda-aware config
                old_config = cell['source'][i+1:end_idx]
                cell['source'] = cell['source'][:i+1] + cuda_check_code + cell['source'][end_idx:]
                break

# 3. Fix the NameError in LoRA cell
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            if '# Configure LoRA for efficient fine-tuning' in line:
                # Add model check before LoRA config
                model_check_code = [
                    '# Make sure model is defined before applying LoRA\n',
                    'if "model" not in locals() or model is None:\n',
                    '    print("Model not found. Loading model first.")\n',
                    '    # Load the model (copy from earlier cell)\n',
                    '    model = AutoModelForCausalLM.from_pretrained(\n',
                    '        MODEL_NAME,\n',
                    '        quantization_config=bnb_config,\n',
                    '        device_map="auto",\n',
                    '        trust_remote_code=True,\n',
                    '        torch_dtype=torch.bfloat16\n',
                    '    )\n',
                    '    model = prepare_model_for_kbit_training(model)\n',
                    '\n'
                ]
                
                # Insert the model check code
                cell['source'] = cell['source'][:i+1] + model_check_code + cell['source'][i+1:]
                break

# Write the updated notebook
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Updated notebook written to {output_path}")
