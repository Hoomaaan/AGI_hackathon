from optimum.neuron.models.training import (
    consolidate_model_parallel_checkpoints_to_unified_checkpoint,
)
from transformers import AutoModel, AutoTokenizer
from argparse import ArgumentParser
from shutil import copyfile
import os
import peft

parser = ArgumentParser()
parser.add_argument(
    "-i",
    "--input_dir",
    help="source checkpoint directory containing sharded adapter checkpoint files",
    required=True,
)
parser.add_argument(
    "-o",
    "--output_dir",
    help="destination directory for final merged model (adapters merged into base model)",
    required=True,
)
args = parser.parse_args()

consolidated_ckpt_dir = os.path.join(args.input_dir, "consolidated")

# Consolidate the adapter shards into a PEFT-compatible checkpoint
print("Consolidating LoRA adapter shards")
consolidate_model_parallel_checkpoints_to_unified_checkpoint(
    args.input_dir, consolidated_ckpt_dir
)
copyfile(
    os.path.join(args.input_dir, "adapter_default/adapter_config.json"),
    os.path.join(consolidated_ckpt_dir, "adapter_config.json"),
)

# Load AutoPeftModel using the consolidated PEFT checkpoint
peft_model = peft.AutoPeftModelForCausalLM.from_pretrained(consolidated_ckpt_dir)

# Merge adapter weights into base model, save new pretrained model
print("Merging LoRA adapter shards into base model")
merged_model = peft_model.merge_and_unload()
print(f"Saving merged model to {args.output_dir}")
merged_model.save_pretrained(args.output_dir)

print(f"Saving tokenizer to {args.output_dir}")
tokenizer = AutoTokenizer.from_pretrained(args.input_dir)
tokenizer.save_pretrained(args.output_dir)

# Load the pretrained model and print config
print("Merged model config:")
model = AutoModel.from_pretrained(args.output_dir)
print(model)
