import os
import yaml
from itertools import product

# Paths
base_config_path = "algo/stasc.yaml"
output_dir = "algo/stasc_versions"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Mapping from char to boolean
char_to_bool = {'t': True, 'f': False}

# Boolean flag names in config
flag_keys = [
    "initial_answer_with_new_model",
    "only_better_correction",
    "train_from_initial_model"
]

# Load base config
with open(base_config_path, "r") as f:
    base_config = yaml.safe_load(f)

# Iterate over all 8 binary combinations
for bits in product("tf", repeat=3):
    config = base_config.copy()
    suffix = ''.join(bits)

    # Apply booleans
    for key, bit in zip(flag_keys, bits):
        config[key] = char_to_bool[bit]

    # Update run name specification
    config["run_name_specification"] = ""

    # Write to new file
    output_path = os.path.join(output_dir, f"stasc_{suffix}.yaml")
    with open(output_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"Generated: {output_path}")
