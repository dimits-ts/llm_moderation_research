# Mitigating Polarization in Online Discussions Through Adaptive Moderation Techniques

This repository houses the implementation, output and results of experiments exploring the role of LLMs in the moderation of deliberation systems.

The project focuses on the creation, evaluation and exploitation of synthetic dialogues by LLMs.

## Requirements

### Environment

The code is technically cross-platform, but has only been tested on Linux.

### Dependencies

The project so far only requires the `llama_cpp` python library and Python 3.12 to run. In case the `gpu_layers` parameter is used for loading the model, CUDA support must also be enabled.

The platform-specific (Linux x86) conda environment used in this project can be found up-to-date [here](https://github.com/dimits-ts/conda_auto_backup/blob/master/llm.yml).

## Use

There are many ways with which to use the synthetic conversation framework:
1. (Preferred) Run `create_synthetic.py`, see usage below
1. (Preferred) Run `execute_all.sh` in order to batch calls to `create_synthetic.py`, see usage below
1. Run `tutorial.ipynb` with modified parameters
1. Create a new python script leveraging the framework library found in the `tasks` modules

### Running the python script
```
Usage: create_synthetic.py [-h] --input_file INPUT_FILE --output_dir OUTPUT_DIR --model_path MODEL_PATH
                           [--max_tokens MAX_TOKENS] [--ctx_width_tokens CTX_WIDTH_TOKENS]
                           [--random_seed RANDOM_SEED] [--inference_threads INFERENCE_THREADS]
                           [--gpu_layers GPU_LAYERS]
```
### Running the automated batch script
```
Usage: scripts/execute_all.sh --python_script_path <python script path> --input_dir <input_directory> --output_dir <output_directory> --model_path <model_file_path>
```

## Structure

The project is structured as follows:

- `data`: input prompts in JSON format (see `tasks.conversations.LlmConvData`)
- `scripts`: automation scripts for batch processing of experiments
- `tasks`: a task-specific library developed for this project 
- `output`: the output data from the experiments 

- `create_synthetic.py`: a convenient script automatically loading a conversation from serialized data, executing the synthetic dialogue using a local LLM, and serializing the output
- `tutorial.ipynb`: a notebook containing notes on the experiments, implementation and design details, and example code for our framework

## Documentation

Since the project is still nascent and its API constantly shifts, there is no separate, stable documentation. However, we provide up-to-date documentation:

- For implementation details consult the docstrings found in the python source files.

- For higher level details such as performance considerations, choice of libraries/models, and platform-specific caveats, visit the markdown comments of the provided notebook.

- The provided notebook also explains the thought process between the experiments themselves.