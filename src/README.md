# Mitigating Polarization in Online Discussions Through Adaptive Moderation Techniques

This repository houses the implementation, output and results of experiments exploring the role of LLMs in the moderation of deliberation systems.

The project focuses on the creation, evaluation and exploitation of synthetic dialogues by LLMs.

## Requirements

### Environment

The code is technically cross-platform, but has only been tested on Linux. Note that some notebooks may rely on `bash` commands.

### Dependencies

The project so far only requires the `llama_cpp` python library and Python 3.12 to run. In case the `gpu_layers` parameter is used for loading the model, CUDA support must also be enabled.

The platform-specific (Linux x86) conda environment used in this project can be found up-to-date [here](https://github.com/dimits-ts/conda_auto_backup/blob/master/llm.yml).

## Use

**Tip**: Usage tips can also be found in `notebooks/tutorial`, with more details.

### Synthetic conversation creation

There are many ways with which to use the synthetic conversation framework:
1. (Preferred) Run `create_synthetic.py`, see usage below
1. (Preferred) Run `conversation_execute_all.sh` in order to batch calls to `create_synthetic.py`, see usage below
1. Run `notebooks/tutorial.ipynb` with modified parameters
1. Create a new python script leveraging the framework library found in the `lib` module

#### Running the python script
```
Usage: create_synthetic.py [-h] --input_file INPUT_FILE --output_dir OUTPUT_DIR --model_path MODEL_PATH
                           [--max_tokens MAX_TOKENS] [--ctx_width_tokens CTX_WIDTH_TOKENS]
                           [--random_seed RANDOM_SEED] [--inference_threads INFERENCE_THREADS]
                           [--gpu_layers GPU_LAYERS]
```

#### Running the automated batch script
```
Usage: scripts/execute_all.sh --python_script_path <python script path> --input_dir <input_directory> --output_dir <output_directory> --model_path <model_file_path>
```

### Synthetic annotation for conversations

Similarly, in order to run LLM annotation on the conversations:
1. (Preferred) Run `llm_annotation.py`, see usage below
1. (Preferred) Run `annotation_execute_all.sh` in order to batch calls to `llm_annotation.py`, see usage below
1. Run `notebooks/tutorial.ipynb` with modified parameters
1. Create a new python script leveraging the framework library found in the `lib` module

#### Running the python script
```
Usage: llm_annotation.py [-h] --prompt_input_path PROMPT_INPUT_PATH --conv_path CONV_PATH --output_dir OUTPUT_DIR --model_path MODEL_PATH [--max_tokens MAX_TOKENS]
                         [--ctx_width_tokens CTX_WIDTH_TOKENS] [--random_seed RANDOM_SEED] [--inference_threads INFERENCE_THREADS] [--gpu_layers GPU_LAYERS]
```

#### Running the automated batch script
```
Usage: scripts/annotation_execute_all.sh --python_script_path <python script path> --conv_input_dir <input_directory> --prompt_path <input_path> --output_dir <output_directory> --model_path <model_file_path>
```

## Structure

The project is structured as follows:

- `data`: input prompts in JSON format (see `tasks.conversations.LlmConvData`)
- `models`: directory for local LLM instances
- `scripts`: automation scripts for batch processing of experiments
- `lib`: the LLM conversation library we developed
- `tasks`: a task-specific library largely used in post-experiment preprocessing and analysis
- `output`: the output data from the experiments 
- `notebooks`: notebooks relating to research notes, implementation notes and data analysis 

Notable files:
- `create_synthetic.py`: script automatically loading a conversation's parameters, executing the synthetic dialogue using a local LLM, and serializing the output
- `llm_annotation.py`: script loading a previously concluded conversation from serialized data, executing an annotation job using a local LLM, and serializing the output
- `notebooks/tutorial.ipynb`: a notebook containing notes on the experiments, implementation and design details, as well as example code for our framework

## Documentation

Since the project is still nascent and its API constantly shifts, there is no separate, stable documentation. However, we provide up-to-date documentation:

- For implementation details consult the docstrings found in the python source files.

- For higher level details such as performance considerations, choice of libraries/models, and platform-specific caveats, visit the markdown comments of the provided notebook.

- The provided notebook also explains the thought process between the experiments themselves.