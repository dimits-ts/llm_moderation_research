import llama_cpp

import tasks.conversation
import tasks.models
import tasks.util


OUTPUT_DIR = "output"

MAX_TOKENS = 512
# see this for a discussion on ctx width for llama models
# https://github.com/ggerganov/llama.cpp/issues/194
CTX_WIDTH_TOKENS = 1024
MODEL_PATH = "/home/dimits/bin/llm_models/llama-2-13b-chat.Q5_K_M.gguf"
RANDOM_SEED = 42
INFERENCE_THREADS = 4


def main():
    print("Loading LLM...")
    llm = llama_cpp.Llama(
        model_path=MODEL_PATH,
        seed=RANDOM_SEED,
        n_ctx=CTX_WIDTH_TOKENS,
        n_threads=INFERENCE_THREADS,
        # will vary from machine to machine
        n_gpu_layers=12,
        # if ran on Linux, model size does not matter since the model uses mmap for lazy loading
        # https://github.com/ggerganov/llama.cpp/discussions/638
        # still have to pay some performance costs of course
        use_mmap=True,
        # using llama-2 leads to well-known model collapse
        # https://www.reddit.com/r/LocalLLaMA/comments/17th1sk/cant_make_the_llamacpppython_respond_normally/
        chat_format="alpaca",
        mlock=True,  # keep memcached model files in RAM if possible
        verbose=False,
    )
    print("Model loaded.")

    model = tasks.models.LlamaModel(llm, max_out_tokens=MAX_TOKENS, seed=RANDOM_SEED)
    data = tasks.conversation.LLMConvData.from_json_file("data/polarized_1.json")
    generator = tasks.conversation.LLMConvGenerator(data=data, user_model=model, moderator_model=model)
    conv = generator.produce_conversation()

    print("Beginning conversation...")
    conv.begin_conversation(verbose=True)
    output_path = tasks.util.generate_datetime_filename(output_dir=OUTPUT_DIR, file_ending=".json")
    conv.to_json_file(output_path)
    print("Conversation saved to ", output_path)


if __name__ == "__main__":
    main()
