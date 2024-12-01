import os
os.environ['TORCH_DISTRIBUTED_BACKEND'] = 'gloo'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'

from llama_models.llama3.api.datatypes import UserMessage, SystemMessage, CompletionMessage, StopReason
from llama_models.llama3.reference_impl.generation import Llama

def ask_question(
        ckpt_dir: str, 
        question: str, 
        temperature: float = 0.9, 
        top_p: float = 0.9, 
        max_seq_len: int = 512, 
        max_gen_len: int = 50):
    # Initialize the model
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=4,
        model_parallel_size=1,
        #tokenizer_path=ckpt_dir
        tokenizer_path=ckpt_dir+"/tokenizer.model"
    )

    # Prepare the dialog with the user's question
    dialog = [
        #SystemMessage(content="You are a 1 star michelin chef"),
        UserMessage(content=question)
        #TextCompletionContent(content=question)
    ]

    ###WARM UP!!! <-- very important, dont' remove
    # Generate a response
    result = generator.chat_completion_raw(
        messages=dialog,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p
    )

    # Generate a response
    result = generator.chat_completion_raw(
        messages=dialog,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p
    )

    ####
    # Generate a response
    result = generator.chat_completion(
        messages=dialog,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        echo=False
    )

    # Output the generated response
    out_message = result.generation
    print(f"> {out_message.role.capitalize()}: {out_message.content}")
    
# Example usage
if __name__ == "__main__":
    ckpt_dir = "/Users/username/.llama/checkpoints/Llama3.2-1B"  # Path to your model directory
    question = "What is the recipe for pancakes?"
    ask_question(ckpt_dir, question)
