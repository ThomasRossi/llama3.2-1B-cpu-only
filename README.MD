# LLAMA3.2-1B CPU ONLY SETUP
## _for Mac Silicon and CPU only machines_

Not every task needs GPU compute, not every dev setup has CUDA devices.

Use at your own risk, nothing is guaranteed to work, see MIT LICENSE.

## Install

- locate the current `model.py` and `generator.py` files in the reference_impl folder of `llama_models/llama3/reference_impl`
- if you virtualenvs it should be in a folder similar to this one:
- `/Users/<username>/.virtualenvs/<yourvirtualenvname>/lib/python3.11/site-packages/llama_models/llama3/reference_impl`
- replace the generator.py and model.py with the ones provided in this repo

If you are curious the version provided here:

model.py
- line 153 -> use cpu if not cuda
- line 154-172 -> use the cpu device

generator.py
-line 103 -> "gloo" engine
-don't use device.cuda anywhere
-line 153 -> use FloatTensor not cuda.HalfTensor
-line 224-248 -> force device="cpu" not "cuda"

## Running

you can try out the provided `example_generator.py`, an interesting note:
- you need to ask a couple of questions for warm up!
- if you keep the process live, jsut expect the first couple of interactions to be gibberish and then it starts working as expected

## License

MIT

**Free Software, Hell Yeah!**

