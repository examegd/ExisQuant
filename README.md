# exisquant

experimental hybrid vector-scalar quantization framework for local llms, built entirely from scratch.

---

this is my very first project in the field of neural network quantization and machine learning infrastructure. as a 16-year-old developer, i wanted to understand exactly how weights and activations interact under extreme compression. 

i am quite proud of this baseline prototype: while compressing the model down to ~3.3 bits per weight, the resulting hybrid model still retains its linguistic scaffolding and generates correctly spelled, grammatically coherent english text, rather than collapsing into completely random binary noise or spamming chinese characters (which is a classic failure mode for severely quantized qwen models!).

---

## how it works (architecture)

exisquant operates as a two-stage hybrid compression framework:

1. **activation-aware calibration (imatrix)**
   using pytorch forward hooks, the system monitors live activation signals during inference. it identifies critical "outlier" column channels (for example, column #56 in qwen-2.5-0.5b, which processed signals 82.3x stronger than average) that are crucial for the model's logic.

2. **hybrid awq patching (3-bit/4-bit)**
   to protect the model's cognitive capabilities, the top 30% of the most active columns are preserved in lossless 4-bit (q4_0) format, while the remaining 70% of unimportant columns are aggressively compressed to 3-bit (level 4) and mapped to the shortest prefix codes to maximize compression.

3. **vector quantization (vq via k-means)**
   the framework groups 1d weight arrays into 4d spatial coordinates. it runs a fast k-means clustering algorithm to find 256 optimal multidimensional templates (centroids), storing only an 8-bit index of the closest template. this achieves a flat 50% saving (exactly 2.0 bits per weight) on compressed blocks.

4. **native c++ decompressor**
   to avoid heavy python runtime overhead during decompression, the decoder is written in native c++17. it uses a fast lookup table (lut) to restore the 4d vectors from 8-bit indices and patches the target gguf file in milliseconds, using only standard system libraries and win32 file dialogs.

---

## repository structure

* `packer.py` — weight-only vector quantization compiler (4d k-means).
* `unpacker.cpp` — high-speed native c++17 decompressor and gguf patcher.
---

## compilation (c++)

to compile the native decompressor with open-source windows compilers:

```bash
g++ -O3 -std=c++17 unpack_vq.cpp -o unpacker.exe -lcomdlg32 -static
