
- use existing code for a transformer
- only top layer of encoder is cross-attended by decoder
- decoder has two cross (both have same cross data source see above) then two self-only layers
- encoder uses a 3 stack of [5 mamba layers (remember mamba layer is two mamba and no FF), then 1 flash attn 2]
- all use softmax-1
- decoder uses only flash-attn 2 with softmax-1
- we do NOT pass the whole of encoder output to the decoder:
  - instead we split the token indecies excluding the last one into N random sets of same length (if doesnt fit we drop some indecies, so len can be maximal)
  - we create N masks using these random sets but the last token is additionally always included
  - the decoder now has N as a subbatch dimension
- otherwise we always use state-of the art ways of configuring the architecture
- we use some standard word/subword tokenization
- normal positional encoding 
- training loop is:
  - taking some texts for the batch
  - passing batch through the encoder
  - decoder task is to reconstruct original text based on the subset masked last hidden states of the encoder
  - using normal next token prediction loss

choose other parameters so we can run on a RTX 5090 comfortably, batch size should depend on input length, and we do train with variable input length, so that we can also train very long inputs, for which we require small batchs size.


todo:
Newer models reduce KV heads to save memory.

Examples:

LLaMA 3

Mistral 7B

Typical pattern:

Q heads: 32
KV heads: 8

python train_mamba_flash_hybrid.py --estimate-memory --minibatch 128:64 256:32 512:16 1024:4 4096:1 --hidden-size 2048 --num-heads 32
python train_mamba_flash_hybrid.py --estimate-memory --minibatch 128:64 256:32 512:16 1024:4 4096:1 --hidden-size 2048 --num-heads 16
this would fit barely onto a RTX 5090 with 32GB memory.