# External embeddings

The named tokens and the opaque codes are two strings fed to the same Qwen2.5-1.5B LoRA. This page adds two other ways to put the same MLII window into that model. Neither is a reproduction of the paper it borrows from. The split, the four tasks, the epoch rule, and the classification head stay the ones in the README.

## Fixed-scale codes

HeartLLM trains a lead-wise encoder, rounds the latent with a fixed-scale quantizer, and then pretrains the language model to forecast those codes on 12-lead records. The arm here keeps only the quantizer.

One MLII beat, 325 samples, goes through two stride-5 convolutions and is pooled to 13 frames. Each frame is a 4-dimensional vector. A sigmoid squashes each coordinate, and each coordinate is rounded to one of 5 levels. The four levels pack into one integer in `0 … 624`, written `<FSQ_n>`. Thirteen of those tokens are the beat.

The encoder and a small decoder are trained for 5 epochs to reconstruct DS1 beats. DS2 is not in that loss. After that the quantizer is frozen. Unseen codes at scoring time use `<ECG_UNK>`, the same rule as the named tokens. The language model is not pretrained to forecast the codes. It receives the same LoRA classification training as the named arm.

## Patch reprogramming

TIME-LLM cuts a series into patches and cross-attends those patches to text prototypes drawn from a frozen language model, then forecasts. The arm here keeps the patch and the prototypes, and replaces the forecast with the beat-classification head.

The 325 samples are 13 patches of 25. A linear layer lifts each patch to Qwen's width, 1536. Sixty-four rows of the frozen input embedding, taken at a fixed stride through the vocabulary, are the prototypes. One cross-attention block reads them. Its output is the `inputs_embeds` sequence. The prototypes do not train. The patch layer, the attention, the LoRA adapters, and the classification head do.

There is no text prompt and no forecast loss.

## What is shared

Both arms use Qwen2.5-1.5B, LoRA on `q_proj` and `v_proj` with rank 16 and alpha 32, AdamW at `1e-4`, batch 64, two epochs, and seed 42. DS1 validation selects the epoch. Ventricular versus rest also selects the threshold there. DS2 is scored once. A non-finite loss or gradient aborts the run.
