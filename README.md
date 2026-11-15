# ecgtok

Named tokens for one MIT-BIH beat, compared with opaque energy codes of the same window. Both strings are classified by Qwen2.5-1.5B with the same LoRA setup. The question is whether names a reader already uses beat a code that only records how much energy sat in each slice of the beat.

The tokens are defined in [docs/taxonomy.md](docs/taxonomy.md). This page states the protocol, the scores, and how to rerun them.

## Protocol

Beats are the MIT-BIH Arrhythmia Database QRS annotations that map to an AAMI class [1], [3]. The lead is MLII. Paced records 102, 104, 107, and 217 are excluded. The remaining records follow the de Chazal DS1/DS2 division [2].

DS1 trains the model. The last 15% of each DS1 record, in time, selects the epoch. For ventricular versus rest, that same slice selects the decision threshold by F1, over 0.05 to 0.95 in steps of 0.05. DS2 is scored once from the selected checkpoint. It is not used to choose an epoch or a threshold.

Four tasks:

- Ventricular ectopic beats versus the rest (`V` and `E` against every other AAMI class). The reported F1 is the F1 of the ventricular class.
- Five AAMI classes: N, S, V, F, Q.
- The MIT-BIH beat symbols that appear in DS1 training, without collapsing them.
- The rhythm label in force at the beat, taken from annotation notes that start with `(`.

Macro AUC is the mean of one-versus-rest AUCs over classes that have both positives and negatives on that split. Macro F1 uses the same classes. A class that is missing from a split is omitted from that split's mean, not filled in as zero.

The model is Qwen2.5-1.5B [4]. LoRA adapts `q_proj` and `v_proj` with rank 16 and alpha 32 [5]. New token rows are a separate float32 table. Pretrained embedding rows stay frozen. Training uses AdamW at `1e-4`, batch 64, two epochs, seed 42, and a maximum length of 24. A non-finite loss or gradient aborts the run. The binary loss is weighted by the negative-to-positive ratio. Multiclass loss uses inverse-frequency weights, capped at 10.

## Results

The table is the seed-42 run. Numbers are also in [results/mitbih_qwen.json](results/mitbih_qwen.json). A later run of the same script can move the third decimal: the seed fixes the initialization and the shuffle, not every GPU kernel.

DS2 is the held-out split. Named tokens have the higher DS2 AUC on all four tasks. On the DS1 validation slice, which is what selected the epoch, named tokens win only ventricular versus rest. Opaque codes score higher there for AAMI-5, symbols, and rhythm. The two splits do not agree, and the table keeps both.

| Task | Split | Named AUC | Opaque AUC | Named F1 | Opaque F1 |
|---|---|---:|---:|---:|---:|
| Ventricular vs rest | DS1-val | 0.986 | 0.946 | 0.859 | 0.652 |
| Ventricular vs rest | DS2 | 0.969 | 0.929 | 0.650 | 0.508 |
| AAMI-5 | DS1-val | 0.784 | 0.827 | 0.515 | 0.429 |
| AAMI-5 | DS2 | 0.771 | 0.685 | 0.337 | 0.263 |
| Beat symbols | DS1-val | 0.834 | 0.845 | 0.351 | 0.349 |
| Beat symbols | DS2 | 0.679 | 0.611 | 0.220 | 0.143 |
| Rhythm | DS1-val | 0.719 | 0.826 | 0.218 | 0.321 |
| Rhythm | DS2 | 0.712 | 0.614 | 0.156 | 0.124 |

Ventricular F1 is the positive-class F1. Every other F1 is a macro F1. Epoch 2 was selected for both arms of ventricular versus rest, symbols, and rhythm, and for the opaque arm of AAMI-5. The named AAMI-5 arm selected epoch 1 (DS1-val macro AUC 0.784, against 0.768 at epoch 2). DS2 for that arm is the epoch-1 checkpoint.

Ventricular thresholds, chosen on DS1 validation: 0.80 named, 0.85 opaque. At those thresholds the DS2 operating point is sensitivity 0.859 and positive predictive value 0.523 for named tokens, against 0.704 and 0.397 for opaque codes. The AUC gain does not mean the ventricular decisions are precise.

AAMI-5 validation has no Q beats in the DS1 tails, so that macro AUC is the mean of N, S, V, and F. DS2 scores all five. Named DS2 AUCs: F 0.865, N 0.789, Q 0.776, S 0.483, V 0.939. Opaque: F 0.606, N 0.795, Q 0.452, S 0.667, V 0.906. Supraventricular beats are the class the named tokens do not win.

Symbol `S` has two beats in the whole set and is not scored on DS2. Nine symbol classes are scored on DS1 validation and ten on DS2.

Rhythm DS2, named against opaque: AFIB 0.794 / 0.660, AFL 0.754 / 0.509, VT 0.895 / 0.710, B 0.828 / 0.665, N 0.711 / 0.672, NOD 0.663 / 0.527, SVTA 0.604 / 0.589, T 0.443 / 0.578. Nine rhythm classes are scored on DS1 validation and eight on DS2. Ventricular flutter is annotated in the database, but those annotations do not fall on QRS beats in this table, so flutter is not a class.

## Other embeddings

Two further inputs use the same window, the same Qwen2.5-1.5B LoRA, and the same DS1/DS2 rule. They are described in [docs/external.md](docs/external.md). Fixed-scale quantization (FSQ) is the discretizer from HeartLLM [6], reduced to one MLII lead and a reconstruction loss. It does not pretrain Qwen to forecast ECG codes. Patch reprogramming follows TIME-LLM [7]: 13 patches cross-attend to 64 frozen rows of Qwen's input embedding. It does not forecast the trace. The scores are in [results/external_qwen.json](results/external_qwen.json).

| Task | Split | Named | Opaque | FSQ | Reprogram |
|---|---|---:|---:|---:|---:|
| Ventricular vs rest | DS1-val | 0.986 | 0.946 | 0.970 | 0.970 |
| Ventricular vs rest | DS2 | 0.969 | 0.929 | 0.942 | 0.925 |
| AAMI-5 | DS1-val | 0.784 | 0.827 | 0.854 | 0.882 |
| AAMI-5 | DS2 | 0.771 | 0.685 | 0.704 | 0.707 |
| Beat symbols | DS1-val | 0.834 | 0.845 | 0.881 | 0.905 |
| Beat symbols | DS2 | 0.679 | 0.611 | 0.631 | 0.662 |
| Rhythm | DS1-val | 0.719 | 0.826 | 0.792 | 0.833 |
| Rhythm | DS2 | 0.712 | 0.614 | 0.672 | 0.707 |

The cells are AUC. On DS2, named tokens remain the highest of the four inputs on every task. On the DS1 slice that selects the epoch, reprogramming is highest for AAMI-5, symbols, and rhythm, and FSQ is next on AAMI-5 and symbols. Named tokens remain highest on that slice only for ventricular versus rest. The AAMI-5 reprogramming arm selected epoch 1. Every other new arm selected epoch 2.

## Reproduce

The database and the weights are downloaded. They are not in git.

```powershell
uv sync --extra dev --extra train
uv run python scripts/download_mitbih.py data/mitdb
uv run python scripts/download_qwen.py
uv run python scripts/build_beats.py data/mitdb cache/beats.parquet
uv run python scripts/train_qwen.py cache/beats.parquet
uv run python scripts/train_external.py data/mitdb models/Qwen2.5-1.5B
```

`train_external.py` reads the recordings directly and writes `results/external_qwen.json`. It needs the same GPU recipe.

`train_qwen.py` reads `models/Qwen2.5-1.5B`. The reported recipe needs a CUDA GPU. Batch 64 uses about 5 GB. The script writes `runs/qwen_mitbih.json` and `runs/train.log`.

## References

[1] G. B. Moody and R. G. Mark, "The impact of the MIT-BIH arrhythmia database," *IEEE Eng. Med. Biol. Mag.*, vol. 20, no. 3, pp. 45–50, May/Jun. 2001.

[2] P. de Chazal, M. O'Dwyer, and R. B. Reilly, "Automatic classification of heartbeats using ECG morphology and heartbeat interval features," *IEEE Trans. Biomed. Eng.*, vol. 51, no. 7, pp. 1196–1206, Jul. 2004.

[3] Association for the Advancement of Medical Instrumentation, *Testing and reporting performance results of cardiac rhythm and ST-segment measurement algorithms*, ANSI/AAMI EC57:2012.

[4] Qwen Team, "Qwen2.5 technical report," arXiv:2412.15115, 2024.

[5] E. J. Hu *et al.*, "LoRA: Low-rank adaptation of large language models," in *Proc. Int. Conf. Learn. Represent.*, 2022.

[6] J. Yang *et al.*, "HeartLLM: Discretized ECG tokenization for LLM-based diagnostic reasoning," arXiv:2508.15338, 2025.

[7] M. Jin *et al.*, "Time-LLM: Time series forecasting by reprogramming large language models," in *Proc. Int. Conf. Learn. Represent.*, 2024.
