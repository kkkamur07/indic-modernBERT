### Guidelines 

Beware of the hardware requirements for wave, tiling and tensor quantization of the GPU. 

1. Tensors 128 Bytes : So around 64 size in FP16
2. Wave : SM Processors
3. Tiling : Tiles are in blocks of 128*256 blocks. 


### Tokenizations Approaches

1. BPE + SuperBPE
2. Sentence Piece

We need to do `ScriptNormalization` and how are we going to do that is the question mostly with `indicNLP` library and sampling strategy as well to build a very good tokenizer. 

Need to inspect the folder structure of the `sangrah` dataset and then download random datasets to ensure that we have fair representation, we can calculate the size of the dataset as proxy and then only download the required.

### Vendored SuperBPE patch: `merges.txt` leading-space parsing

SuperBPE stage 2 reloads stage-1 `merges.txt` and extends it. Our Hindi BPE checkpoints
include word-initial tokens with a **leading space** (HF/GPT-2 convention). `BPE::save`
writes those lines as two leading spaces, e.g. `"  क"` means merge `(" ", "क")`.

**Upstream bug:** `do_train_extend` in the fork used `line.split(" ")` (all spaces),
which mis-parses those lines and can panic (`Option::unwrap() on None`) or spam
` not found in word_to_id` during stage 2.

**Our local fix** (in the nested submodule, not upstream yet):

| File | Change |
|------|--------|
| `tokenizers/src/models/bpe/model.rs` | `parse_bpe_merge_line()` — handles `"  …"` via `strip_prefix("  ")` |
| `tokenizers/src/models/bpe/trainer.rs` | `do_train_extend` uses `parse_bpe_merge_line` instead of `split(" ")` |

**If you update `tokenizers_superbpe`:** re-check this fix is still present; re-apply if
the submodule was reset. Rebuild the editable wheel after Rust changes:

```bash
uv pip install -e _support_repo/superbpe/tokenizers_superbpe/bindings/python --force-reinstall --no-deps
```

**Smoke check:** `make validate-superbpe` (tiny Hindi phrase corpus).