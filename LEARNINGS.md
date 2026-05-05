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