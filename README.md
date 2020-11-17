# MusicTransformer
Music Transformer - project for CM0492

Sequence to sequence transformer using Global Relative Attention for improved complexity.
- Based on [Attention is All You Need](https://arxiv.org/abs/1706.03762) and [Music Transformer](https://arxiv.org/abs/1809.04281).
- Pyorch Transformer implementation based on [this](https://pytorch.org/tutorials/beginner/transformer_tutorial.html) tutorial and on [this](https://github.com/jason9693/MusicTransformer-pytorch) implementation.
- Preprocess implementation repository is [here](https://github.com/jason9693/midi-neural-processor).

## Dataset: PIANO-E-COMPETITION
The dataset is a set of piano recordings of a piano competition spanning various years.
The music is piano music from the XVIII to XX century: Albeniz, Beethoven, Chopin, etc.

MIDI encoding depends on the type of music:
- live recordings vs. non-live midi composition
- instrumentation
- style
- ...

The MIDI encoding used for this project is the second presented in Music Transformer.
Vocabulary is composed by 388+3 tokens: 
- 128 NOTE-ON events
- 128 NOTE-OFF events
- 100 TIME-SHIFT events
- 32 VELOCITY bins
- <sos>, <eos>, <pad> tokens

## Training
Training is done using sliding sequences, as in the tutorial.

- Given a sequence <A, B, C, D>, input sequence is <A, B, C>, target sequence is <B, C, D>
- Input sequences **B** batches of subsequences of length **L** randomly sampled from longer sequences of variable length _len_ starting anywhere from index _0_ to _len_-**L**.

The scheduler is Noam Optimizer, discussed in the paper Attention is All You Need.
The Loss function is cross entropy with label smoothing.



