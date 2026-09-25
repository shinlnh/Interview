# 3.2 Masked Language Modeling — MLM

[← Mục lục Machine Learning and Mathematics](./README.md)

**Mức đã trao đổi:** **3/5**

### Interviewer có thể hỏi thêm

- What is Masked Language Modeling?

  Masked Language Modeling, or MLM, is a self-supervised pretraining objective. The model receives a sequence in which some tokens have been hidden or replaced and learns to recover the original tokens from their surrounding context.

  Because the prediction can use tokens on both the left and the right, MLM helps the model learn bidirectional contextual representations.

- Which famous model uses MLM?

  **BERT** is the best-known model pretrained with MLM. RoBERTa, ALBERT, and many other encoder-only Transformer models also use variants of this objective.

- How is BERT trained?

  The original BERT is first pretrained on unlabeled text using two objectives:

  - **Masked Language Modeling:** Predict selected original tokens from their context.
  - **Next Sentence Prediction:** Predict whether one sentence follows another in the source text.

  For MLM, BERT selects 15% of the input tokens. Among those selected tokens, 80% are replaced by `[MASK]`, 10% by a random token, and 10% are left unchanged. After pretraining, the model is fine-tuned on a downstream task using labeled data.

- What happens when a token is masked?

  The token's original identity is hidden from the input, usually by replacing it with `[MASK]`. BERT processes the entire sequence with bidirectional self-attention and uses the final hidden state at that position to predict a probability distribution over the vocabulary.

  The token is not assigned a score of zero. Its contextual representation is computed from the visible tokens on both sides. Also, because of BERT's 80/10/10 replacement rule, a selected token is not always literally replaced by `[MASK]`.

- What is the training target?

  The target is the **original token ID at each selected position**. Positions that were not selected for MLM do not contribute to the MLM loss. In original BERT, Next Sentence Prediction has a separate binary target, but it is not part of the token-level MLM target.

- Which loss is commonly used?

  MLM commonly uses **cross-entropy loss over the vocabulary**, evaluated only at the selected token positions. If $\mathcal{M}$ is the set of selected positions, the loss is:

```math
\mathcal{L}_{\text{MLM}} = -\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log p_\theta\!\left(x_i \mid \tilde{x}\right)
```

  Here, $x_i$ is the original token and $\tilde{x}$ is the corrupted input sequence. In code, unselected positions are typically ignored with a special label such as `-100`.

- Why is cross-entropy used?

  Predicting a masked token is a multi-class classification problem: the model produces one logit for every token in the vocabulary, while the original token provides the correct class. Cross-entropy converts these logits through softmax and minimizes the negative log-probability of the correct token.

  It is differentiable, strongly penalizes confident wrong predictions, and is equivalent to maximum likelihood for a categorical target. Original BERT also uses cross-entropy for NSP, but that is a separate binary classification loss.

- What is the difference between MLM and autoregressive language modeling?

  **MLM** corrupts selected input tokens and predicts their original values using context from both sides. It normally computes loss only at the selected positions and can predict several positions in parallel.

  **Autoregressive language modeling** factorizes a sequence from left to right:

```math
p(x_1,\ldots,x_T) = \prod_{t=1}^{T}p(x_t\mid x_{\lt t})
```

  It uses a causal attention mask so position $t$ cannot access future tokens. During generation, each predicted token is appended to the context before predicting the next one. The `[MASK]` corruption used by MLM and the causal attention mask used by an autoregressive model therefore serve different purposes.

- BERT vs GPT training objective?

  The original **BERT** is an encoder model pretrained with MLM and NSP. Its objective is to recover masked tokens from bidirectional context and learn representations that can be fine-tuned for understanding tasks such as classification or question answering.

  **GPT** is a decoder-only model pretrained with causal next-token prediction. It maximizes the probability of each token conditioned only on earlier tokens, which directly trains it to generate text from left to right.

  Modern BERT-like models may omit NSP, and modern GPT-like models may add instruction tuning or preference optimization, but their core pretraining objectives remain bidirectional masked-token prediction versus causal next-token prediction.

- Why can BERT use information from both left and right context?

  BERT uses a Transformer encoder without a causal triangular mask. Except for padding restrictions, every position can attend to tokens before and after it through self-attention. When a selected token is hidden, its representation can therefore combine evidence from both directions while predicting the original token.

- Why is MLM not naturally autoregressive?

  MLM does not learn the left-to-right factorization required for autoregressive generation. It predicts selected positions from a corrupted sequence, often predicts several positions in parallel, and may use future context that would not exist during left-to-right generation.

  It also introduces special `[MASK]` inputs that normally do not appear in generated text. An MLM can generate through iterative masking and resampling, but this requires a separate procedure and is not its native training behavior.

- What is a tokenizer?

  A tokenizer converts raw text into a sequence of token IDs that a model can process. A typical tokenizer normalizes text, segments it into words or subwords, maps those pieces to a fixed vocabulary, and inserts special tokens such as `[CLS]`, `[SEP]`, or `[MASK]` when required.

  Subword or byte-level tokenization limits out-of-vocabulary problems because an unfamiliar word can be represented by smaller known units. Decoding performs the reverse mapping from token IDs back to text.

- BPE vs Unigram tokenizer?

  **Byte Pair Encoding — BPE** starts with small symbols, such as characters or bytes, and repeatedly merges the most frequent adjacent pair. The learned merge rules are then applied to segment new text. BPE is simple and fast, but its greedy merge process does not explicitly model the probability of every possible segmentation.

  **Unigram tokenization** starts with a large candidate vocabulary and assigns a probability to each token. It repeatedly removes tokens whose removal hurts the corpus likelihood the least. At inference, it can use the Viterbi algorithm to select the highest-probability segmentation or sample alternative segmentations.

  In short, BPE builds the vocabulary by adding frequent merges, whereas Unigram begins with many candidates and prunes them using a probabilistic objective.

- Is likelihood used in tokenizer training?

  It depends on the tokenizer algorithm. A **Unigram language-model tokenizer** explicitly optimizes corpus likelihood by assigning probabilities to tokens and considering possible segmentations:

```math
p(x) = \sum_{s\in\mathcal{S}(x)} \prod_{u\in s}p(u)
```

  Here, $\mathcal{S}(x)$ is the set of valid segmentations of string $x$. By contrast, standard BPE greedily chooses frequent pair merges rather than directly maximizing this likelihood. Therefore, likelihood is central to Unigram training but is not a universal objective for every tokenizer.

---
