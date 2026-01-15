# CSC8614 — TP2

**Name** : ROLAND Jade

## Environment & Reproducibility

- **OS** : Windows 10
- **Python** : 3.10
- **Main libraries** :
  - torch
  - transformers
  - scikit-learn
  - plotly

### Installation / environment activation

`python -m venv venv`
`source venv/Scripts/activate`
`cd ./TP2`
`pip install -r requirements.txt`

---

![](./screenshots/'Capture d'écran 2026-01-15 100132.png')

### Question 2:
The `settings` object is of type **Python dictionary** (`<class 'dict'>`).
It contains the model's hyperparameters. Its main keys are:
* `n_layer` : 12 (Number of transformer layers).
* `n_head` : 12 (Number of attention heads).
* `n_embd` : 768 (Dimension of embedding vectors).
* `n_vocab`, `n_ctx` : (Vocabulary size and context window).

This configuration corresponds to the standard specifications of the **GPT-2 Small** architecture (117 million parameters).

### Question 3:
The `params` object is also a **dictionary** (`<class 'dict'>`).
It contains the pre-trained model tensors. Its structure includes:
* `wte` : *Word Token Embeddings* (Token embedding matrix).
* `wpe` : *Word Position Embeddings* (Positional encoding).
* `blocks` : Contains hidden layers (Multi-Head Attention + MLP).
* `b`, `g` : Parameters associated with biases and normalization (LayerNorm).

### Question 4:
The `__init__` method of a GPT model expects a structured configuration to size the layers. Although `settings` is a dictionary, the code converts key-value pairs into attributes usable by the model.

### Question 5.1:
`df.sample(frac=1, random_state=123)` --> the instruction performs a random shuffle of the dataset.
Raw datasets are often sorted by label. Without this shuffle, the train/test split risks placing a single class in the test set.

### Question 5.2:

![](./screenshots/'Capture d'écran 2026-01-15 100207.png')
![](./screenshots/'Capture d'écran 2026-01-15 100220.png')

* **Ham (Non-spam) :** 3860 samples (~86.6 %).
* **Spam :** 597 samples (~13.4 %).

![](./screenshots/'Capture d'écran 2026-01-15 100234.png')

The data is **heavily unbalanced**. A model could achieve high accuracy by systematically predicting the majority class ("Ham"), without ever learning to detect spam, hence the use of `class_weights` during training.

### Question 7:

![](./screenshots/'Capture d'écran 2026-01-15 100252.png')
![](./screenshots/'Capture d'écran 2026-01-15 100317.png')

* **Training size (subsampled) :** 2000 samples.
* **Batch size :** 16 (visible in the capture `Input batch shape`).

$$\text{Number of batches} = \frac{\text{Training Size}}{\text{Batch Size}} = \frac{2000}{16} = 125$$

There will therefore be **125 iterations** per epoch.

### Question 8.1
Number of classes :** `num_classes = 2` (Binary : Ham vs Spam).

### Question 8.2:

![](./screenshots/'Capture d'écran 2026-01-15 100332.png')

`Linear(in_features=768, out_features=50257, bias=False)` serves to predict the next word among the vocabulary.
`Linear(in_features=768, out_features=2, bias=True)` sends the hidden state to two logits for classification.

### Question 8.3:
`requires_grad = False` freezes internal layers to preserve linguistic knowledge (grammar, semantics) acquired by GPT-2 during its pre-training: only the new head is trained. This reduces computational cost.

---

### Question 10:

![](./screenshots/'image.png')

Analyzing the logs of the first training attempt (with `lr=5e-5`) reveals disappointing results:
Global accuracy remains frozen around **86.6%**, while accuracy specific to the Spam class remains stuck at **0.00%**. The loss function oscillates but shows no clear convergence towards zero.

This is a **"Mode Collapse"** or convergence towards the majority class.
Despite applying `class_weights`, the learning rate (`5e-5`) was too low. Since the model body is frozen and the classification head is initialized randomly, this rate does not allow gradients to modify weights sufficiently to exit this trivial local minimum in a few epochs.

---

### Question 11:
To remedy the problem identified in Q10, we modify the **learning rate**. Thus, we force the optimizer to perform larger weight updates. This should allow the new classification head to "break" the initial bias and learn the discriminating characteristics of Spam.

In the code, we therefore switch the LR from `5e-5` to `1e-3` (multiplication by 20) :
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

We restart the training loop with this new optimizer, and we expect to see Spam Accuracy increase rapidly (> 50%).

![](./screenshots/'Capture d'écran 2026-01-15 122750.png')

* **Global Accuracy:** Dropped to ~64.66%.
* **Spam Accuracy:** Jumped significantly to **98.00%**.

The intervention was successful in waking up the model. The Spam accuracy skyrocketed from 0% to 98%, proving the new head is learning features.
However, we observe an overcorrection. The combination of a high learning rate and aggressive class weights caused the model to become "paranoid," likely classifying many legitimate messages (Ham) as Spam (False Positives), which explains why the global accuracy dropped below the baseline of 86%.