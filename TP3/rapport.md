# CSC8614 — TP3

**Name**: ROLAND Jade

## Environment & Reproducibility

- **OS**: Windows 10
- **Python**: 3.10

### Installation / environment activation

`python -m venv venv`
`source venv/Scripts/activate`
`cd ./TP2`
`pip install -r requirements.txt`

---

## Question 1:
Clear differences are visible in the model architecture when comparing the logs before and after injection.

**Before LoRA:** The Transformer blocks (`TransformerBlock`) consist of standard linear layers (`nn.Linear`). For example, in the attention module, `W_query`, `W_key`, `W_value`, and `out_proj` are direct instances of `Linear`.
**After LoRA:** These same layers have been replaced by our custom class `LinearWithLoRA`.

By inspecting the detailed structure, we can observe that each `LinearWithLoRA` encapsulates two distinct components working in parallel:
* `(linear)`: The original linear layer (whose weights are frozen).
* `(lora)`: Our `LoRALayer` module containing the low-rank matrices $A$ and $B$ (which are trainable).



This confirms that the recursive modification of the model worked: the architecture was adapted to inject LoRA parameters without altering the main path of the pre-trained model.

---

## Question 2:
Following the LoRA injection (with rank $r=8$) and the freezing procedure, we obtain the following statistics:

* **Trainable Parameters:** 1,327,104
* **Total Parameters:** 164,364,288
* **Trainable Fraction:** **0.81%**

This result demonstrates the efficiency of the LoRA method. Instead of having to update all 164 million parameters of GPT-2, we only need to train about 1.3 million parameters (less than 1%). This significantly reduces the VRAM memory required to store optimizer states.

---

## Question 3:
After modifying the model for the binary classification task (replacing the output head), the statistics change significantly:

* **Trainable Parameters:** 1,328,642
* **Total Parameters:** 125,768,450
* **Trainable Fraction:** **1.06%**

**Comparison:**

**Increase in Trainable Parameters (+1,538):**
The number of trainable parameters increased very slightly compared to Question 2 (going from 1,327,104 to 1,328,642). This corresponds exactly to the addition of the new classification head (`out_head`).
    * This layer projects the hidden dimension (768) to the number of classes (2).
    * Calculation: $(768 \times 2 \text{ weights}) + (2 \text{ biases}) = 1,538$ additional parameters.

**Massive Decrease in Total Parameters (-38.6M):**
The total number of parameters dropped drastically (from ~164M to ~125M).
    * **Cause:** To perform classification, we removed the original GPT-2 vocabulary layer (`lm_head`), which served to predict the next word among 50,257 possibilities. This matrix was very voluminous ($768 \times 50,257 \approx 38.6$ million parameters).
    * It was replaced by a much smaller matrix dedicated to binary classification.

**Increase in Fraction (0.81% $\to$ 1.06%):**
The percentage of trainable parameters mechanically increased. This is not because we are training significantly more parameters, but because the denominator (the total number of model parameters) significantly decreased following the removal of the vocabulary head.

---

## Question 4:
The loss shows a rapid convergence. It starts high at 3.17 (Batch 0), then drops significantly to 0.36 by Batch 10 and reaches extremely low values (e.g., 0.0055 at Batch 60). Although there are some fluctuations and spikes towards the end (e.g., rising to 0.98 at Batch 140), the Average Loss for the epoch is 0.2104, which is quite low. The fluctuations are likely due to the small batch_size=8. With few examples per step, a single "hard" example can cause a temporary spike in loss.

The model achieved an accuracy of 91.69% on the training set after just one epoch. This is highly reasonable and can be due to multiple factors:

- Spam classification is semantically distinct. It is a much easier task than complex reasoning or generation.
- GPT-2 already understands English grammar and semantics. It does not need to learn the language from scratch, only to adapt its existing knowledge to distinguish two specific categories.

The result proves that modifying only ~1% of parameters is sufficient to steer the model effectively towards a new task.

---

## Question 5:

**Test Set Accuracy: 96.99%**

The Test accuracy is actually higher than the average Training accuracy (91.69%).
An accuracy of ~97% on unseen data is an excellent result. It confirms that the model has successfully learned the features of Spam messages without memorizing the training data. There is no overfitting (which would be characterized by high Train accuracy but low Test accuracy).

It can be due to two reasons:
- The Training accuracy (91.69%) is the average over the entire epoch. At the start of the epoch, the model was untrained and performing poorly. The Test set was evaluated only after the model finished learning.
- During training (model.train()), dropout layers randomly disable neurons, making prediction harder. During testing (model.eval()), dropout is turned off, often leading to better performance.

---

Finally, we performed a manual inference test on specific, unseen sentences to verify the model's behavior in real-world scenarios.

'Call immediately for free vacation.'
SPAM 0.61 (61%)
The model correctly flagged the message.

'Hey, are we still meeting for lunch tomorrow?'
HAM 0.98 (98%)
The model correctly identified this as normal conversation with very high confidence.

This laboratory successfully demonstrated the power of LoRA.
We built the LoRA layers and the injection mechanism from scratch. We transformed a 124M parameter model using only ~1.3M trainable parameters (1.06% of the total). Despite freezing 99% of the model, we achieved a Test Accuracy of 96.99%, proving that LoRA allows for effective fine-tuning with a fraction of the computational cost required for full fine-tuning.