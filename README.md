# ModelWeave


## Potential Models?

- OLMo-1 7B, base and instruct variants (2x)
- OLMo-2 7B, base and instruct variants (2x)
- Pythia 6.9B (1x)
- Command R 7B (1x) (plus subvariants?)
- Aya 23 8B (plus subvariants?)
- Aya expanse 8B (plus subvariants?)



## Overarching idea
### Paper inspirations
#### Procedural Knowledge in Pretraining Drives Reasoning in Large Language Models (ProcKnow)
- The procedural knowledge paper by Ruis et al. 2024 provides us with a great datapoint on the recall vs. generalization spectrum!
- It appears that for factual questions, models learn specific answers from specific documents (we know this as documents containing answers receive high scores by EK-FAC influence functions w.r.t the correct completion of a factual recall prompt)
- Meanwhile, for a given reasoning task, there is a set of documents that is moderately influential w.r.t the correct completion of prompts within that task. For another reasoning task, it is a different set

#### Arithmetic Without Algorithms: Language Models Solve Math With a Bag of Heuristics (BoH)
- There is a set of neurons in current LLMs (Llama 3 8B, Pythia-6.9B) that is responsible for arithmetic capabilities, dubbed Bag-of-Heuristics 
- This is primarily a sparse set of MLP neurons, which achieves a 96% faithfulness score w.r.t. explaining downstream arithmetic capability
- The BoH represents a group of memorized heuristics w.r.t. individual operands or dynamics between them, emerges throughout training

#### Fine-tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking (FteeM)
- Models fine-tuned on code and math are better at entity tracking (ET)
- After fine-tuning, we see that the same circuit that performs ET in the base model also does so in the FT model and restores 88% of FT performance
- This common circuit works via groups of self_attn headsd that implement semantic subtasks
- ET is performed by indentifying and transporting positions of queried entities in the context
- Multiple self_attn head groups collaborate to pass information downstream
- This scheme and the role of heads/head groups remains the same between the base and the FT model

### Taking this forward
#### Project constraints
- Cannot access LLM training data aside from OLMo-1, OLMo-2 and Pythia
- Could reproduce both on the fully trained base models?
    - Ie. identify BoH on [OLMo-1, OLMo-2, Pythia, Command R, Aya 23, Aya Expanse]
    - Ie. fine-tune [OLMo-1, OLMo-2, Pythia, Command R, Aya 23, Aya Expanse] on Math data, identify ET circuit in base and FT
    