We compare ROME against several open sourced state-of-the-art model editors. All are implemented in their respective folders. Implementations other than FT/FT+L are adapted from third parties.
- Fine-Tuning (FT): Direct fine-tuning.
- Constrained Fine-Tuning (FT+L): FT with $L_\infty$ norm constraint. Inspired by Zhu et al. [[Paper]](https://arxiv.org/abs/2012.00363)
- Knowledge Neurons (KN): Dai et al. [[Code]](https://github.com/EleutherAI/knowledge-neurons) [[Paper]](https://arxiv.org/abs/2104.08696)
- Knowledge Editor (KE): De Cao et al. [[Code]](https://github.com/eric-mitchell/mend) [[Paper]](https://arxiv.org/abs/2104.08164)
- Model Editor Networks with Gradient Decomposition (MEND): Mitchell et al. [[Code]](https://github.com/eric-mitchell/mend) [[Paper]](https://arxiv.org/abs/2110.11309)