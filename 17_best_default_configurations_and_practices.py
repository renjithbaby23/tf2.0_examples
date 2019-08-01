"""
######### Default DNN configuration ###########
_________________________________________________________
| Hyperparameter          | Default value               |
---------------------------------------------------------
| Kernel initializer      | LeCun initialization        |
| Activation function     | SELU                        |
| Normalization           | None (self-normalization)   |
| Regularization          | Early stopping              |
| Optimizer               | Nadam                       |
| Learning rate schedule  |Performance scheduling       |
---------------------------------------------------------

Don’t forget to standardize the input features! Of course, you should also try to reuse
parts of a pretrained neural network if you can find one that solves a similar problem,
or use unsupervised pretraining if you have a lot of unlabeled data, or pretraining on
an auxiliary task if you have a lot of labeled data for a similar task.


The default configuration in the above table may need to be tweaked as follows:

• If your model self-normalizes:
— If it overfits the training set, then you should add alpha dropout (and always
use early stopping as well). Do not use other regularization methods, or else
they would break self-normalization.
• If your model cannot self-normalize (e.g., it is a recurrent net or it contains skip
connections):
— You can try using ELU (or another activation function) instead of SELU, it
may perform better. Make sure to change the initialization method accord‐
ingly (e.g., He init for ELU or ReLU).
— If it is a deep network, you should use Batch Normalization after every hidden
layer. If it overfits the training set, you can also try using max-norm or l 2 reg‐
ularization.
• If you need a sparse model, you can use l 1 regularization (and optionally zero out
the tiny weights after training). If you need an even sparser model, you can try
using FTRL instead of Nadam optimization, along with l 1 regularization. In any
case, this will break self-normalization, so you will need to switch to BN if your
model is deep.
• If you need a low-latency model (one that performs lightning-fast predictions),
you may need to use less layers, avoid Batch Normalization, and possibly replace
the SELU activation function with the leaky ReLU. Having a sparse model will
also help. You may also want to reduce the float precision from 32-bits to 16-bit
(or even 8-bits) (see ???).
• If you are building a risk-sensitive application, or inference latency is not very
important in your application, you can use MC Dropout to boost performance
and get more reliable probability estimates, along with uncertainty estimates.

"""