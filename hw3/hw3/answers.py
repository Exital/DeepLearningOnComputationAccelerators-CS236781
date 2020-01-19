r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0, seq_len=0,
        h_dim=0, n_layers=0, dropout=0,
        learn_rate=0.0, lr_sched_factor=0.0, lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=128, seq_len=120,
        h_dim=128, n_layers=12, dropout=0.1,
        learn_rate=0.002, lr_sched_factor=0.002, lr_sched_patience=0.1,
    )
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = .0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "The"
    temperature = .0001
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**
Denying the model to train over the whole text prevents overfitting. With access to the entire text, 
the model might learn it through memorization

"""

part1_q2 = r"""
**Your answer:**


"""

part1_q3 = r"""
**Your answer:**
The batches are connected via the hidden states which flow thorugh them.
If we would shuffle the batches, those connections might be lost.

"""

part1_q4 = r"""
**Your answer:**
1. The temperature holds a trade-off in it.
Higher temperature causes the model to be more risky, and try unfamilier approaches.
While lower temperature leads the model to select options with higher probabilities.
That is why we prefer lower temp while sampling.

Consider the softmax equation when discussing the next points:
$P_t(a) = \frac{e^{q_t(a)/t}}{\sum_{i=1}^n e^{q_t(i)/t}}$

2. For higher temperature, $t \longrightarrow \infty$, the actions have almost the same probability. That is why the
model is likely to pick letter randomly, with no bias towards any specific letter.

3. For lower temperature, $t \longrightarrow 0$ actions with expected high yield reward get higher probabilities,
and so the model is more probable to select those letters over the others.

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=20,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 32
    hypers['h_dim'] = 256
    hypers['z_dim'] = 256
    hypers['x_sigma2'] = 0.45
    hypers['learn_rate'] = 0.00025
    hypers['betas'] = (0.9, 0.99)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
$\sigma^2$ controls the level of randomness of the model. It regulates the relative part of the reconstruction loss.
Lower $\sigma$ means lower variance, and so we'll get more weight to the reconstruction loss.
Higher $\sigma$ means higher variance, which will lead to mroe weight for the kl-div loss.

As much as the reconstruction loss is bigger with respect to kl-div loss, the results of the model will be
more similar to the dataset the model was trained on.
The opposite will result with a more silimar distribution to the one of the dataset. 

"""

part2_q2 = r"""
**Your answer:**
1. The role of the reconstruction loss is to make sure the distribution of the reconstruction
will be closer to the distribution of the data set. Given the encoding of x $P_{\beta}(x|z)$,
when we assume Z~N(0,1).

The role of kl-div loss is to make sure the distribution of the data post encoding ($q_{\alpha}(z|x)$)
will be close to the standard normal distribution.


2. When the kl-div loss component is given more weight, we'd get that the distribution after
the encoding phase will be closer to standard normal dostribution.


3. If we weren't using the kl-div component in the loss function we would have overfitting.
In order to minimize the the expression:

$E_{u~N(0,1)} [ \frac{1}{2\alpha^2} || x-\psi_{\beta}(\mu_{\alpha}(x) + \sum_{\alpha}^{0.5}(x)u) ||]$
we would reconstruct the avarage of the dataset after the encoding.

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=32, z_dim=128,
        data_label=1, label_noise=0.28,
        discriminator_optimizer=dict(
            type='Adam',
            weight_decay=0.02,
            betas=(0.5,0.999),
            lr=0.0002,
        ),
        generator_optimizer=dict(
            type='Adam',
            weight_decay=0.02,
            betas=(0.5,0.999),
            lr=0.00021,
        ),
    )
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**
When we train the discriminator  we use the data from the generator, but we don't want to update it.
Since we train the discriminator to return a value close to zero for a sample from the generator
(So that it could differ betweeen fake and real samples), when we train the generator we pass a sample
from the generator through the discriminator and update the parameters only of the generator.

"""

part3_q2 = r"""
**Your answer:**
1. No, because low generator loss can occurr due to the discriminator not being able to differ between
real and fake images.

2. That means the generator improves rapidly and the discriminator can't keep up.

"""

part3_q3 = r"""
**Your answer:**
The fact the the discriminator can't differ well enought between fake and real images creates noise at the results of GAN.
On the other hand, VAE has destination in mind, and it tries to converge to it.

The above causes a difference between the two.

We can see that during the training, VAE produces smooth images which overtime come closer to the state of the original picture.
And we can see the GAN produces images with noise, as expected.

"""

# ==============


