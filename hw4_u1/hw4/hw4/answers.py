r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(batch_size=32,
              gamma=0.98,
              beta=0.25,
              learn_rate=7*1e-4,
              eps=1e-6,
              )
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    # ========================
    return hp


def part1_aac_hyperparams():
    hp = dict(batch_size=1,
              gamma=0.97,
              beta=0.5,
              delta=1.0,
              learn_rate=1e-6,
              eps=1e-8,
              )
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    # ====== YOUR CODE: ======
    #raise NotImplementedError()
    # ========================
    return hp


part1_q1 = r"""
**Your answer:**
We want our training to converge into the best advantage of an action (a measure of how good is it for us to take the action in a given state).
The advantage is a product of substracting the baseline from the reward, and the basline compensates the variance since it's not depended on the state,
and thus so does the variance of the advantage is being reduced.

An example of where this might help us is a case where sometimes similar states produce different rewards.
When we stack a high amount of experience we will eventually average over the different rewards, due to the low variance and converge to a good advantege.




"""


part1_q2 = r"""
**Your answer:**
$V_{\pi}(s)$ is the expectation of $q_{\pi}(s,a)$ over all actions.
Each q-value is an expectation over possible tragjectories, starting with a certain action from the initial state.

It's not possible to compute all the possible tragjectories from all states and all actions, we use the q-values as a sample.
That way, we can approximate a close enough function with $V_{\pi}$ while we run regression on each batch of episodes.
"""


part1_q3 = r"""
**Your answer:**



First experiment analysis:

* The vanilla wants to minimize the loss. It's seems noisy since it's extremely reliant on the rewards.
This is why, in order to reach zero loss, it reduces the rewards, which leads to the behaviour we in the mean reward graph,
where the vanilla reaches a plateu.

* The rise of the entropy suggests that we take more deliberate actions.
When the entropy is close to zero, it means the distribution os un-even.

* The baseline graph looks like a constant with jitter becasue we estimate the average for every batch, and the estimation is appromiately close to the average by definition.

The baseline mean rewrad shows the best result becusae it is affected by the best advantege.
The player will pick the reward that gives him the best improvement w.r.t the state.



Comparison with AAC:
* The loos_p and loss_e of AAC shows high volatility because we reduced the batch size.
Now the platyer gets to experience more distributions.

* The AAC mean reward graph is better according to Question 2.

"""
