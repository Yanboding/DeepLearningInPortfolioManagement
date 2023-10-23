# Deep Learning in Portfolio Management

* This project aims to apply the latest reinforcement learning method to solve Protfolio Management problem.

## Problem Setup
At each timestamp t, the MDP problem is formulated as follows:
* State: a tensor with shape (f, n, m) that represents f features(e.g. close, low and high prices) of m preselected assets in passed n timestamps counting from the current timestamp t.
* Action: portfolio weights on m assets. The weights should sum to 1.
* Reward: the log return of time t.
## Usage

## Algorithm implemented
- [x] [DDPG](https://arxiv.org/abs/1509.02971)
- [x] [BAC](https://arxiv.org/abs/2006.15637)
- [ ] TD3
- [ ] A2C
## Reference
* [A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem](https://arxiv.org/pdf/1706.10059.pdf)

