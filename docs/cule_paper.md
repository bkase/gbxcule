0
2
0
2

t
c
O
5

]

G
L
.
s
c
[

2
v
7
6
4
8
0
.
7
0
9
1
:
v
i
X
r
a

Accelerating Reinforcement Learning through GPU
Atari Emulation

Steven Dalton∗, Iuri Frosio∗, & Michael Garland
NVIDIA, USA
{sdalton,ifrosio,mgarland}@nvidia.com

Abstract

We introduce CuLE (CUDA Learning Environment), a CUDA port of the Atari
Learning Environment (ALE) which is used for the development of deep rein-
forcement algorithms. CuLE overcomes many limitations of existing CPU-based
emulators and scales naturally to multiple GPUs. It leverages GPU parallelization
to run thousands of games simultaneously and it renders frames directly on the
GPU, to avoid the bottleneck arising from the limited CPU-GPU communication
bandwidth. CuLE generates up to 155M frames per hour on a single GPU, a ﬁnding
previously achieved only through a cluster of CPUs. Beyond highlighting the differ-
ences between CPU and GPU emulators in the context of reinforcement learning,
we show how to leverage the high throughput of CuLE by effective batching of
the training data, and show accelerated convergence for A2C+V-trace. CuLE is
available at https://github.com/NVlabs/cule.

1

Introduction

Initially triggered by the success of DQN [12], research in Deep Reinforcement Learning (DRL) has
grown in popularity in the last years [9, 11, 12], leading to intelligent agents that solve non-trivial
tasks in complex environments. But DRL also soon proved to be a challenging computational
problem, especially if one wants to achieve peak performance on modern architectures.

Traditional DRL training focuses on CPU environments that execute a set of actions {at−1} at time
t − 1, and produce observable states {st} and rewards {rt}. These data are migrated to a Deep
Neural Network (DNN) on the GPU to eventually select the next action set, {at}, which is copied
back to the CPU. This sequence of operations deﬁnes the inference path, whose main aim is to
generate training data. A training buffer on the GPU stores the states generated on the inference
path; this is periodically used to update the DNN’s weights θ, according to the training rule of the
DRL algorithm (training path). A computationally efﬁcient DRL system should balance the data
generation and training processes, while minimizing the communication overhead along the inference
path and consuming, along the training path, as many data per second as possible [1, 2]. The solution
to this problem is however non-trivial and many DRL implementations do not leverage the full
computational potential of modern systems [16].

We focus our attention on the inference path and move from the traditional CPU implementation
of the Atari Learning Environment (ALE), a set of Atari 2600 games that emerged as an excellent
DRL benchmark [3, 10]. We show that signiﬁcant performance bottlenecks primarily stem from
CPU environment emulation: the CPU cannot run a large set of environments simultaneously, the
CPU-GPU communication bandwidth is limited, and the GPU is consequently underutilized. To
both investigate and mitigate these limitations we introduce CuLE (CUDA Learning Environment), a

∗Equal contribution

Preprint. Under review.

Table 1: Average training times, raw frames to reach convergence, FPS, and computational resources
of existing accelerated DRL schemes, compared to CuLE. Data from [7]; FPS are taken from the
corresponding papers, if available, and measured on the entire Atari suite for CuLE.

Algorithm

Ape-X DQN [7]
Rainbow [6]
Distributional (C51) [4]
A3C [11]
GA3C [1, 2]
Prioritized Dueling [19]
DQN [12]

Gorila DQN [13]
Unreal [8]

Stooke (A2C / DQN) [16]
IMPALA (A2C + V-Trace) [5]

CuLE (emulation only)
CuLE (inference only, A2C, single batch)
CuLE (training, A2C + V-trace, multiple batches)
CuLE (training, A2C + V-trace, multiple batches)*

N/A
N/A
1 hour
mins

*FPS measured on Asterix, Assault, MsPacman, and Pong.

Time

Frames

5 days
10 days
10 days
4 days
1 day
9.5 days
9.5 days

4 days
—

hours
mins/hours

FPS

50K
—
—
2K
8K
—
—

—
—

Resources

376 cores, 1 GPU
1 GPU
1 GPU
16 cores
16 cores, 1 GPU
1 GPU
1 GPU

> 100 cores
16 cores

35K
250K

40 CPUs, 8 GPUs (DGX-1)
100-200 cores, 1 GPU

41K-155K
39K-125K
26K-68K
142-187K

System I (1 GPU)
System I (1 GPU)
System I (1 GPU)
System III (4 GPUs)

22,800M
200M
200M
—
—
200M
200M

—
250M

200M
200M

N/A
N/A
200M
200M

Table 2: Systems used for experiments.

System

Intel CPU

NVIDIA GPU

I
II
III

12-core Core i7-5930K @3.50GHz
6-core Core i7-8086K @5GHz
20-core Core E5-2698 v4 @2.20GHz × 2

Titan V
Tesla V100
Tesla V100 × 8, NVLink

DRL library containing a CUDA enabled Atari 2600 emulator, that renders frames directly in the
GPU memory, avoids off-chip communication and achieves high GPU utilization by processing
thousands of environments in parallel—something so far achievable only through large and costly
distributed systems. Compared to the traditional CPU-based approach, GPU emulation improves
the utilization of the computational resources: CuLE on a single GPU generates more Frames
Per Second2 (FPS) on the inference path (between 39K and 125K, depending on the game, see
Table 1) compared to its CPU counterpart (between 12.5K and 19.8K). Beyond offering CuLE
(https://github.com/NVlabs/cule) as a tool for research in the DRL ﬁeld, our contribution can
be summarized as follow:

(1) We identify common computational bottlenecks in several DRL implementations that prevent
effective utilization of high throughput compute units and effective scaling to distributed systems.

(2) We introduce an effective batching strategy for large environment sets, that allows leveraging the
high throughput generated by CuLE to quickly reach convergence with A2C+V-trace [5], and show
effective scaling on multiple GPUs. This leads to the consumption of 26-68K FPS along the training
path on a single GPU, and up to 187K FPS using four GPUs, comparable (Table 1) to those achieved
by large clusters [17, 5].

(3) We analyze advantages and limitations of GPU emulation with CuLE in DRL, including the effect
of thread divergence and of the lower (compared to CPU) number of instructions per second per
thread, and hope that our insights may be of value for the development of efﬁcient DRL systems.

2 Related Work

The wall clock convergence time of a DRL algorithm is determined by two main factors: its sample
efﬁciency, and the computational efﬁciency of its implementation. Here we analyze the sample and
computational efﬁciency of different DRL algorithms, in connection with their implementation.

2Raw frames are reported here and in the rest of the paper, unless otherwise speciﬁed. These are the frames
that are actually emulated, but only 25% of them are rendered and used for training. Training frames are obtained
dividing the raw frames by 4—see also [5].

2

We ﬁrst divide DRL algorithms into policy gradient and Q-value methods, as in [16]. Q-learning
optimizes the error on the estimated action values as a proxy for policy optimization, whereas policy
gradient methods directly learn the relation between a state, st, and the optimal action, at; since at
each update they follow, by deﬁnition, the gradient with respect to the policy itself, they improve
the policy more efﬁciently. Policy methods are also considered more general, e.g. they can handle
continuous actions easily. Also the on- or off-policy nature of an algorithm profoundly affects both its
sample and computational efﬁciency. Off-policy methods allow re-using experiences multiple times,
which directly improves the sample efﬁciency; additionally, old data stored in the GPU memory
can be used to continuously update the DNN on the GPU, leading to high GPU utilization without
saturating the inference path. The replay buffer has a positive effect on the stability of learning
as well [12]. On-policy algorithms saturate the inference path more easily, as frames have to be
generated on-the-ﬂy using the current policy and moved from the CPU emulators to the GPU for
processing with the DNN. On-policy updates are generally effective but they are also more prone to
fall into local minima because of noise, especially if the number of environment is small — this is the
reason why on-policy algorithms largely beneﬁt (in term of stability) from a signiﬁcant increase of
the number of environments.

Policy gradient algorithms are often on-policy: their efﬁcient update strategy is counterbalanced by
the bottlenecks in the inference path and competition for the use of the GPU along the inference
and training path at the same time. Acceleration by scaling to a distributed system is possible but
inefﬁcient in this case: in IMPALA [5] a cluster with hundreds of CPU cores is needed to accelerate
A2C, while training is desynchronized to hide latency. As a consequence, the algorithm becomes
off-policy, and V-trace was introduced to deal with off-policy data (see details in the Appendix).
Acceleration on a DGX-1 has also been demonstrated for A2C and PPO, using large batch sizes
to increase the GPU occupancy, and asynchronous distributed models that hide latency, but require
periodic updates to remain synchronized [16] and overall achieves sublinear scaling with the number
of GPUs.

3 CUDA Learning Environment (CuLE)

In CuLE, we emulate the functioning of
many Atari consoles in parallel using the
CUDA programming model, where a se-
quential host program executes parallel pro-
grams, known as kernels, on a GPU. In a
trivial mapping of the Atari emulator to
CUDA, a single thread emulates both the
Atari CPU and TIA to execute the ROM
code, update the Atari CPU and TIA reg-
isters as well as the game state in the 128
bytes RAM, and eventually render the pix-
els in the output frame. However, the con-
trasting nature of the game code execution
and renderings tasks, the ﬁrst dominated
by reading from the RAM/ROM and writ-
ing tens of bytes to RAM, while the sec-
ond writes hundreds of pixels to the frame-
buffer, poses a serious issue in terms of per-
formance, such as thread divergence and
an imbalanced number of registers required
by the ﬁrst and second tasks. To mitigate
these issues, CuLE uses two CUDA ker-
nels: the ﬁrst one ﬁrst loads data from the
GPU global memory, where we store the
state of each emulated Atari processor, and
the 128 bytes RAM data containing the current state of the game; it also reads ROM instructions
from the constant GPU memory, executes them to update the Atari CPU and game states, and stores

Figure 1: Our CUDA-based Atari emulator uses an
Atari CPU kernel to emulate the functioning of the Atari
CPU and advance the game state, and a second TIA ker-
nel to emulate the TIA and render frames directly in
GPU memory. For episode resetting we generate and
store a cache of random initial states. Massive paral-
lelization on GPU threads allows the parallel emulation
of thousands of Atari games.

3

the updated Atari CPU and game states back into GPU global memory. It is important to notice that
this ﬁrst kernel does not execute the TIA instructions read from the ROM, but copies them into the
TIA instruction buffer in GPU global memory, which we implemented to decouple the execution
of the Atari CPU and TIA instructions in CuLE. The second CuLE kernel emulates the functioning
of the TIA processor: it ﬁrst reads the instructions stored in the TIA instruction buffer, execute
them to update the TIA registers, and renders the 160 × 210 output framebuffer in global GPU
memory. Despite this implementation requires going through the TIA instruction twice, it has several
advantages over the single-kernel trivial implementation. First of all, the requirements in terms of
registers per thread and the chance of having divergent code are different for the Atari CPU and TIA
kernels, and the use of different kernels achieves a better GPU usage. A second advantage that we
exploit is that not all frames are rendered in ALE: the input of the RL algorithm is the pixelwise
maximum between the last two frames in a sequence of four, so we can avoid calling the TIA kernel
when rendering of the screen is not needed. A last advantage, not exploited in our implementation
yet, is that the TIA kernel may be scheduled one the GPU with more than one thread per game, as
rendering of diverse rows on the screen is indeed a parallel operation - we leave this optimization for
future developments of CuLE.

To better ﬁt our execution model, our game reset strategy is also different from the one in the existing
CPU emulators, where 64 startup frames are executed at the end of each episode. Furthermore,
wrapper interfaces for RL, such as ALE, randomly execute an additional number of frames (up to
30) to introduce randomness into the initial state. This results into up to 94 frames to reset a game,
which may cause massive divergence between thousands of emulators executing in SIMD fashion on
a GPU. To address this issue, we generate and store a cache of random initial states (30 by default)
when a set of environments is initialized in CuLE. At the end of an episode, each emulator randomly
selects one of the cached states as a seed and copies it into the terminal emulator state.

Some of the choices made for the implementation of CuLE are informed by ease of debugging, like
associating one state update kernel to one environment, or need for ﬂexibility, like emulating the
Atari console instead of directly writing CUDA code for each Atari game. A 1-to-1 mapping between
threads and emulators is not the most computationally efﬁcient way to run Atari games on a GPU, but
it makes the implementation relatively straightforward and has the additional advantage that the same
emulator code can be executed on the CPU for debugging and benchmarking (in the following, we
will refer to this implementation as CuLECPU). Despite of this, the computational advantage provided
by CuLE over traditional CPU emulation remains signiﬁcant.

4 Experiments

Atari emulation We measure the FPS under different conditions: we get an upper bound on the
maximum achievable FPS in the emulation only case, when we emulate the environments and use a
random policy to select actions. In the inference only case, we measure the FPS along the inference
path: a policy DNN selects the actions, CPU-GPU data transfer occur for CPU emulators, while both
emulation and DNN inference run on the GPU when CuLE is used. This is the maximum throughput
achievable by off-policy algorithms, when data generation and consumption are decoupled and run
on different devices. In the training case, the entire DRL system is at work: emulation, inference,
and training may all run on the same GPU. This is representative of the case of on-policy algorithms,
but the FPS are also affected by the computational cost of the speciﬁc DRL update algorithm; in our
experiments we use a vanilla A2C [14], with N-step bootstrapping, and N = 5 as the baseline (for
details of A2C and off-policy correction with V-trace, see the Appendix).

Figs. 2(a)-2(b) show the FPS generated by OpenAI Gym, CuLECPU, and CuLE, on the entire set
of Atari games, as a function of the number of environments. In the emulation only case, CPU
emulation is more efﬁcient for a number of environments up to 128, when the GPU computational
power is not leveraged because of the low occupancy. For a larger number of environments, CuLE
signiﬁcantly overcomes OpenAI Gym, for which FPS are mostly stable for 64 environments or
more, indicating that the CPU is saturated: the ratio between the median FPS generated by CuLE
with 4096 environment (64K) and the peak FPS for OpenAI Gym (18K) is 3.56×. In the inference
only case there are two additional overheads: CPU-GPU communication (to transfer observations),
and DNN inference on the GPU. Consequently, CPU emulators achieve a lower FPS in inference

4

FPS

FPS per environment

(a) emulation only

(b) inference only

(c) emulation only

(d) inference only

Figure 2: FPS and FPS / environment on System I in Table 2, for OpenAI Gym [14], CuLECPU, and
CuLE, as a function of the number of environments, under different load conditions: emulation only,
and inference only. The boxplots indicate the minimum, 25th, 50th, 75th percentiles and maximum
FPS, for the entire set of 57 Atari games.

Table 3: Training FPS, DNN’s Update Per Second (UPS), time to reach a given score, and corre-
sponding number of training frames for four Atari games, A2C+V-trace, and different conﬁgurations
of the emulation engines, measured on System I in Table 2 (System III for the multi-GPU case). The
best metric in each row is in bold.
Engine

CuLE, 4 GPUs

CuLE, 1 GPU

OpenAI Gym

Game

Envs
Batches
N-steps
SPU

Training KFPS
UPS
Time [mins]
Training Mframes (for average score: 800)

Training KFPS
UPS
Time [mins]
Training Mframes (for average score: 1,000)

Training KFPS
UPS
Time [mins]
Training Mframes (for average score: 1,500)

Training KFPS
UPS
Time [mins]
Training Mframes (for average score: 18)

120
1
5
5

4.2
7.0
20.2
5.0

4.3
7.1
8.1
2.0

4.0
6.7
16.6
4.0

4.3
7.2
21.2
5.5

120
5
5
1

3.4
28.3
—
—

3.3
27.9
35.2
7.0

3.3
27.1
20.5
4.0

3.4
28.1
12.2
2.5

120
20
20
1

3.0
24.7
42.6
7.5

3.0
24.8
14.4
2.5

2.8
23.7
14.7
2.5

3.0
24.9
8.4
1.5

1200
20
20
1

4.9
4.1
44.2
13.0

4.9
4.1
27.1
8.0

4.8
4.0
12.4
3.5

4.8
4.0
8.7
2.5

1200
1
5
5

10.6
1.8
18.8
12.0

11.9
2.0
—
—

9.0
1.5
—
—

10.5
1.8
—
—

1200
5
5
1

11.5
9.6
9.4
6.5

12.5
10.4
14.0
10.5

9.6
8.0
6.9
4.0

11.2
9.3
5.9
4.0

1200
20
20
1

11.0
9.1
9.9
6.5

12.1
10.0
3.4
2.5

9.2
7.7
11.8
6.5

10.6
8.9
3.1
2.0

1200×4
20×4
20
1

42.7
8.9
7.9
18.0

46.6
9.7
2.5
7.0

35.5
7.4
2.4
3.0

41.7K
8.7
2.4
6.0

—

t
l
u
a
s
s
A

x
i
r
e
t
s
A

n
a
m
c
a
P
s
M

g
n
o
P

only when compared to emulation only; the effects of the overheads is more evident for a small
number of environments, while the FPS slightly increase with the number of environments without
reaching the emulation only FPS. CuLE’s FPS are also lower for inference only, because of the latency
introduced by DNN inference, but the FPS grow with the number of environments, suggesting that
the computational capability of the GPU is still far from being saturated.

Factors affecting the FPS Figs. 2(a)-2(b) shows that the throughput varies dramatically across
games: 4096 CuLECPU environments run at 27K FPS on Riverraid, but only 14K FPS for Boxing: a
1.93× difference, explained by the different complexity of the ROM code of each game. The ratio
between the maximum and minimum FPS is ampliﬁed in the case of GPU emulation: Riverraid runs
in emulation only at 155K FPS when emulated by CuLE and 4096 environments, while UpNDown
runs at 41K FPS —a 3.78× ratio.

To better highlight the impact of thread divergence on throughput, we measure the FPS for CuLE,
emulation only, 512 environments, and four games (Fig. 3). All the environments share the same
initial state, but random action selection leads them to diverge after some steps. Each environment
resets at the end of an episode. The FPS is maximum at the very beginning, when all the environments
are in similar states and the chance to execute the same instruction in all the threads is high. When
they move towards different states, code divergence negatively impacts the FPS, until it reaches an

5

(a) Asterix, GPU

(b) Pong, GPU

(c) Ms Pacman, GPU

(d) Assault, GPU

Figure 3: FPS as a function of the environment step, measured on System I in Table 2 for emulation
only on four Atari games, 512 environments, for CuLE; each panel also shows the number of resetting
environments. FPS is higher at the beginning, when all environments are in similar states and thread
divergence within warps is minimized; after some steps, correlation is lost, FPS decreases and
stabilizes. Minor oscillations in FPS are possibly associated to more or less computational demanding
phases in the emulation of the environments (e.g., when a goal is scored in Pong).

asymptotic value. This effect is present in all games and particularly evident for MsPacman in Fig. 3;
it is not present in CPU emulation (see Appendix). Although divergence can reduce FPS by 30%
in the worst case, this has to be compared with case of complete divergence within each thread and
for each instruction, which would yield 1/32 (cid:39) 3% of the peak performances. Minor oscillations of
the FPS are also visible especially for games with a repetitive pattern (e.g. Pong), where different
environments can be more or less correlated with a typical oscillation frequency.

Performances during training Fig. 4 compares
the FPS generated by different emulation engines on
a speciﬁc game (Assault)3, for different load condi-
tions, including the training case, and number of envi-
ronments. As expected, when the entire training path
is at work, the FPS decreases even further. However,
for CPU emulators, the difference between FPS in
the inference only and training cases decreases when
the number of environments increases, as the system
is bounded by the CPU computational capability and
CPU-GPU communication bandwidth. In the case of
the CPU scaling to multiple GPUs would be ineffec-
tive for on-policy algorithms, such GA3C [1, 2], or
sub-optimal, in the case of distributed systems [5, 16].
On the other hand, the difference between inference
only and training FPS increases with the number of
environments for CuLE, because of the additional
training overhead on the GPU. The potential speed-up provided by CuLE for vanilla A2C and Assault
in Fig. 4 is 2.53× for 1,024 environments, but the system is bounded by the GPU computational
power; as a consequence, better batching strategies that reduce the training computational overhead as
well as scaling to multiple GPUs are effective to further increase the speed-up ratio, as demonstrated
later in this Section.

Figure 4: FPS generated by different emula-
tion engines on System I in Table 2 for As-
sault, as a function of the number of environ-
ments, and different load conditions for A2C
with N-step bootstrapping, N = 5).

When data generation and training can be decoupled, like for off-policy algorithms, training can be
easily moved to a different GPU and the inference path can be used at maximum speed. The potential
speed-up provided by CuLE for off-policy algorithms is then given by the ratio between the inference
only median FPS for CuLE (56K) and CuLECPU (18K), which is 3.11× for 4,096 environments.
Furthermore, since the FPS remains ﬂat for CPU emulation, the advantage of CuLE ampliﬁes (for
both on- and off-policy methods) with the number of environments.

Frames per second per environment Fig. 2(c)-2(d) show the FPS / environment for different
emulation engines on System I, as a function of the number of environments. For 128 environments
or fewer, CPU emulators generate frames at a higher rate (compared to CuLE), because CPUs are
optimized for low latency, and execute a high number of instructions per second per thread. However,
the FPS / environment decrease with the number of environments, that have to share the same
CPU cores. Instead, the GPU architecture maximizes the throughput and has a lower number of

3Other games for which we observe a similar behavior are reported in the Appendix, for sake of space.

6

(a) Assault, 20M training frames

(b) Asterix, 20M training frames

(c) Ms-Pacman, 20M training frames

(d) Pong, 8M training frames

Figure 5: Average testing score and standard deviation on four Atari games as a function of the
training time, for A2C+V-trace, System III in Table 2, and different batching strategies (see also
Table 3). Training frames are double for the multi-GPU case (black line). Training performed on
CuLE or OpenAI Gym; testing performed on OpenAI Gym environments (see the last paragraph of
Section 4).

instructions per second per thread. As a consequence, the FPS / environment is smaller (compared
to CPU emulation) for a small number of environments, but they are almost constant up to 512
environments, and starts decreasing only after this point. In practice, CuLE environments provide
an efﬁcient means of training with a diverse set of data and collect large statistics about the rewards
experienced by numerous agents, and consequently lowering the variance of the value estimate. On
the other hand, samples are collected less efﬁciently in the temporal domain, which may worsen the
bias on the estimate of the value function by preventing the use of large N in N-step bootstrapping.
The last paragraph of this Section shows how to leverage the high throughput generated by CuLE,
considering these peculiarities.

Memory limitations Emulating a massively large number of environments can be problematic
considering the relatively small amount of GPU DRAM. Our PyTorch [15] implementation of A2C
requires each environment to store 4 84x84 frames, plus some additional variables for the emulator
state. For 16K environments this translates into 1GB of memory, but the primary issue is the combined
memory pressure to store the DNN with 4M parameters and the meta-data during training, including
the past states: training with 16K environments easily exhausts the DRAM on a single GPU (while
training on multiple GPUs increases the amount of available RAM). Since we did not implement
any data compression scheme as in [7], we constrain our training conﬁguration to fewer than 5K
environments, but peak performance in terms of FPS would be achieved for a higher number of
environments - this is left as a possible future improvement.

A2C We analyze in detail the case of A2C with CuLE on a single GPU. As a baseline, we consider
vanilla A2C, using 120 OpenAI Gym CPU environments that send training data to the GPU to update
the DNN every N = 5 steps. This conﬁguration takes, on average, 21.2 minutes (and 5.5M training
frames) to reach a score of 18 for Pong and 16.6 minutes (4.0M training frames) for a score of
1,500 on Ms-Pacman (Fig. 5, red line; ﬁrst column of Table 3). CuLE with 1,200 environments
generates approximately 2.5× more FPS compared to OpenAI Gym, but this alone is not sufﬁcient
to improve the convergence speed (blue line, Fig. 5). CuLE generates larger batches but, because
FPS / environment is lower when compared to CPU emulation, fewer Updates Per Second (UPS) are
performed for training the DNN (Table 3), which is detrimental for learning.

A2C+V-trace and batching strategy To better leverage CuLE, and similar in spirit to the approach
in IMPALA [5], we employ a different batching strategy on the GPU, but training data are read in
batches to update the DNN every Steps Per Update (SPU) steps. This batching strategy signiﬁcantly
increases the DNN’s UPS at the cost of a slight decrease in FPS (second columns of OpenAI Gym

7

and CuLE in Table 3), due to the fact that the GPU has to dedicate more time to training. Furthermore,
as only the most recent data in a batch are generated with the current policy, we use V-trace [5] for
off-policy correction. The net result is an increase of the overall training time when 120 OpenAI Gym
CPU environments are used, as this conﬁguration pays for the increased training and communication
overhead, while the smaller batch size increases the variance in the estimate of the value function and
leads to noisy DNN updates (second column in Table 3, orange lines in Fig. 5). Since CuLE does
not suffer from the same computational bottlenecks, and at the same time beneﬁts from the variance
reduction associated with the large number (1,200) of environments, using the same batching strategy
with CuLE reduces the time to reach a score of 18 for Pong and 1,500 for Pacman respectively to
5.9 and 6.9 minutes. The number of frames required to reach the same score is sometimes higher
for CuLE (Table 3), which can lead to less sample efﬁcient implementation when compared to the
baseline, but the higher FPS largely compensates for this. Extending the batch size in the temporal
dimension (N-steps bootstrapping, N = 20) increases the GPU computational load and reduces both
the FPS and UPS, but it also reduces the bias in the estimate of the value function, making each DNN
update more effective, and leads to an overall decrease of the wall clock training time, the fastest
convergence being achieved by CuLE with 1,200 environments. Using OpenAI Gym with the same
conﬁguration results in a longer training time, because of the lower FPS generated by CPU emulation.

Generalization for different systems Table 4 reports the FPS for the implementations of vanilla
DQN, A2C, and PPO, on System I and II in Table 2. The speed-up in terms of FPS provided by
CuLE is consistent across different systems, different algorithms, and larger in percentage for a
large number of environments. Different DRL algorithms achieve different FPS depending on the
complexity and frequency of the training step on the GPU.

Table 4: Average FPS and min/max GPU utilization during training for Pong with different algorithms
and using different emulation engines on different systems (see Table 2); CuLE consistently leads to
higher FPS and GPU utilization.

Algorithm

Emulation engine

FPS [GPU utilization %]

System I [256 envs]

System I [1024 envs]

System II [256 envs]

System II [1024 envs]

DQN

A2C

PPO

OpenAI
CuLECPU
CuLE

OpenAI
CuLECPU
CuLE

OpenAI
CuLECPU
CuLE

6.4K [15-42%]
7.2K [16-43%]
14.4K [16-99%]

12.8K [2-15%]
10.4K [2-15%]
19.6K [97-98%]

12K [3-99%]
10K [2-99%]
14K [95-99%]

8.4K [0-69%]
8.6K [0-72%]
25.6K [17-99%]

15.2K [0-43%]
14.2K [0-43%]
51K [98-100%]

10.6K [0-96%]
10.2K [0-96%]
36K [95-100%]

10.8K [26-32%]
6.8K [17-25%]
11.2K [48-62%]

24.4K [5-23%]
12.8K [1-18%]
23.2K [97-98%]

16.0K [4-33%]
9.2K [2-28%]
14.4K [43-98%]

21.2K [28-75%]
20.8K [8-21%]
33.2K [57-77%]

30.4K [3-45%]
25.6K [3-47%]
48.0K [98-99%]

19.2K [4-62%]
18.4K [3-61%]
28.0K [45-99%]

5 Conclusion

As already shown by others in the case of DRL on distributed system, our experiments show that
proper batching coupled with a slight off-policy gradient policy algorithm can signiﬁcantly accelerate
the wall clock convergence time; CuLE has the additional advantage of allowing effective scaling
of this implementation to a system with multiple GPUs. CuLE effectively allows increasing the
number of parallel environments but, because of the low number of instructions per second per
thread on the GPU, training data can be narrow in the time direction. This can be problematic for
problems with sparse temporal rewards, but rather than considering this as a pure limitation of CuLE,
we believe that this peculiarity opens the door to new interesting research questions, like active
sampling of important states [6, 19] that can then be effectively analyzed on a large number of parallel
environments with CuLE. CuLE also hits a new obstacle, which is the limited amount of DRAM
available on the GPU; studying new compression schemes, like the one proposed in [6], as well as
training methods with smaller memory footprints may help extend the utility of CuLE to even larger
environment counts, and design better GPU-based simulator for RL in the future. Since these are
only two of the possible research directions for which CuLE is an effective investigation instrument,
CuLE comes with a python interface that allows easy experimentation and is freely available to any
researcher at https://github.com/NVlabs/cule.

8

6

Impact Statement

As interest in deep reinforcement learning has grown so has the computational requirements for
researchers in this ﬁeld. However, the reliance of DRL on the CPU, especially for environment simu-
lation/emulation, severely limits the utilization of the computational resources typically accessible to
DL researchers, speciﬁcally GPUs. Though Atari is a specialized DRL environment, it is arguably one
of the most studied in recent times and provides access to several training environments with various
levels of difﬁculty. The development and testing of DRL using Atari games remains a relevant and
signiﬁcant step toward more efﬁcient algorithms. There are two impact points for CuLE: 1) Provide
access to an accelerated training environment to researchers with limited computational capabilities.
2) Facilitate research in novel directions that explore thousands of agents without requiring access
to a distributed system with hundreds of CPU cores. Although leaving RL environments "as-is"
on CPUs and parallelizing across multiple nodes is indeed the shortest path to make progress is it
also inherently inefﬁcient, in terms of the resource utilization on the local machine, and expensive,
since it requires access to a large number of distributed machines. The more efﬁcient use of the
computational resources could also lead to a smaller carbon footprint.

7 Appendix

7.1 Reinforcement Learning, A2C and V-trace

Reinforcement learning In RL, an agent observes a state st at time t and follows a policy π = π(st)
to select an action at; the agent also receives a scalar reward rt from the environment. The goal of
RL is to optimize π such that the sum of the expected rewards is maximized.

In model-free policy gradient methods π(at|st; θ) is the output of a policy DNN with weights θ, and
represents the probability of selecting action at in the state st. Updates to the DNN are generally
aligned in the direction of the gradient of E[Rt], where Rt = (cid:80)∞
i=0 γirt+i is the discounted reward
from time t, with discount factor γ ∈ (0, 1] (see also REINFORCE [20]) The vanilla implementation
updates θ along ∇θ log π(at|st; θ)Rt, which is an unbiased estimator of ∇θE[Rt]. The training
procedure can be improved by reducing the variance of the estimator by subtracting a learned baseline
bt(st) and using the gradient ∇θ log π(at|st; θ)[Rt − bt(st)]. One common baseline is the value
function V π(st) = E[Rt|st], which is the expected return for the policy π starting from st. The
policy π and the baseline bt can be viewed as actor and critic in an actor-critic architecture [18].

A2C A2C [14] is the synchronous version of A3C [11], a successful actor-critic algorithm, where a
single DNN outputs a softmax layer for the policy π (at|st; θ), and a linear layer for V (st; θ). In
A2C, multiple agents perform simultaneous steps on a set of parallel environments, while the DNN is
updated every tmax actions using the experiences collected by all the agents in the last tmax steps.
This means that the variance of the critic V (st; θ) is reduced (at the price of an increase in the bias)
by N -step bootstrapping, with N = tmax. The cost function for the policy is then:

log π (at|st; θ)

(cid:105)
(cid:104) ˜Rt − V (st; θt)

+ βH [π (st; θ)] ,

(1)

where θt are the DNN weights θ at time t, ˜Rt = (cid:80)k−1
i=0 γirt+i + γkV (st+k; θt) is the bootstrapped
discounted reward from t to t + k and k is upper-bounded by tmax, and H [π (st; θ)] is an entropy
term that favors exploration, weighted by the hyper-parameter β. The cost function for the estimated
value function is:
(cid:105)2
(cid:104) ˜Rt − V (st; θ)
which uses, again, the bootstrapped estimate ˜Rt. Gradients ∇θ are collected from both of the cost
functions; standard optimizers, such as Adam or RMSProp, can be used for optimization.

(2)

,

V-trace
In the case where there is a large number of environments, such as in CuLE or IMPALA [5],
the synchronous nature of A2C become detrimental for the learning speed, as one should wait for all
the environments to complete tmax steps before computing a single DNN update. Faster convergence
is achieved (both in our paper and in [5]) by desynchronizing data generation and DNN updates,

9

which in practice means sampling a subset of experiences generated by the agents, and updating the
policy using an approximate gradient, which makes the algorithm slightly off-policy.

To correct for the off-policy nature of the data, that may lead to inefﬁciency or, even worse, instabilities,
in the training process, V-trace is introduced in [5]. In summary, the aim of off-policy correction is to
give less weight to experiences that have been generated with policy µ, called the behaviour policy,
when it differs from the target policy, π; for a more principled explanation we remand the curios
reader to [5].

For a set of experiences collected from time t = t0 to time t = t0 + N following some policy µ, the
N -steps V-trace target for V (st0 ; θ) is deﬁned as:

vt0 = V (st0; θ) + (cid:80)t0+N −1
δtV = ρt
ρt = min (cid:0)¯ρ,

t=t0
(cid:0)rt + γV (st+1; θ) − V (st; θ)(cid:1)
(cid:1)

γt−t0

(cid:16) (cid:81)t−1
i=t0

π(at|st)
µ(at|st)
π(ai|si)
µ(ai|si)

(cid:1);

ci = min (cid:0)¯c,

(cid:17)

ci

δtV ,

(3)

(4)

(5)

(6)

ρt and ci are truncated importance sampling (IS) weights, and (cid:81)t−1
i=t0 ci = 1 for s = t, and ¯ρ ≥ ¯c.
Notice that, when we adopt the proposed multi-batching strategy, there are multiple behaviour policies
µ that have been followed to generate the training data — e.g., N different policies are used when
SPU=1 in Fig. 5. Eqs. 5-6 do not need to be changed in this case, but we have to store all the µ(ai|si)
in the training buffer to compute the, V-trace corrected, DNN update. In our implementation, we
compute the V-trace update recursively as:

vt = V (st; θ) + δtV + γcs

(cid:0)vt+1 − V (st+1; θ)(cid:1).

At training time t, we update θ with respect to the value output, vs, given by:

(cid:0)vt − V (st; θ)(cid:1)∇θV (st; θ),

whereas the policy gradient is given by:

ρt∇ω log πω(as|st)(cid:0)rt + γvt+1 − V (st; θ)(cid:1).
(9)
An entropy regularization term that favors exploration and prevents premature convergence (as in
Eq. 1) is also added.

7.2 Thread divergence is not present in the case of CPU emulation

We show here that thread divergence, that affects GPU-based emulation (see Fig. 3), does not affect
CPU-based emulation. Fig. 6 shows the FPS on four Atari games where all the environments share
the same initial state. In constrast with GPU emulation, the CPU FPS do not peak at the beginning of
the emulation period, where many environments are correlated.

7.3 Performance during training - other games

For sake of space, we only report (Fig. 7) the FPS measured on system I in Table 2 for three additional
games, as a function of different load conditions and number of environments.

7.4 Correctness of the implementation

To demonstrate the correctness of our implementation, and thus that policies learned with CuLE
generalize to the same game emulated by OpenAI Gym, we report in Fig. 8 the average scores
achieved in testing, while training an agent with with A2C+V-trace and CuLE. The testing scores
measured on CuLECPU and OpenAI Gym environments do not show any relevant statistical difference,
even for the case of Ms-Pacman, where the variability of the scores is higher because of the nature of
the game.

10

(7)

(8)

(a) Asterix, CPU

(b) Pong, CPU

(c) Ms Pacman, CPU

(d) Assault, CPU

Figure 6: FPS as a function of the environment step, measured on System I in Table 2 for emulation
only on four Atari games, 512 environments, for CuLECPU; each panel also shows the number of
resetting environments. A peak in the FPS at the beginning of the emulation period, as in the case of
GPU emulation in Fig. 3, is not visible in this case.

(a) Pong

(b) MsPacman

(c) Asterix

Figure 7: FPS generated by different emulation engines on System I in Table 2 for different Atari
games, as a function of the number of environments, and different load conditions (the main A2C [14]
loop is run here, with N-step bootstrapping, N = 5.

(a) Assault

(b) Asterix

(c) Ms-Pacman

(d) Pong

Figure 8: Average testing scores measured on 10 CuLECPU and OpenAI Gym environments, while
training with A2C+V-trace and CuLE, as a function of the training frames; 250 environments are
used for Ms-Pacman, given its higher variability. The shaded area represents 2 standard deviations.

11

References

[1] Mohammad Babaeizadeh, Iuri Frosio, Stephen Tyree, Jason Clemons, and Jan Kautz. GA3C:

gpu-based A3C for deep reinforcement learning. CoRR, abs/1611.06256, 2016.

[2] Mohammad Babaeizadeh, Iuri Frosio, Stephen Tyree, Jason Clemons, and Jan Kautz. Rein-
forcement learning through asynchronous advantage actor-critic on a gpu. In ICLR, 2017.
[3] M. G. Bellemare, Y. Naddaf, J. Veness, and M. Bowling. The arcade learning environment: An
evaluation platform for general agents. Journal of Artiﬁcial Intelligence Research, 47:253–279,
jun 2013.

[4] Marc G. Bellemare, Will Dabney, and Rémi Munos. A distributional perspective on reinforce-

ment learning. CoRR, abs/1707.06887, 2017.

[5] Lasse Espeholt, Hubert Soyer, Rémi Munos, Karen Simonyan, Volodymyr Mnih, Tom Ward,
Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, and Koray Kavukcuoglu.
IMPALA: scalable distributed deep-rl with importance weighted actor-learner architectures.
CoRR, abs/1802.01561, 2018.

[6] Matteo Hessel, Joseph Modayil, Hado van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney,
Daniel Horgan, Bilal Piot, Mohammad Gheshlaghi Azar, and David Silver. Rainbow: Combining
improvements in deep reinforcement learning. CoRR, abs/1710.02298, 2017.

[7] Dan Horgan, John Quan, David Budden, Gabriel Barth-Maron, Matteo Hessel, Hado van
Hasselt, and David Silver. Distributed prioritized experience replay. CoRR, abs/1803.00933,
2018.

[8] Max Jaderberg, Volodymyr Mnih, Wojciech Marian Czarnecki, Tom Schaul, Joel Z Leibo,
David Silver, and Koray Kavukcuoglu. Reinforcement learning with unsupervised auxiliary
tasks. arXiv preprint arXiv:1611.05397, 2016.

[9] Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval
Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning.
CoRR, abs/1509.02971, 2015.

[10] Marlos C. Machado, Marc G. Bellemare, Erik Talvitie, Joel Veness, Matthew J. Hausknecht,
and Michael Bowling. Revisiting the arcade learning environment: Evaluation protocols and
open problems for general agents. CoRR, abs/1709.06009, 2017.

[11] Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap,
Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforce-
ment learning. In Maria Florina Balcan and Kilian Q. Weinberger, editors, Proceedings of The
33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine
Learning Research, pages 1928–1937, New York, New York, USA, 20–22 Jun 2016. PMLR.

[12] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G.
Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Pe-
tersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan
Wierstra, Shane Legg, and Demis Hassabis. Human-level control through deep reinforcement
learning. Nature, 518(7540):529–533, February 2015.

[13] Arun Nair, Praveen Srinivasan, Sam Blackwell, Cagdas Alcicek, Rory Fearon, Alessandro De
Maria, Vedavyas Panneershelvam, Mustafa Suleyman, Charles Beattie, Stig Petersen, Shane
Legg, Volodymyr Mnih, Koray Kavukcuoglu, and David Silver. Massively parallel methods for
deep reinforcement learning. CoRR, abs/1507.04296, 2015.

[14] OpenAI. Openai baselines: Acktr & a2c, 2017.
[15] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito,
Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in
pytorch. In NIPS-W, 2017.

[16] Adam Stooke and Pieter Abbeel. Accelerated methods for deep reinforcement learning. CoRR,

abs/1803.02811, 2018.

[17] Adam Stooke and Pieter Abbeel. Accelerated methods for deep reinforcement learning. CoRR,

abs/1803.02811, 2018.

[18] Richard S. Sutton and Andrew G. Barto. Introduction to Reinforcement Learning. MIT Press,

Cambridge, MA, USA, 1st edition, 1998.

12

[19] Ziyu Wang, Nando de Freitas, and Marc Lanctot. Dueling network architectures for deep

reinforcement learning. CoRR, abs/1511.06581, 2015.

[20] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforce-

ment learning. Machine learning, 8(3-4):229–256, 1992.

13


