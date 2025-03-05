# Distributed Machine Learning LAB

This repo hosts all relevant files about the distributed ML Lab in Neuchatel.

## Building LLMs

For this lab we will use a library to help us build Large Language Models and train them easily. It should already be installed if you followed the overall setup for labs.

You can find the libary here: [https://github.com/NikolayBlagoev/simplellm/tree/main](https://github.com/NikolayBlagoev/simplellm/tree/main)

Let us set up a simple model and dataset which we will use throughout this lab (the file can be found in [primer/intro.py](primer/intro.py))
```python
from simplellm.llama import CausalLLama, LLama # get our models
from simplellm.tokenizers import SPTokenizer # get our tokenizer
from simplellm.dataloaders import TinyStories # get our dataset
from simplellm.losses import causalLLMLoss # our loss
from torch.optim import Adam
import torch.nn.functional as F
dmodel = 288
num_heads = 6
n_layers = 6
seq_l = 256
batch_size = 3
device = "cuda"

# make the tokenizer
tokenizer = SPTokenizer()
# make the model
net = LLama(CausalLLama,tokenizer.vocab_size,dmodel=dmodel,num_heads=num_heads,
                device=device, n_layers=n_layers, ctx_size=seq_l,padding_idx=tokenizer.pad_id)
ds = TinyStories(tokenizer,batch_size=batch_size, seq_l=seq_l)
# we can iterate the dataset with:
iter_ds = iter(ds)
optim = Adam(net.parameters(),lr=8e-4)
for itr in range(5_000):
    optim.zero_grad()
    x = next(iter_ds)
    x = x.to(device)
    target = x.clone().detach()
    x = net(x)
    loss = causalLLMLoss(x,target,tokenizer.vocab_size)
    # log the loss:
    print(itr,loss.item())
    loss.backward()
    optim.step()
```

Throughout this lab we will be expanding this implementation

## Primer on Batched Stochastic Gradient Descent

As we all remember, when performing SGD we do the following steps:

1. Run a forward pass through our model for a given sample $x$
2. Compute the loss $L(x,y)$ given some target $y$
3. Compute the gradient per weight $\frac{\delta y}{\delta W}$
4. Update the weights with $W = W - \lambda \frac{\delta y}{\delta W}$ where $\lambda$ is the learning rate

When we use a batch of size larger than 1 the same process is performed, except that the gradient upate is the average across all the gradients for the different samples. This gives less noisy updates over time. The larger the batch is usually better (though there are things such as TOO LARGE of a batch, but in the context of LLMs this is hardly ever the case).

## From Federated to Data Parallel

Previously we learnt about federated learning as a means of preserving the privacy of each participant. Data Parallelism (DP) is pretty much the same as Federated Learning, except the concern is about throughput and not privacy, thus making our lives a lot easier when implementing DP solutions.

As we mentioned the larger your batch, usually the better. But with large models, the GPU memory is usually your bottleneck. So you quite often can't even do a batch of more than 2 samples. This however would make our gradient updates too noisy and lead to poor convergence.

A very obvious solution is - if one GPU is not enough, let's use multiple. This is usually how problems in distributed systems are solved. Similar to federated learning, we will have multiple nodes (or workers or devices if you prefer) perform training locally on some subset of shared data and then at the end of an iteration, average their gradients, before performing an update step. It is easy to see how this is equivalent to increasing your batch size in batched SGD. In the context of DP, each device is said to train a mini-batch, with the global size of the batch being the summation of all the mini-batches. So, if you have 4 devices, each training with a mini-batch of size 4 samples, your global batch size is 16 and is equivalent to doing batched SGD with batch size of 16.

In DP all workers perform their local iteration in parallel, thus you can have a speed up with the number of devices. However, you pay with an increase in communication, as each device will need to communicate with every other their updates during the aggregation phase. There is extensive literature on optimising this phase, as it can be incredibly costly.

Let us see how to implement this in torch. First we need to set up our group of workers that will be communicating with each other. PyTorch provides an abstraction for this for you with their [torch.distributed](https://pytorch.org/docs/stable/distributed.html) package. They support three backends - gloo, nccl, and mpi. Use gloo if you want to do cpu to cpu communication (and mpi if gloo is failing). Use nccl if you are doing gpu to gpu communication. Fair warning, nccl does not allow a gpu to communication with itself, so if you are hosting multiple nodes on the same gpu device, nccl will not work. Therefore for our demos we will be using the gloo backend. In practice, however, you should use nccl as it has significantly faster throughput.

When initialising the distributed communication you need to provide 5 things to torch - the desired backend, the world size (how many devices will participate), the rank of the current device (unique per device in the range of $[0,world size]$), the master address and port. The last two are used when devices need to discover each other. One device (the master) will need to bind to the given address and listen for incoming messages. To do it in code is quite simple:

```python
import torch.distributed as dist
import os
from sys import argv
rank = int(argv[1])
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
dist.init_process_group("gloo", rank=rank, world_size=3)
```

The process will block until 3 different processes reach the init_process_group statement.

The next thing we need to modify is at the weight update step. We need to synchronise across all devices. Thus we will modify that part of our primer to:

```python
dist.barrier() # wait for everyone
tmp = []
for param in net.parameters():
    if param.grad == None:
        tmp.append(torch.zeros_like(param).view(-1))
        continue
    tmp.append(param.grad.view(-1))
    param.grad = None
prev_grad = torch.cat(tmp).to("cpu")
dist.all_reduce(prev_grad, op = dist.ReduceOp.SUM)
```

The last line runs an [all reduce](https://en.wikipedia.org/wiki/Collective_operation#Reduce), which will make the `prev_grad` tensor the same across all devices. What we would like is to reduce to the average across all devices, however gloo supports only sum. So we would need to add an extra operation to average the prev_grad tensor.

The working full file is available in [intro_DP_GA.py](DP/gradient_aggr/intro_DP_GA.py):

```python
from simplellm.llama import CausalLLama, LLama # get our models
from simplellm.tokenizers import SPTokenizer # get our tokenizer
from simplellm.dataloaders import TinyStories # get our dataset
from simplellm.losses import causalLLMLoss # our loss
from torch.optim import SGD, Adam
import torch.nn.functional as F
import torch
import torch.distributed as dist
import os
from sys import argv
rank = int(argv[1])
os.environ["MASTER_ADDR"] = "localhost"
world_size = 3
os.environ["MASTER_PORT"] = "29500"
dist.init_process_group("gloo", rank=rank, world_size=world_size)
torch.manual_seed(0)
dmodel = 288
num_heads = 6
n_layers = 6
seq_l = 256
batch_size = 1
device = "cuda"

# make the tokenizer
tokenizer = SPTokenizer()
# make the model
net = LLama(CausalLLama,tokenizer.vocab_size,dmodel=dmodel,num_heads=num_heads,
                device=device, n_layers=n_layers, ctx_size=seq_l,padding_idx=tokenizer.pad_id)
ds = TinyStories(tokenizer,batch_size=batch_size, seq_l=seq_l,skip=rank*3000) # skip so we can have different things
# we can iterate the dataset with:
iter_ds = iter(ds)

optim = Adam(net.parameters(),lr=8e-4)

sizes = []
len_sizes = []
for param in net.parameters():
    sizes.append(param.shape)
    len_sizes.append(len(param.view(-1)))

for itr in range(5_000):
    optim.zero_grad()
    x = next(iter_ds)
    target = x.clone().detach()
    x = x.to(device)

    x = net(x)
    loss = causalLLMLoss(x,target,tokenizer.vocab_size)
    # log the loss:
    print(itr,loss.item())
    loss.backward()

    dist.barrier() # wait for everyone

    tmp = []
    for param in net.parameters():
        if param.grad == None:
            tmp.append(torch.zeros_like(param,device="cpu").view(-1))
            continue
        tmp.append(param.grad.view(-1))
        param.grad = None
    prev_grad = torch.cat(tmp).to("cpu")
    dist.all_reduce(prev_grad, op = dist.ReduceOp.SUM)
    tmp = torch.split(prev_grad, len_sizes)
    for i, param in enumerate(net.parameters()):
        param.grad = tmp[i].view(sizes[i]).to(device)/world_size # average
    optim.step()
    torch.cuda.empty_cache()
```


As we learnt in Federated Learning, it is also possible to synchronise on model weights, rather than on gradients. The modification to the above code to accomodate is simple and can be found in [intro_DP_WA.py](DP/weight_aggr/intro_DP_WA.py)

Torch provides a nice abstraction to our own implementation for [Data Parallel Training](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel). From personal experience, the timing between the two is almost identical. It is also a bit more limiting as it only synchronises gradients and not weights and has a few quirks to it.

### What to remember

Data Parallelism is useful when you want to increase your batch size by training independently on multiple devices. It can also provide a speed up in computation by parallelising across devices, at the cost of greater overhead in communication. In practical applications, Data Parallel communication is the bottleneck of your training.

### A small note on optimizers

We have seen Weight Aggregation and Gradient Aggregation as methods for synchronising in Data Parallel training. Both have their strengths, but a major one of Gradient Aggregation is that since the gradients will be identical across all devices, we can use better optimizers like Adam, without the need to synchronise the states of the optimizer. This is trivial to see, since the optimizer state is updated deterministically and depends only on the gradient of each parameter. With weight aggregation this is harder to achieve.


## Model Parallelism

We learnt about data parallelism. But what if we cannot fit our model on a single device, because it is just **SO** big? We could try offloading, but that is for people who don't have access to multiple devices. We are in distributed systems. We have as many devices as we ~~want~~ can afford. So let us think of a smart way we can make use of our devices to run a model that doesn't fit on one of them.

One thing we can exploit about traditional models is that they are very modular - groups of layers performing similar operations. This is especially true for Large Language Models, where the same sub-architecture of a transformer block is repeated a number of times throughout the model. Each block performs roughly the same operations, communicates the same size of data, and has similar memory requirements. Thus, what we can do, is separate groups of these similar modules across different device and communicate the activations between them. Each device has a sequence of layers ($L_i, L_{i+1}.., L_{i+\delta} \in L$) from the model and no two devices have the same layers (the divisions are non-overlapping). Each device thus constitutes a stage $S$. Devices in one stage $S_i$ receive the activations from the previous $S_{i-1}$ and send them to the next one $S_{i+1}$. Thus all stages form a pipeline. From an outside perspective, a pipeline behaves as one single centralised model - nothing about the actual training changes. As we will see later, we can have multiple devices in a stage, each device communicating with a different device from the previous and the next stage to form multiple pipelines.

The communication flow is fairly straight forward - the first stage gets the data ```x = next(iter_ds)```, performs the embedding ```x = embed(x)```, runs the activations through the local layers it has ```x = layers(x)```, and sends the last output to the next device ```send(x,rank+1)```. Each subsequent device receives some activations ```receive(x,rank-1)``` (which will populate the x tensor with the received one, thus they need to have matching shape), processes them internally, and sends them to the next device. Once the last device has ran the activations through its local sub-section of the model, it performs the de-embedding, and computes the loss. This constitutes a forward pass. Then it runs a local backwards pass, and sends the gradient of the input it received to the previous device ```send(x.grad,rank-1)```. This continues from device to device in reverse order of the forward pass, until we reach again the first device. All devices then perform an optimisation step and the next iteration begins.


The implementation of the above is sligthly more annoying, due to the possibility of a deadlock that we will run into later. In the single forward, single backward case that we described above, we can utilise the distributed pytorch functions ```send``` and ```recv```, to achieve the following:
````python
from simplellm.llama import LLamaFirstStage, LLamaLastStage, LLamaStage # get our models
from simplellm.tokenizers import SPTokenizer # get our tokenizer
from simplellm.dataloaders import TinyStories # get our dataset
from simplellm.losses import causalLLMLoss # our loss
from torch.optim import SGD, Adam
import torch.nn.functional as F
import torch
import torch.distributed as dist
import os
from sys import argv
rank = int(argv[1])
os.environ["MASTER_ADDR"] = "localhost"
world_size = 3
os.environ["MASTER_PORT"] = "29500"
dist.init_process_group("gloo", rank=rank, world_size=world_size)
torch.manual_seed(0)
dmodel = 288
num_heads = 6
n_layers = 6 // world_size
seq_l = 256
batch_size = 3
device = "cuda"

# make the tokenizer

# make the model
if rank == 0:
    tokenizer = SPTokenizer()
    net = LLamaFirstStage(tokenizer.vocab_size,dmodel=dmodel,num_heads=num_heads,
                device=device, n_layers=n_layers, ctx_size=seq_l)
    ds = TinyStories(tokenizer,batch_size=batch_size, seq_l=seq_l) # no skip
    iter_ds = iter(ds)
elif rank == 1:
    net = LLamaStage(dmodel=dmodel,num_heads=num_heads,
                device=device, n_layers=n_layers, ctx_size=seq_l)
elif rank == 2:
    tokenizer = SPTokenizer()
    net = LLamaLastStage(tokenizer.vocab_size,dmodel=dmodel,num_heads=num_heads,
                device=device, n_layers=n_layers, ctx_size=seq_l)
    ds = TinyStories(tokenizer,batch_size=batch_size, seq_l=seq_l) # no skip
    iter_ds = iter(ds)



optim = Adam(net.parameters(),lr=8e-4)

for itr in range(5_000):
    optim.zero_grad()
    # FORWARD PASS:
    if rank == 0:
        out = next(iter_ds)
        out = out.to(device)
        out = net.embed(out)

        dist.send(out.to("cpu"),1)

    elif rank == 1:

        inp_batch = torch.empty((batch_size,seq_l,dmodel))
        dist.recv(inp_batch,0)
        with torch.no_grad():
            inp_batch = inp_batch.to(device)
            inp_batch.requires_grad_()
            inp_batch.retain_grad()

        out = net(inp_batch)
        dist.send(out.to("cpu"),2)

    elif rank == 2:
        target = next(iter_ds)
        inp_batch = torch.empty((batch_size,seq_l,dmodel))
        dist.recv(inp_batch,1)
        with torch.no_grad():
            inp_batch = inp_batch.to(device)
            inp_batch.requires_grad_()
            inp_batch.retain_grad()

        logits = net(inp_batch)
        loss = causalLLMLoss(logits,target,tokenizer.vocab_size)
        print(loss.item())
        loss.backward()


    # BACKWARD PASS:
    if rank == 2:
        dist.send(inp_batch.grad.to("cpu"),1)
    elif rank == 1:
        inp_grad = torch.empty((batch_size,seq_l,dmodel))
        dist.recv(inp_grad,2)
        out.backward(inp_grad.to(device))
        dist.send(inp_batch.grad.to("cpu"),0)
    elif rank == 0:
        inp_grad = torch.empty((batch_size,seq_l,dmodel))
        dist.recv(inp_grad,1)
        out.backward(inp_grad.to(device))


    optim.step()
    torch.cuda.empty_cache()
````

Code is also alvailable in [intro_PP_1F1B.py](PP/1F1B/intro_PP_1F1B.py).

Simple enough so far right?

Torch does provide an implementation of [pipeline parallelism](https://pytorch.org/docs/stable/distributed.pipelining.html), but it is still in alpha and can be quite buggy at times.

### Microbatches

One major issue with Pipeline parallelisation is that devices are underutilised. What does this mean?

When we perform the sequential execution of 1 batch per device down the pipeline, all but one device are actually idle. With bigger models and higher communication overheads, increasing device utilisation (decreasing their idle time) can **signficantly** reduce training time. The simplest way to reduce this idle time is split the tasks (the batch) into smaller micro-batches and stream them through the pipeline. Thus we divide our batch $B$ into micro-batches $B_1,B_2...B_i$. Device in stage 0 processes microbatch $B_1$ and sends it to a device in stage 1. While communicating and without waiting for the corresbonding backwards pass of $B_1$, the device in stage 1 processes microbatch $B_2$ and sends it to the next device and so on. During the backwards pass it is important to have the corresponding input microbatch and its activation graph to compute the gradients correctly. Gradient updates are accumulated (for Torch this means that they are stored in the ```.grad``` parameter) per parameter. During the update step, the gradients need to be scaled by the number of microbatches (essentially averaged across all micro-batches).

An example of such a flow is presented below [[1]](https://arxiv.org/abs/2207.11912):

![Gpipe Execution](imgs/gpipe-fig.png)

You can even do multiple of these "waves" of forward and backwards microbatches before reaching the update phase.

However, even with such a method for pipelining, "bubbles" (times where devices are idle) can appear. In the above figure we can see device 1 having a huge idle time between the last forward pass and the first backwards pass.

A lot of contemporary research [[2]](https://arxiv.org/abs/2007.01045) [[3]](https://arxiv.org/abs/2401.10241) [[4]](https://dl.acm.org/doi/10.1145/3492321.3519563) is heavily focused on reducing these bubbles. For example, [Dapple](https://arxiv.org/abs/2007.01045) proposes interleaving the forward and backwards passes as:

![Dapple Execution](imgs/dapple.jpg)

While in this case the execution time being roughly the same, the memory requirements are signfiicantly lower compare to GPipe, which is similar to the approach mentioned above.

Implementing pipeline parallelism with microbatches is left as homework.

### Intuition on how does micro-batching work

Micro-batches can improve your throughput and also help you increase your batch size. It may be a bit hard to see why models still converge, so I include here a little description.

We understood so far how Data Parallel training is equivalent to the central training with just a larger batch size. Performing pipeline parallelism with micro-batches is not much different. You can think of each execution of the micro-batch as a separate Data Parallel execution, since the weights of the models are the same for all micro-batches. The only difference being that the logical Data Parallel model executing micro-batch $B_2$ begins its execution one time step *after* the logical model executing micro-batch $B_1$. At the end, the accumulated gradients need to be scaled, which is equivalent to averaging across the different DP nodes.


## Distributed Training for Large Language Models

Due to the sheer sizes of Large Language Models and the datasets we train them on, we make use of all techniques together when training them. We divide the model into multiple stages distributing them across devices, forming pipelines. On these pipelines we sequentially run micro-batches. Each stage has multiple devices. Thus we form multiple pipelines running in parallel and devices in the same stage synchronise their models between each other, forming Data Parallel training.

![LLM Execution](imgs/dtfm.webp)

*image courtesy of [DTFM](https://arxiv.org/abs/2206.01288)*

As homework you can attempt to create such a training with everything we learnt so far.
