# AGMA

This repo is the source code of the paper "AGMA: Agreement Generation Multi-agent Reinforcement Learning". It is constructed based on the [SMAC](https://github.com/oxwhirl/smac) and [PyMARL](https://github.com/oxwhirl/pymarl).

## Installation

Clone this repo and set up SMAC:

```sh
cd Sub_AVG
bash install_sc2.sh
```

The installation is same with [PyMARL](https://github.com/oxwhirl/pymarl). For quick start, you can install the necessary packages directly by (not recommended):

```sh
pip install -r requirements.txt
```

If needed, add  directory that contains StarCraftII to the system's path ('xxx' is your local file path):

```sh
export PATH="$PATH:/xxx/AGMA/3rdparty/StarCraftII"
```

## Run Sub-AVG

We provide  `run_AGMA.sh` , `run_QMIX.sh`, `run_VDN.sh`, `run_QTRAN.sh` and `run_IQL.sh` for quick start.

## Results

The results save path is:  "./results/sacred".

