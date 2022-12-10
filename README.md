# Reproducing of Practical GAN-based Synthetic IP Header Trace Generation using NetShare 

### Motivation
Packet and flow level header traces are critical to many network management tasks, for instance they are used to develop new types of anomaly detection algorithms but access to such traces remains challenging due to business and privacy concerns. An alternative is to generate synthetic traces. 

In this project, we aim to reproduce NetShare, which can tackle many of the challenges by carefully understanding the limitations of GAN-based methods. They followed the following key ideas in building NetShare to tackle:

1. Learning synthetic models for a merged flow-level trace across epochs instead of treating header traces from each epoch as an independent tabular dataset. This reformulation captures the intra-and inter-epoch correlations of traces.
2. Data parallelism learning was introduced in this approach to improve the scalability.
3. To deal with privacy concerns for sharing the traces, differentially-private model training was used.

![Netshare Image](backup_results/netshare-pipeline.jpg)

The major contribution of the project is to effectively split the input into different chunks with respect to time and then encode and train them with a time-series GAN. This allows us to process the data faster and parallely train various models to generate results faster. 

### Implementation
The implementation didn't work out of the box so we modified some of the code base to make it easier to reproduce the work. You can find more instructions to reproduce the work below. 

Most of the implementation was redone to ensure that all the experiments can be run on a single machine as opposed to the multi-machine setup recommended by the authors. 

### Results

We observed the following graphs after running the experiments inside the `eval` folder. 


#### UGR16 Results
<p float="center">
  <img src="backup_results/plots/ugr16/cdf_ugr16_byt.jpg" width="250" />
  <img src="backup_results/plots/ugr16/cdf_ugr16_flow_size.jpg" width="250" /> 
  <img src="backup_results/plots/ugr16/cdf_ugr16_pkt.jpg" width="250" />
  <img src="backup_results/plots/ugr16/bar_proto.jpg" width="250" />
</p>

The above graphs denote the 
[CDF](https://en.wikipedia.org/wiki/Cumulative_distribution_function) with respect to
- Bytes
- Packets
- 5 Tuple
- Type of protocol

These graphs here indicate that how the real traces and the synthetic traces have high co-relation across several properties. Hence, it generates traces with high fidelity.

#### CAIDA Results
<p float="center">
  <img src="backup_results/plots/caida/cms_line_csiphash_dstip_top_10.jpg" width="250" />
  <img src="backup_results/plots/caida/cms_line_csiphash_srcip_top_10.jpg" width="250" /> 
  <img src="backup_results/plots/caida/cms_line_csiphash_srcip-dstip-srcport-dstport-proto_top_10.jpg" width="250" />
</p>

The above graphs denote the 
[Count Min Sketch](https://en.wikipedia.org/wiki/Count%E2%80%93min_sketch) with respect to
- Destination IP
- Source IP
- Five-tuple aggregation

They also indicate that donwstream applications perform similar to the original traces on the telemetry applications.

#### Downstream Testing (Botnet) Results
<p float="center">
  <img src="backup_results/plots/botnet/anomaly_botnet_bar.jpg" width="300" />
  <img src="backup_results/plots/botnet/cdf_caida_flow_size.jpg" width="300" /> 
</p>

The above graphs denote the 
[Spearman Correlation](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient) and CDF with respect to the flow size.

Based on the above results, we can also determine that the generated traces perform similar to the actual traces for anomaly detection. 

#### Conculsion
Based on the graphs,we draw the following observations from the results:

1. NetShare achieves high fidelity on feature distribution metrics across traces.

2. NetShare-generated traces perform with high-accuracy on Anomaly Detection Tasks

3. NetShare preserves the relative rank-order of Anomaly Detection algorithm performance

4. NetShare-generated traces perform similar to original traces on Telemetry

Did you find the results interesting ? You can replicate the experiment by following setup guide below! 

### Project details 

**Class Project:** Akanksha Cheeti, Annus Zulfiqar, Ashwin Nambiar,Syed Hasan Amin, Murayyiam Parvez, Syed Muhammed Abubaker

[[Class Project Slides](https://github.com/annuszulfiqar2021/NetShare/blob/project_ready/CS536_ProjectPresentation.pptx.pdf)][[Class Project Report](https://github.com/annuszulfiqar2021/NetShare/blob/project_ready/purdue-cs536-fall22-paper1.pdf)]

[[paper (SIGCOMM 2022)](https://dl.acm.org/doi/abs/10.1145/3544216.3544251)][[talk (SIGCOMM 2022)](https://www.youtube.com/watch?v=mWnFIncjtWg)]

**Authors:** [[Yucheng Yin](https://sniperyyc.com/)] [[Zinan Lin](http://www.andrew.cmu.edu/user/zinanl/)] [[Minhao Jin](https://www.linkedin.com/in/minhao-jin-1328b8164/)] [[Giulia Fanti](https://www.andrew.cmu.edu/user/gfanti/)] [[Vyas Sekar](https://users.ece.cmu.edu/~vsekar/)]

### CS536 Project Setup

> Please download the dataset file [here](https://drive.google.com/file/d/1GmA1Jzqf4RuN7IJUCjInv9IoMcXmJhYO/view?usp=sharing) and unzip to `data/` directory in the project directory before proceeding to the next step.

#### Run the following Makefile targets in this order
```sh
conda activate NetShare

cd /path/to/NetShare/home-directory/

# 1. To preprocess the dataset without differential privacy
make preprocess-no-dp

# 2. Clean the previous training outputs before retraining
make clean-results

# 3. To train the GAN
make train-no-dp

# 4. To generate using the trained GAN
make generate-no-dp

```

### Setup
#### Single-machine setup
Single-machine is only recommended for very small datasets and quick validation/prototype as GANs are very computationally expensive. We recommend using virtual environment to avoid conflicts (e.g., Anaconda).

```Bash
# Assume Anaconda is installed
# create virtual environment
conda create --name NetShare python=3.6

# installing dependencies
cd util/
pip3 install -r requirements.txt
```
### Dataset preparation
#### Description
Datasets used for the experiments

1. [UGR16](https://nesg.ugr.es/nesg-ugr16/) dataset consists of traffic (including attacks) from NetFlow v9 collectors in a Spanish ISP network. We used data from the third week of March 2016. 

2. [CAIDA](https://www.caida.org/catalog/datasets/passive_dataset/) contains anonymized traces from high-speed monitors on a commercial backbone link. Our subset is from the New York collector in March 2018. (**Require an CAIDA account to download the data**)

### Refererence
Part of the source code is adapated from the following open-source projects:

- [DoppelGANger](https://github.com/fjxmlzn/DoppelGANger)
- [GPUTaskScheduler](https://github.com/fjxmlzn/GPUTaskScheduler)
- [BSN](https://github.com/fjxmlzn/BSN)
