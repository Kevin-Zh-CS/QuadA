# QuadA Safety Evaluation & Alignment Suite

In this software suite, we provide two bash scripts to support the reproduction of the results obtained in our paper:

- `evaluate.sh` for reproducing the safety assessment results, and
- `align.sh` for enhancing LLMs via activation approximation-aware alignment (QuadA)

<br/>

## Preprequisites
Our scripts have been tested on a Ubuntu 20.04 platform with 4 NVIDIA A100 GPUs with 40GB VRAM.

> Note that QuadA's DPO alignment training is very memory intensive, and it could take more than 10 hours to complete using only four A100 40GB GPUs.

- Python 3.13.3
- CUDA 12.6
- Conda (optional)

<br/>

## Dependencies
### Switching to a Recommended CUDA version
We recommend using CUDA 12.6 for testing, although other versions may also work. To list & switch bewteen install CUDA versions in your local CUDA environment, use the provided script:

```bash
source scripts/cuda.sh
source scripts/cuda.sh 12.6
```

### Install Python Dependencies
```bash
pip install -r requirements.txt
```

or:

```bash
conda env create -f environment.yml --name quada
conda activate quada
```

If you choose to use `conda`, please create an editable installation of `transformers` by running:

```bash
cd transformers
pip install -e .
```

<br/>

## Usage
### Safety Evaluation
Update test parameters in `evaluate.sh` before running the script, an example completed set of parameters is given below:

```bash
cuda_devices=0,1,2,3
noise_source="LN"
noise_std=0.075
model_name="meta-llama/Llama-3.1-8B-Instruct"
modeling_file_path="./transformers/src/transformers/models/llama/modeling_llama.py"
```
After setting the parameters, run the evaluation pipeline:

```bash
./evaluate.sh
```

<br/>

### Safety Enhancement
First, adjust DPO training parameters such as batch sizes and offload config (`none`, `parameter`, `optimizer`, or `all`) in `scripts/dpo.sh`, if needed, depending on the number of GPUs available in your test environment.

Then, update test parameters in `align.sh` before running the script, a example completed set of arguments is given below:

```bash
--model_name_or_path=meta-llama/Llama-3.1-8B-Instruct
--output_dir=output/alignment/Llama-3.1-8B-Instruct-dpo
```

After setting the arguments, run the QuadA pipeline:

```bash
./align.sh
```

Output models are stored in the `output_dir`. To evaluate the safety of an output model, run `evaluate.sh` with the `model_name` variable set as the path to the output model in `output_dir`, an example is given below:

```bash
cuda_devices=0,1,2,3
noise_source="LN"
noise_std=0.075
model_name="./output/alignment/Llama-3.1-8B-Instruct-dpo"
modeling_file_path="./transformers/src/transformers/models/llama/modeling_llama.py"
```
