# AlphaPulldown manual:
# Example3
# Aims: Model activation of phosphoinositide 3-kinase by the influenza A virus NS1 protein (PDB: 3L4Q)
## 1st step: compute multiple sequence alignment (MSA) and template features using provided pbd templates (run on CPU)

This complex can not be modeled with vanilla AlphaFold Multimer, since it is a host-pathogen interaction.
Firstly, download sequences of NS1(Uniprot: [P03496](https://www.uniprot.org/uniprotkb/P03496/entry)) and P85B(uniprot:[P23726](https://www.uniprot.org/uniprotkb/P23726/entry)) proteins.
Then download the multimeric template in either pdb or mmCIF format(PDB: [3L4Q](https://www.rcsb.org/structure/3L4Q)).
Create directories named "fastas" and "templates" and put the sequences and pdb/cif files in the corresponding directories.
Finally, create a text file with description for generating features (description.csv).

**Please note**, the first column must be an exact copy of the protein description from your fasta files. Please consider shortening them in fasta files using your favorite text editor for convenience. These names will be used to generate pickle files with monomeric features!
The description.csv for the NS1-P85B complex should look like:
```
>sp|P03496|NS1_I34A1, 3L4Q.cif, A
>sp|P23726|P85B_BOVIN, 3L4Q.cif, C
```
In this example we refer to the NS1 protein as chain A and to the P85B protein as chain C in multimeric template 3L4Q.cif.

**Please note**, that your template will be renamed to a PDB code taken from *_entry_id*. If you use a *.pdb file instead of *.cif, AlphaPulldown will first try to parse the PDB code from the file. Then it will check if the filename is 4-letter long. If it is not, it will generate a random 4-letter code and use it as the PDB code.

Now run:
```bash
  create_individual_features_with_templates.py \
    --description_file=description.csv \
    --fasta_paths=fastas/P03496.fasta,fastas/P23726.fasta \
    --path_to_mmt=templates/ \
    --data_dir=/scratch/AlphaFold_DBs/2.3.2/ \
    --save_msa_files=True \
    --output_dir=features\ 
    --use_precomputed_msas=True \
    --max_template_date=2050-01-01 \
    --skip_existing=True
```
It is also possible to combine all your fasta files into a single fasta file.
```create_individual_features_with_templates.py``` will compute the features similarly to the create_individual_features.py, but will utilize the provided templates instead of the PDB database.
 
 ------------------------

## 1.1 Explanation about the parameters

See [Example 1](https://github.com/KosinskiLab/AlphaPulldown/blob/main/manuals/example_1.md#11-explanation-about-the-parameters)

## 2nd step: Predict structures (run on GPU)

#### **Task 1**
To predict structure we can use the usual ```run_multimer_jobs.py``` in custom mode (See [Example 2](https://github.com/KosinskiLab/AlphaPulldown/blob/main/manuals/example_2.md#2nd-step-predict-structures-run-on-gpu)) with an extra ```--multimeric_mode=True``` flag, that deactivates per-chain multimeric binary mask.
The user can also specify the depth of the MSA that is taken for modelling to increase the influence of the template on the predicted model. This can be done by using the flag ```--msa_depth```. Please note, that only the first 2 AlphaFold models are guided by the templates. To specify the model name you want to apply use the following flag: ```--model_names=model_1_multimer_v3,model_2_multimer_v3``` (for models 1 and 2).
If you do not know the exact MSA depth, there is another flag ```--gradient_msa_depth=True``` for exploring the desired MSA depth. This flag generates a set of logarithmically distributed points (denser at lower end) with the number of points equal to the number of predictions. The MSA depth (```num_msa```) starts from 16 and ends with the maximum value taken from the model config file. The ```extra_num_msa``` is always calculated as ```4*num_msa```.
The command line interface for using custom mode will then become:

```
run_multimer_jobs.py \
  --mode=custom \
  --num_cycle=3 \
  --num_predictions_per_model=<any number you want> \
  --output_path=<path to output directory> \ 
  --data_dir=<path to AlphaFold data directory> \ 
  --protein_lists=custom_mode.txt \
  --monomer_objects_dir=<path to features generated by create_individual_features_with_templates.py> \
  --multimeric_mode=True \
  --msa_depth=<any number you want> \
  --gradient_msa_depth=<True or False, overwrites msa_depth if provided> \
  --model_names=<coma separated names of the models> \
  --job_index=<corresponds to the string number from custom_mode.txt, don't provide for sequential execution>
```


### Running on a computer cluster in parallel

On a compute cluster, you may want to run all jobs in parallel as a [job array](https://slurm.schedmd.com/job_array.html). For example, on SLURM queuing system at EMBL we could use the following ```create_feature_jobs_SLURM.sh``` sbatch script:
```bash
#!/bin/bash

#A typical run takes couple of hours but may be much longer
#SBATCH --job-name=array
#SBATCH --time=5:00:00

#log files:
#SBATCH -e logs/create_individual_features_%A_%a_err.txt
#SBATCH -o logs/create_individual_features_%A_%a_out.txt

#qos sets priority
#SBATCH --qos=normal

#SBATCH -p htc-el8
#Limit the run to a single node
#SBATCH -N 1

#Adjust this depending on the node
#SBATCH --ntasks=8
#SBATCH --mem=32000

module load HMMER/3.3.2-gompic-2020b
module load HH-suite/3.3.0-gompic-2020b
module load Anaconda3
source activate AlphaPulldown

  create_individual_features_with_templates.py \
    --description_file=description.csv \
    --fasta_paths=fastas/P03496.fasta,fastas/P23726.fasta \
    --path_to_mmt=templates/ \
    --data_dir=/scratch/AlphaFold_DBs/2.3.2/ \
    --save_msa_files=True \
    --output_dir=features \ 
    --use_precomputed_msas=True \
    --max_template_date=2050-01-01 \
    --skip_existing=True \
    --job_index=$SLURM_ARRAY_TASK_ID
```

and the following ```run_multimer_jobs_SLURM.sh``` sbatch script:

```bash
#!/bin/bash

#A typical run takes couple of hours but may be much longer
#SBATCH --job-name=array
#SBATCH --time=2-00:00:00

#log files:
#SBATCH -e logs/run_multimer_jobs_%A_%a_err.txt
#SBATCH -o logs/run_multimer_jobs_%A_%a_out.txt

#qos sets priority
#SBATCH --qos=normal

#SBATCH -p gpu-el8

#Reserve the entire GPU so no-one else slows you down
#SBATCH --gres=gpu:1

#Limit the run to a single node
#SBATCH -N 1

#Adjust this depending on the node
#SBATCH --ntasks=8
#SBATCH --mem=64000

module load Anaconda3 
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1
source activate AlphaPulldown

MAXRAM=$(echo `ulimit -m` '/ 1024.0'|bc)
GPUMEM=`nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|tail -1`
export XLA_PYTHON_CLIENT_MEM_FRACTION=`echo "scale=3;$MAXRAM / $GPUMEM"|bc`
export TF_FORCE_UNIFIED_MEMORY='1'

run_multimer_jobs.py  \
  --mode=custom \
  --num_cycle=3 \
  --num_predictions_per_model=5 \
  --output_path=<path to output directory> \ 
  --data_dir=<path to AlphaFold data directory> \ 
  --protein_lists=custom_mode.txt \
  --monomer_objects_dir=/path/to/monomer_objects_directory \
  --multimeric_mode=True \
  --msa_depth=128 \
  --model_names=model_1_multimer_v3,model_2_multimer_v3 \
  --gradient_msa_depth=False \
  --job_index=$SLURM_ARRAY_TASK_ID    
```
and then run using:

```
mkdir -p logs
count=`grep -c "" description.csv` #count lines even if the last one has no end of line
sbatch --array=1-$count create_feature_jobs_SLURM.sh
count=`grep -c "" custom_mode.txt` #likewise for predictions
sbatch --array=1-$count run_multimer_jobs_SLURM.sh
```
After the successful run one can evaluate and visualise the results in a usual manner (see e.g. [Example 2](https://github.com/KosinskiLab/AlphaPulldown/blob/main/manuals/example_2.md#2nd-step-predict-structures-run-on-gpu))
