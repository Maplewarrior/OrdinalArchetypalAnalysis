#!/bin/sh

### Specify queue
#BSUB -q hpc

#### Job name
#BSUB -J Bernoulli_RWD

#BSUB -n 4
#BSUB -R "span[hosts=1]"

#BSUB -R "rusage[mem=50GB]"

#BSUB -W 72:00

#BSUB -u aejew@dtu.dk
#BSUB -u emiliewedenborg@gmail.com



#BSUB -oo resultatBernoulli_%J.out
#BSUB -oo PerformanceBernoulli_%J.err

##### Start notification
#BSUB -B

#### End notification
#BSUB -N
 

module load python3

source test-env/bin/activate
python -m pip install scipy tqdm
python3 -c "import scipy"
python3 -c "from ClosedFormArchetypalAnalysis import ClosedFormArchetypalAnalysis"
python3 -c "from MUPoissonSparse import MUPoissonSparse"
python3 -c "from MUBernoulli import MUBernoulli"
python3 -c "import time"
python3 -c "import csv"
python3 -c "import numpy"
python3 -c "import tqdm"
python3 -c "from AAPoissonSparse import AAPoissonSparse"
python3 testBernoulli.py
