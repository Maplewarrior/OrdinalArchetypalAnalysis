# Introduction
This repository contains the code introduced in the paper *Modeling Human Responses by Ordinal Archetypal Analysis*, accepted at 2024 IEEE International Workshop on Machine Learning for Signal Processing.

We introduce a novel framework for Archetypal Analysis (AA) tailored to ordinal data, particularly from questionnaires. Unlike existing methods, the proposed method, Ordinal Archetypal Analysis (OAA), bypasses the two-step process of transforming ordinal data into continuous scales and operates directly on the ordinal data. We extend traditional AA methods to handle the subjective nature of questionnaire-based data, acknowledging individual differences in scale perception. We introduce the Response Bias Ordinal Archetypal Analysis (RBOAA), which learns individualized scales for each subject during optimization. The effectiveness of these methods is demonstrated on synthetic data and the European Social Survey dataset, highlighting their potential to provide deeper insights into human behavior and perception. The study underscores the importance of considering response bias in cross-national research and offers a principled approach to analyzing ordinal data through archetypal analysis.

## :zap: Modeling Human Responses by Ordinal Archetypal Analysis
- To reproduce the results outlined in the paper and familiarize yourself with the codebase consult demo.ipynb.
- Both synthetic data and data from the ESS8 are used in the paper. The ESS8 dataset is publicly available on the ESS website and we direct readers here to obtain a copy. To help reproduce the results obtained for the Two-step Archetypal Analysis (Fernández et al., 2021), we provide the outputs of the Ordered Stereotype Model (OSM) in the "data" folder. For the ESS8 dataset, we provide arrays of likert scale values achieved upon convergence. This is further elaborated on in demo.ipynb.

###  :electric_plug: Installation
To setup the repository follow the below steps:
- clone the repository to your local machine.
- create a virtual environment.
- Run the following commands in the venv:

```
$ pip install -r requirements.txt
$ pip install -e .
```

###  :package: Commands
Interfacing with the codebase is done via main.py. Here the user can run analyses or visualize results. When carrying out analyses, experiments are configured by the .yaml configuration file along with the argparser arguments specified.

- Analyse:
```
$ python main.py analyse --config-path configs/ESS8_config.yaml --X-path data/ESS8/ESS8_GB.csv --save-folder ESS8_GB
```

- Visualize:
```
$ python main.py visualize --config-path configs/ESS8_config.yaml --save-folder ESS8_GB
```

###  :file_folder: File Structure
```
.
├── src
│   ├── methods
│   │   ├── __init__.py
│   │   └── CAA_class.py
│   │   └── OAA_class.py
│   │   └── RBOAA_class.py
│   ├── misc
│   │   ├── __init__.py
│   │   └── loading_bar_class.py
│   ├── utils
│   │   ├── __init__.py
│   │   └── synthetic_data_class.py
|   |   └── eval_measures.py
|   |   └── corruptData.py
|   |   └── filter_ESS8.py
|   |   └── result_maker.py
|   |   └── visualize_results.py
|   |   └── AA_result_class.py
│   └── visualizations
│       ├── NMI_archetypes.py
│       └── NMI_stability.py
│       └── archetypal_answers.py
│       └── denoising.py
│       └── functions.py
│       └── loss_archetypal_plot.py
│       └── response_bias.py
├── demo.py
├── requirements.txt
└── README.md
```

## :star2: Credit/Acknowledgment
This work is authored by Anna Emilie J. Wedenborg, Michael A. Harborg, Andreas Bigom, and Oliver Elmgreen, Marcus Presutti, Andreas Råskov, Fumiko Kano Glückstad, Mikkel Schmidt, and Morten Mørup.


##  :lock: License
This work has been given a MIT license.
