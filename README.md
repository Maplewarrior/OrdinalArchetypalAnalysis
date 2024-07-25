# Introduction
This reposetory contains the code introduced in the paper *Modeling Human Responses by Ordinal Archetypal Analysis*, accepted at 2024 IEEE International Workshop on Machine Learning for Signal Processing.
We introduce a novel framework for Archetypal Analysis (AA) tailored to ordinal data, particularly from questionnaires. Unlike existing methods, the proposed method, Ordinal Archetypal Analysis (OAA), bypasses the two-step process of transforming ordinal data into continuous scales and operates directly on the ordinal data. We extend traditional AA methods to handle the subjective nature of questionnaire-based data, acknowledging individual differences in scale perception. We introduce the Response Bias Ordinal Archetypal Analysis (RBOAA), which learns individualized scales for each subject during optimization. The effectiveness of these methods is demonstrated on synthetic data and the European Social Survey dataset, highlighting their potential to provide deeper insights into human behavior and perception. The study underscores the importance of considering response bias in cross-national research and offers a principled approach to analyzing ordinal data through archetypal analysis.

## :zap: Modeling Human Responses by Ordinal Archetypal Analysis
- Write how to reproduce results from paper. 
- Maybe something about the data 

###  :electric_plug: Installation
- pip install ... 

```
$ pip install .... 
```

###  :package: Commands
- Maybe a description of a simple function call 

###  :file_folder: File Structure
```
.
├── src
│   ├── inference
│   │   ├── ResultMaker.py
│   │   └── __init__.py
│   │   └── utils.py
│   ├── methods
│   │   ├── .DS_Store
│   │   └── __init__.py
│   │   └── CAA_class.py
│   │   └── OAA_class.py
│   │   └── RBOAA_class.py
│   ├── misc
│   │   ├── __init__.py
│   │   └── loading_bar_class.py
│   ├── utils
│   │   ├── __init__.py
│   │   └── ??
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
Credit the authors here.

##  :lock: License
Add a license here, or a link to it.
