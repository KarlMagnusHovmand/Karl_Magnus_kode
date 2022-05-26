# Data analysis project

Our project is titled **Correlation between fuel pricing and car registration** and documents the effect from increasing diesel and petrol prices on how many cars are registered monthly in Denmark.

The **results** of the project can be seen from running [Dataproject.ipynb](Dataproject.ipynb).

We apply the **following datasets**:

1. https://www.statbank.dk/statbank5a/SelectVarVal/Define.asp?Maintable=PRIS111&PLanguage=0 
2. https://www.statistikbanken.dk/statbank5a/default.asp?w=1920

**Dependencies:** 

Apart from a standard Anaconda Python 3 installation, the project requires the following installations:

import numpy as np

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

import pandas as pd

import pydst

dst = pydst.Dst(lang='en')

import seaborn as sns
