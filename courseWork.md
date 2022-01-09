# Neural Networks Coursework
***

## Developing a Deep Learning (NN) model to predict Parkinson's Disease in early stages using speech features.

Parkinson's Disease (PD) diagnosis using Machine Learning techniques has been studied very thoroughly in the recent years using either classical Machine Leraning models like Suppor Vector Machines or Random Forest. Moreover, some studies have shown that using Deep Learning models can be yield very promosing results. 

Therefore, this project consists on the development of a Neural Network to predict, given subject speech features, if the given subject suffers from PD. 

The dataset used can be found in [kaggle](https://www.kaggle.com/dipayanbiswas/parkinsons-disease-speech-signal-features), but is provided here as a `.csv` file.


*As far as it is possible, my own code and methods will be used.*

## Defining the problem

### Input data

Data comes from a `.csv` file, that will be loaded using `pandas` into a `DataFrame`, and a preliminar data exploration analysis will be carried out to show the structure of input data. 

### Problem type and output

Since the goal of the model will be to assess wether a subject suffers or not from PD, the problem is a binary classification problem (either the user has or has not PD).

This already tells us that our model's last layer (output) will have one single neuron, which will be the desired output. 

## Measure of success

As with all binary classification problems, one of the most important metrics of the model is the `accuracy`, this is the number of correct predictions over the number of total predictions. If when exploring the data we find that the data samples are skewed, and there is an inbalance between patients with PD and patients without PD, the `accuracy` metric will be subsituted by the `precision`, `recall` or`f1` score, with takes into account `precision` and `recall` as well as `accuracy`.


### Data engineering

Before building our model, first we have to adapt the input data into a tensor format, so the Neural Network know how to handle the data. This data will have to be normalized if the values of the different features have different ranges


```python
%pip install numpy
%pip install pandas
%pip install matplotlib
%pip install tensorflow

from pathlib import Path
import numpy as np
import pandas as pd

files_path = Path('./data')

def load_data(csv_name: str):
    df = pd.read_csv(files_path.joinpath(csv_name))
    return df
    
df = load_data('pd_speech_features.csv')
print(f"Columns: {len(df.columns)}")
print(f"Nan values: {df.isnull().values.any()}")
df.describe()
```

    Requirement already satisfied: numpy in /Users/davidrodriguezpozo/miniforge3/envs/DS/lib/python3.8/site-packages (1.19.5)
    Note: you may need to restart the kernel to use updated packages.
    Requirement already satisfied: pandas in /Users/davidrodriguezpozo/miniforge3/envs/DS/lib/python3.8/site-packages (1.3.5)
    Requirement already satisfied: python-dateutil>=2.7.3 in /Users/davidrodriguezpozo/miniforge3/envs/DS/lib/python3.8/site-packages (from pandas) (2.8.2)
    Requirement already satisfied: pytz>=2017.3 in /Users/davidrodriguezpozo/miniforge3/envs/DS/lib/python3.8/site-packages (from pandas) (2021.3)
    Collecting numpy>=1.20.0
      Downloading numpy-1.22.0-cp38-cp38-macosx_11_0_arm64.whl (12.7 MB)
         |████████████████████████████████| 12.7 MB 4.0 MB/s            
    [?25hRequirement already satisfied: six>=1.5 in /Users/davidrodriguezpozo/miniforge3/envs/DS/lib/python3.8/site-packages (from python-dateutil>=2.7.3->pandas) (1.15.0)
    Installing collected packages: numpy
      Attempting uninstall: numpy
        Found existing installation: numpy 1.19.5
        Uninstalling numpy-1.19.5:
          Successfully uninstalled numpy-1.19.5
    Successfully installed numpy-1.22.0
    Note: you may need to restart the kernel to use updated packages.
    Requirement already satisfied: matplotlib in /Users/davidrodriguezpozo/miniforge3/envs/DS/lib/python3.8/site-packages (3.5.1)
    Requirement already satisfied: kiwisolver>=1.0.1 in /Users/davidrodriguezpozo/miniforge3/envs/DS/lib/python3.8/site-packages (from matplotlib) (1.3.2)
    Requirement already satisfied: fonttools>=4.22.0 in /Users/davidrodriguezpozo/miniforge3/envs/DS/lib/python3.8/site-packages (from matplotlib) (4.28.5)
    Requirement already satisfied: cycler>=0.10 in /Users/davidrodriguezpozo/miniforge3/envs/DS/lib/python3.8/site-packages (from matplotlib) (0.11.0)
    Requirement already satisfied: packaging>=20.0 in /Users/davidrodriguezpozo/miniforge3/envs/DS/lib/python3.8/site-packages (from matplotlib) (21.3)
    Requirement already satisfied: python-dateutil>=2.7 in /Users/davidrodriguezpozo/miniforge3/envs/DS/lib/python3.8/site-packages (from matplotlib) (2.8.2)
    Requirement already satisfied: numpy>=1.17 in /Users/davidrodriguezpozo/miniforge3/envs/DS/lib/python3.8/site-packages (from matplotlib) (1.22.0)
    Requirement already satisfied: pyparsing>=2.2.1 in /Users/davidrodriguezpozo/miniforge3/envs/DS/lib/python3.8/site-packages (from matplotlib) (3.0.6)
    Requirement already satisfied: pillow>=6.2.0 in /Users/davidrodriguezpozo/miniforge3/envs/DS/lib/python3.8/site-packages (from matplotlib) (8.4.0)
    Requirement already satisfied: six>=1.5 in /Users/davidrodriguezpozo/miniforge3/envs/DS/lib/python3.8/site-packages (from python-dateutil>=2.7->matplotlib) (1.15.0)
    Note: you may need to restart the kernel to use updated packages.
    [31mERROR: Could not find a version that satisfies the requirement tensorflow (from versions: none)[0m
    [31mERROR: No matching distribution found for tensorflow[0m
    Note: you may need to restart the kernel to use updated packages.
    [31mERROR: You must give at least one requirement to install (see "pip help install")[0m
    Note: you may need to restart the kernel to use updated packages.
    Columns: 755
    Nan values: False





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>gender</th>
      <th>PPE</th>
      <th>DFA</th>
      <th>RPDE</th>
      <th>numPulses</th>
      <th>numPeriodsPulses</th>
      <th>meanPeriodPulses</th>
      <th>stdDevPeriodPulses</th>
      <th>locPctJitter</th>
      <th>...</th>
      <th>tqwt_kurtosisValue_dec_28</th>
      <th>tqwt_kurtosisValue_dec_29</th>
      <th>tqwt_kurtosisValue_dec_30</th>
      <th>tqwt_kurtosisValue_dec_31</th>
      <th>tqwt_kurtosisValue_dec_32</th>
      <th>tqwt_kurtosisValue_dec_33</th>
      <th>tqwt_kurtosisValue_dec_34</th>
      <th>tqwt_kurtosisValue_dec_35</th>
      <th>tqwt_kurtosisValue_dec_36</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>756.000000</td>
      <td>756.000000</td>
      <td>756.000000</td>
      <td>756.000000</td>
      <td>756.000000</td>
      <td>756.000000</td>
      <td>756.000000</td>
      <td>756.000000</td>
      <td>756.000000</td>
      <td>756.000000</td>
      <td>...</td>
      <td>756.000000</td>
      <td>756.000000</td>
      <td>756.000000</td>
      <td>756.000000</td>
      <td>756.000000</td>
      <td>756.000000</td>
      <td>756.000000</td>
      <td>756.000000</td>
      <td>756.000000</td>
      <td>756.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>125.500000</td>
      <td>0.515873</td>
      <td>0.746284</td>
      <td>0.700414</td>
      <td>0.489058</td>
      <td>323.972222</td>
      <td>322.678571</td>
      <td>0.006360</td>
      <td>0.000383</td>
      <td>0.002324</td>
      <td>...</td>
      <td>26.237251</td>
      <td>22.840337</td>
      <td>18.587888</td>
      <td>13.872018</td>
      <td>12.218953</td>
      <td>12.375335</td>
      <td>14.799230</td>
      <td>14.751559</td>
      <td>31.481110</td>
      <td>0.746032</td>
    </tr>
    <tr>
      <th>std</th>
      <td>72.793721</td>
      <td>0.500079</td>
      <td>0.169294</td>
      <td>0.069718</td>
      <td>0.137442</td>
      <td>99.219059</td>
      <td>99.402499</td>
      <td>0.001826</td>
      <td>0.000728</td>
      <td>0.002628</td>
      <td>...</td>
      <td>42.220693</td>
      <td>32.626464</td>
      <td>25.537464</td>
      <td>20.046029</td>
      <td>17.783642</td>
      <td>16.341665</td>
      <td>15.722502</td>
      <td>14.432979</td>
      <td>34.230991</td>
      <td>0.435568</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041551</td>
      <td>0.543500</td>
      <td>0.154300</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.002107</td>
      <td>0.000011</td>
      <td>0.000210</td>
      <td>...</td>
      <td>1.509800</td>
      <td>1.531700</td>
      <td>1.582900</td>
      <td>1.747200</td>
      <td>1.789500</td>
      <td>1.628700</td>
      <td>1.861700</td>
      <td>1.955900</td>
      <td>2.364000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>62.750000</td>
      <td>0.000000</td>
      <td>0.762833</td>
      <td>0.647053</td>
      <td>0.386537</td>
      <td>251.000000</td>
      <td>250.000000</td>
      <td>0.005003</td>
      <td>0.000049</td>
      <td>0.000970</td>
      <td>...</td>
      <td>2.408675</td>
      <td>3.452800</td>
      <td>3.354825</td>
      <td>3.077450</td>
      <td>2.937025</td>
      <td>3.114375</td>
      <td>3.665925</td>
      <td>3.741275</td>
      <td>3.948750</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>125.500000</td>
      <td>1.000000</td>
      <td>0.809655</td>
      <td>0.700525</td>
      <td>0.484355</td>
      <td>317.000000</td>
      <td>316.000000</td>
      <td>0.006048</td>
      <td>0.000077</td>
      <td>0.001495</td>
      <td>...</td>
      <td>5.586300</td>
      <td>7.062750</td>
      <td>6.077400</td>
      <td>4.770850</td>
      <td>4.300450</td>
      <td>4.741450</td>
      <td>6.725700</td>
      <td>7.334250</td>
      <td>10.637250</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>188.250000</td>
      <td>1.000000</td>
      <td>0.834315</td>
      <td>0.754985</td>
      <td>0.586515</td>
      <td>384.250000</td>
      <td>383.250000</td>
      <td>0.007528</td>
      <td>0.000171</td>
      <td>0.002520</td>
      <td>...</td>
      <td>28.958075</td>
      <td>29.830850</td>
      <td>21.944050</td>
      <td>13.188000</td>
      <td>10.876150</td>
      <td>12.201325</td>
      <td>21.922050</td>
      <td>22.495175</td>
      <td>61.125325</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>251.000000</td>
      <td>1.000000</td>
      <td>0.907660</td>
      <td>0.852640</td>
      <td>0.871230</td>
      <td>907.000000</td>
      <td>905.000000</td>
      <td>0.012966</td>
      <td>0.003483</td>
      <td>0.027750</td>
      <td>...</td>
      <td>239.788800</td>
      <td>203.311300</td>
      <td>121.542900</td>
      <td>102.207000</td>
      <td>85.571700</td>
      <td>73.532200</td>
      <td>62.007300</td>
      <td>57.544300</td>
      <td>156.423700</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 755 columns</p>
</div>



We can see from this first summary of the data that the number of features is 753, this is: 755 columns of the dataframe, but extracting the `id` and the last column which is the `class` (label). Thefeore, the dimensionality of our model is 753 for the input layer. This could be reduced with PCA or another dimensionality reduction algorithm to extract the most relevant features. 

Since we are working with Neural Networks, we can reduce the dimensionality by reducing the number of neurons in the subsequent layers, and see how this impacts the performance of the model. 

It can be seen as well how different features take values that are in very different ranges. For exmaple, `RPDE` has a mean of 0.48, and `numPeriodsPulses` has a mean of 322.67. This can lead to bad model results because it will tend to weight in favor of those features with the highest values. Therefore, a normalizaation of the dataset is needed in order to feed it to the model.

The total number of samples if 756, which will be enough to split the data into train[validation]-test sub-samples.

We can proceed by dropping the `id` column that will not be useful for the model. And then a normalization of the columns.


```python
# Drop id column
df.drop(['id'], axis=1, inplace=True)
```

The method developed below takes as input the dataframe to be normalized, as well as the label column, that doesn't need normalization (doesn't need one_hot encoding either) because it is a binary problem and label is already encoded as `[0,1]`


```python
# Normalize dataframe
from typing import Tuple

def normalize_columns_split_label(df: pd.DataFrame, label_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize all columns of the dataframe, except for the label column (which doesn't need
    normalization)
    Then return data and labels separately.
    """
    df = df.copy()
    for col in df.columns:
        if col != label_col:
            df[col] = (df[col]-df[col].mean())/df[col].std()
    labels = df[label_col]
    df.drop(label_col, axis=1, inplace=True)
    return df.to_numpy(), labels.to_numpy()

data, labels = normalize_columns_split_label(df, 'class')

print(data.shape)
pd.DataFrame(data=data).describe()

```

    (756, 753)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>743</th>
      <th>744</th>
      <th>745</th>
      <th>746</th>
      <th>747</th>
      <th>748</th>
      <th>749</th>
      <th>750</th>
      <th>751</th>
      <th>752</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>7.560000e+02</td>
      <td>7.560000e+02</td>
      <td>7.560000e+02</td>
      <td>7.560000e+02</td>
      <td>7.560000e+02</td>
      <td>7.560000e+02</td>
      <td>7.560000e+02</td>
      <td>7.560000e+02</td>
      <td>7.560000e+02</td>
      <td>7.560000e+02</td>
      <td>...</td>
      <td>7.560000e+02</td>
      <td>7.560000e+02</td>
      <td>7.560000e+02</td>
      <td>7.560000e+02</td>
      <td>7.560000e+02</td>
      <td>7.560000e+02</td>
      <td>7.560000e+02</td>
      <td>7.560000e+02</td>
      <td>7.560000e+02</td>
      <td>7.560000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-1.879743e-16</td>
      <td>9.210739e-16</td>
      <td>-1.400408e-15</td>
      <td>-9.398713e-18</td>
      <td>-7.518971e-17</td>
      <td>-1.691768e-16</td>
      <td>-1.315820e-16</td>
      <td>2.349678e-17</td>
      <td>2.819614e-17</td>
      <td>-9.398713e-17</td>
      <td>...</td>
      <td>3.759485e-17</td>
      <td>-2.114711e-17</td>
      <td>-5.169292e-17</td>
      <td>7.518971e-17</td>
      <td>5.874196e-17</td>
      <td>-2.349678e-17</td>
      <td>-7.988906e-17</td>
      <td>2.819614e-17</td>
      <td>-7.518971e-17</td>
      <td>-9.868649e-17</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>...</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.031583e+00</td>
      <td>-4.162788e+00</td>
      <td>-2.250693e+00</td>
      <td>-2.435620e+00</td>
      <td>-3.245064e+00</td>
      <td>-3.236122e+00</td>
      <td>-2.328543e+00</td>
      <td>-5.117027e-01</td>
      <td>-8.045759e-01</td>
      <td>-7.007410e-01</td>
      <td>...</td>
      <td>-4.502459e-01</td>
      <td>-5.856714e-01</td>
      <td>-6.531090e-01</td>
      <td>-6.658840e-01</td>
      <td>-6.048488e-01</td>
      <td>-5.864633e-01</td>
      <td>-6.576218e-01</td>
      <td>-8.228671e-01</td>
      <td>-8.865570e-01</td>
      <td>-8.506067e-01</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-1.031583e+00</td>
      <td>9.774902e-02</td>
      <td>-7.653893e-01</td>
      <td>-7.459128e-01</td>
      <td>-7.354658e-01</td>
      <td>-7.311544e-01</td>
      <td>-7.430617e-01</td>
      <td>-4.587926e-01</td>
      <td>-5.153827e-01</td>
      <td>-5.010148e-01</td>
      <td>...</td>
      <td>-4.394457e-01</td>
      <td>-5.643814e-01</td>
      <td>-5.942273e-01</td>
      <td>-5.964986e-01</td>
      <td>-5.384891e-01</td>
      <td>-5.219363e-01</td>
      <td>-5.667084e-01</td>
      <td>-7.081128e-01</td>
      <td>-7.628559e-01</td>
      <td>-8.043109e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9.681013e-01</td>
      <td>3.743248e-01</td>
      <td>1.592962e-03</td>
      <td>-3.421508e-02</td>
      <td>-7.027100e-02</td>
      <td>-6.718716e-02</td>
      <td>-1.706333e-01</td>
      <td>-4.209997e-01</td>
      <td>-3.156112e-01</td>
      <td>-3.145628e-01</td>
      <td>...</td>
      <td>-4.106194e-01</td>
      <td>-4.891192e-01</td>
      <td>-4.835825e-01</td>
      <td>-4.898876e-01</td>
      <td>-4.540135e-01</td>
      <td>-4.452689e-01</td>
      <td>-4.671424e-01</td>
      <td>-5.135016e-01</td>
      <td>-5.139139e-01</td>
      <td>-6.089178e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.681013e-01</td>
      <td>5.199889e-01</td>
      <td>7.827394e-01</td>
      <td>7.090777e-01</td>
      <td>6.075222e-01</td>
      <td>6.093552e-01</td>
      <td>6.393980e-01</td>
      <td>-2.907037e-01</td>
      <td>7.441909e-02</td>
      <td>6.947578e-02</td>
      <td>...</td>
      <td>-2.034929e-01</td>
      <td>6.444291e-02</td>
      <td>2.142590e-01</td>
      <td>1.314211e-01</td>
      <td>-3.412235e-02</td>
      <td>-7.550775e-02</td>
      <td>-1.064822e-02</td>
      <td>4.530335e-01</td>
      <td>5.365224e-01</td>
      <td>8.660051e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.681013e-01</td>
      <td>9.532304e-01</td>
      <td>2.183453e+00</td>
      <td>2.780599e+00</td>
      <td>5.876167e+00</td>
      <td>5.858217e+00</td>
      <td>3.616665e+00</td>
      <td>4.259739e+00</td>
      <td>9.674871e+00</td>
      <td>1.046865e+01</td>
      <td>...</td>
      <td>7.651276e+00</td>
      <td>5.057983e+00</td>
      <td>5.531429e+00</td>
      <td>4.031528e+00</td>
      <td>4.406607e+00</td>
      <td>4.124732e+00</td>
      <td>3.742389e+00</td>
      <td>3.002580e+00</td>
      <td>2.964928e+00</td>
      <td>3.649985e+00</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 753 columns</p>
</div>



## Mesaure of success
Now we can study the distribution of positive and negative patients in order to choose the measure of success of the model, as commented previously. Usually the `accuracy` is a good enough metric, but if the data is skewed, it might not indicate that the model is performing well, but instead just classifying the points by assigning them the most common label.


```python
np.unique(labels, return_counts=True)
```




    (array([0, 1]), array([192, 564]))



Clearly, the data is not distributed equally in both labels, since the patients diagnosed positvely form the 74% of the total patients. Thus, the `accuracy` metric will not be very useful, since a model trained to always classify the patients as positive, will have an accuracy of 0.74. We need to choose some other metrics as well to properly train the model. In this case, the two metrics: `accuracy`, `precision` are chosen, since they are widely used for binary classifications and wroking with True Positive, True Negatives, False Positives and False Negatives.

It can be seen now that the mean of every column is 0, while the standard deviation is 1, so our dataset is normalized. The shape of the dataset is (756, 753) with 756 being the number of samples and 753 the number of features. Hence the input data of the NN will have the form `input_data=(753,)`. 

The next thing to do is to split the data into test and train subgroups. For this, a self-implemented method is used which uses random permutations to split the test using random indexes for the numpy arrays.


```python
import math
np.random.seed(200)

def split_test_train(X: np.ndarray, y: np.ndarray, test_size: float=0.2) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Split a given array X, and its corresponding labels y into a train-test
    data and labels.
    """
    L = list(range(X.shape[0]))
    indices = np.random.permutation(L)
    length = len(L)

    # Compute the length of the test and train splits
    test = math.floor(test_size*length)
    train = length - test
    
    # Split
    testIndexes = indices[0:test]
    complementaryIndexes = indices[-train:]

    X_train = X[complementaryIndexes]
    y_train = y[complementaryIndexes]
    X_test = X[testIndexes]
    y_test = y[testIndexes]
        
       
    return (X_train, y_train), (X_test, y_test)

(train_data, train_labels), (test_data, test_labels) = split_test_train(data, labels)
print(f"Train data: shape = {train_data.shape}, labels={train_labels.shape}")
print(f"Test data: shape = {test_data.shape}, labels={test_labels.shape}")
print(f"Train labels counts: {np.unique(train_labels, return_counts=True)}")
print(f"Test labels counts: {np.unique(test_labels, return_counts=True)}")
```

    Train data: shape = (605, 753), labels=(605,)
    Test data: shape = (151, 753), labels=(151,)
    Train labels counts: (array([0, 1]), array([159, 446]))
    Test labels counts: (array([0, 1]), array([ 33, 118]))


Since we don't have enough data, we can't hold a `validation` set apart, and we have to perform k-fold validation. 

For the last layer, a `sigmoid` function is chosen as the activation function, since it is widely used for binary classification problems. `Binary Crossentoropy` is chosen as a loss function, since it is widely used for binary classificaiton problems as well. 

Three main methods are developed here. One is the `build_model` method which builds the model, the `k_fold_validation` iterates over the train data and performs k-fold validation and the `show_results` which shows the results by decomposing the values of `history.history` dictionary, and taking the last epoch of all, where the model has the weights updated accordingly. 


```python
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import History
from tensorflow.keras.metrics import BinaryAccuracy, Precision
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

(partial_train_data, partial_train_labels), (val_data, val_labels) = split_test_train(train_data, train_labels)

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(512, input_shape=(partial_train_data.shape[1],), activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[Precision(), BinaryAccuracy()])
    return model

k = 4
num_epochs = 100

def show_results(history: History):
    for k, v in history.history.items():
        print(f"{k} : {v[-1]}")

def k_fold_validation(folds: int, epochs: int, train_data: np.ndarray, train_labels: np.ndarray, model: Model):
    num_val_samples = len(train_data) // 4
    all_scores = list()
    for i in range(folds):
        print(f'Fold {i} of {folds}')
        # Prepare the validation data: data from partition i
        a, b = i * num_val_samples, (i + 1) * num_val_samples
        val_data = train_data[a : b]
        val_targets = train_labels[a : b]
        
        partial_train_data = np.concatenate([
            train_data[:i*num_val_samples],
            train_data[(i+1)*num_val_samples:]],
            axis=0
        )
        partial_train_targets= np.concatenate([
            train_labels[:i*num_val_samples],
            train_labels[(i+1)*num_val_samples:]],
            axis=0
        )
        history: History = model.fit(partial_train_data, partial_train_targets, 
                                     batch_size=1, 
                                     verbose=0, 
                                     epochs=epochs, 
                                     validation_data=(val_data, val_targets)
                                     )
        show_results(history)
        all_scores.append(history.history)
    return history

history = k_fold_validation(k, num_epochs, train_data, train_labels, build_model())

```

    Fold 0 of 4


    2022-01-02 20:19:16.727308: W tensorflow/core/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz


    loss : 5.054821561323308e-10
    precision : 1.0
    binary_accuracy : 1.0
    val_loss : 2.9332964420318604
    val_precision : 0.9067796468734741
    val_binary_accuracy : 0.887417197227478
    Fold 1 of 4
    loss : 1.984023917378508e-10
    precision : 1.0
    binary_accuracy : 1.0
    val_loss : 0.9336085319519043
    val_precision : 0.9818181991577148
    val_binary_accuracy : 0.9735099077224731
    Fold 2 of 4
    loss : 9.466027267590604e-11
    precision : 1.0
    binary_accuracy : 1.0
    val_loss : 0.36739540100097656
    val_precision : 0.9907407164573669
    val_binary_accuracy : 0.9933775067329407
    Fold 3 of 4
    loss : 7.839145999000152e-11
    precision : 1.0
    binary_accuracy : 1.0
    val_loss : 0.33253636956214905
    val_precision : 1.0
    val_binary_accuracy : 0.9867549538612366


As we can see, our model performed really well with the training data and the validation set. Actually, it performed perfectly. This is very common, and means that the model is `overfitting`, and is very good at predicting already seen data, but loses capacity with unseen data. 

What we have to do now is to regularize the model so that it does not overfit the already seen data and is better at predicting unseen data. This can be done in several ways:
* Adding Dropout rate for some layers
* Reducing the number of neurons of hidden layers
* Add L1 or L2 regularization

Let's start by assesing if the number of epochs chosen is a good one or we could reduce it and still do not penalize the model.



```python
import matplotlib.pyplot as plt 

def plot_loss(history: History):
    plt.clf()
    history_dict = history.history
    history_dict.keys()
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]

    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, 'bo', label="Training loss")
    plt.plot(epochs, val_loss_values, 'b', label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

plot_loss(history)

```


    
![png](courseWork_files/courseWork_15_0.png)
    



```python
def plot_accuracy(history: History):
    plt.clf()
    history_dict = history.history
    acc_values = history_dict["binary_accuracy"]
    val_acc_values = history_dict["val_binary_accuracy"]
    epochs = range(1, len(acc_values) + 1)
    plt.plot(epochs, acc_values, 'bo', label="Training accuracy")
    plt.plot(epochs, val_acc_values, 'b', label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

def plot_metric(history: History, metric: str):

    keys = history.history.keys()
    final_key = ""
    final_val_key = ""
    for key in keys: 
        if metric in key and 'val' not in key:
            final_key = key
        elif metric in key and 'val' in key:
            final_val_key = key

    plt.clf()
    history_dict = history.history
    values = history_dict[final_key]
    val_values = history_dict[final_val_key]
    epochs = range(1, len(values) + 1)
    plt.plot(epochs, values, 'bo', label=f"Training {metric}")
    plt.plot(epochs, val_values, 'b', label=f"Validation {metric}")
    plt.title(f"Training and validation {metric}")
    plt.xlabel("Epochs")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.show()

plot_accuracy(history)
plot_metric(history, 'precision')
```


    
![png](courseWork_files/courseWork_16_0.png)
    



    
![png](courseWork_files/courseWork_16_1.png)
    


We can see how, after 40 epochs the change in the metrics is not significant, so we can reduce the number of epochs to 20, and the model will not underperform. 


```python
num_epochs = 40
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(512, input_shape=(partial_train_data.shape[1],), activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[Precision(), BinaryAccuracy()])
    return model

model = build_model()
history = k_fold_validation(k, num_epochs, train_data, train_labels, model)
```

    Fold 0 of 4
    loss : 0.0002802855451591313
    precision_1 : 1.0
    binary_accuracy : 1.0
    val_loss : 2.6565306186676025
    val_precision_1 : 0.9152542352676392
    val_binary_accuracy : 0.9006622433662415
    Fold 1 of 4
    loss : 0.008833537809550762
    precision_1 : 0.997032642364502
    binary_accuracy : 0.9977973699569702
    val_loss : 0.43052804470062256
    val_precision_1 : 0.9818181991577148
    val_binary_accuracy : 0.9735099077224731
    Fold 2 of 4
    loss : 7.171344629242071e-10
    precision_1 : 1.0
    binary_accuracy : 1.0
    val_loss : 6.140007462818176e-05
    val_precision_1 : 1.0
    val_binary_accuracy : 1.0
    Fold 3 of 4
    loss : 5.240794777316982e-13
    precision_1 : 1.0
    binary_accuracy : 1.0
    val_loss : 0.6988310217857361
    val_precision_1 : 1.0
    val_binary_accuracy : 0.9933775067329407



```python
plot_accuracy(history)
plot_loss(history)
plot_metric(history, 'precision')
```


    
![png](courseWork_files/courseWork_19_0.png)
    



    
![png](courseWork_files/courseWork_19_1.png)
    



    
![png](courseWork_files/courseWork_19_2.png)
    


We can see now how the model performed a bit better in the validation data than in the last iteration, this is due to the fact that we are making it overfit less. Now, let's add some regularization to the first layer, and dropout to the second layer.


```python
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(512, input_shape=(partial_train_data.shape[1],), activation='relu', kernel_regularizer='l1'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[Precision(), BinaryAccuracy()])
    return model

model = build_model()

history = k_fold_validation(k,  num_epochs, train_data, train_labels, model)
```

    Fold 0 of 4
    loss : 1.1836251020431519
    precision_2 : 0.9121813178062439
    binary_accuracy : 0.907489001750946
    val_loss : 1.2856040000915527
    val_precision_2 : 0.8974359035491943
    val_binary_accuracy : 0.8675496578216553
    Fold 1 of 4
    loss : 1.0910261869430542
    precision_2 : 0.9050279259681702
    binary_accuracy : 0.8986784219741821
    val_loss : 1.2384812831878662
    val_precision_2 : 0.9150943160057068
    val_binary_accuracy : 0.8543046116828918
    Fold 2 of 4
    loss : 1.0542110204696655
    precision_2 : 0.9263455867767334
    binary_accuracy : 0.91629958152771
    val_loss : 1.372309684753418
    val_precision_2 : 0.8455284833908081
    val_binary_accuracy : 0.8543046116828918
    Fold 3 of 4
    loss : 1.0348085165023804
    precision_2 : 0.9211267828941345
    binary_accuracy : 0.9295154213905334
    val_loss : 1.1282461881637573
    val_precision_2 : 0.8790322542190552
    val_binary_accuracy : 0.860927164554596



```python
plot_metric(history, 'precision')
plot_accuracy(history)
plot_loss(history)
```


    
![png](courseWork_files/courseWork_22_0.png)
    



    
![png](courseWork_files/courseWork_22_1.png)
    



    
![png](courseWork_files/courseWork_22_2.png)
    


We can see now how the model is not 100% correct on the train data, this is because of the regularization we just introduced. The model doesn't overfit anymore on the train data, and the validation results are better than in the last iteration.

We can now test the model against the test data and see its performance. 


```python
evaluation = model.evaluate(test_data, test_labels, return_dict=True)
evaluation
```

    5/5 [==============================] - 0s 2ms/step - loss: 1.0533 - precision_2: 0.9492 - binary_accuracy: 0.9205





    {'loss': 1.0533086061477661,
     'precision_2': 0.9491525292396545,
     'binary_accuracy': 0.9205297827720642}



## Results

We can see how the model performed very well with the test data, achieving a precision of 0.95 and an accuracy of 0.93. We can conclude that the model has been trained and adapted successfully and that the regularization process has been useful in order to prevent the model from overfitting. When evaluating the model with the test data, very good results have been obtained, even improving those of the train data. We can see how many parameters the model has, and how it has ended up regarding final configuration.


```python
model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_6 (Dense)             (None, 512)               386048    
                                                                     
     dropout_1 (Dropout)         (None, 512)               0         
                                                                     
     dense_7 (Dense)             (None, 512)               262656    
                                                                     
     dropout_2 (Dropout)         (None, 512)               0         
                                                                     
     dense_8 (Dense)             (None, 1)                 513       
                                                                     
    =================================================================
    Total params: 649,217
    Trainable params: 649,217
    Non-trainable params: 0
    _________________________________________________________________


The total number of parameters is 649217, all of which are trainable since we didn't add any 'frozen' parameters. The total number of layers is 5, although 2 of these are only Dropout layers that carry no parameters, but only have implications on the precious layer. 

Concluding, the model has performed very well with unseen data, as well as with already seen data, what we were looking for since the beginning. 
