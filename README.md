# Experiment Doc

## DCNN-64

Dose score: 2.7235 <br>
DVH score: 1.6311 <br>

## 2021.08.30 <br>

This is the best performance.<br>
Dose score: 2.6410,<br>
DVH score: 1.8329 <br>
work:

- **DVH曲线几乎不衰减**<br>
  reason:<br>
  it is because the original code denormalized the prediction dose again.<br>
  solution：<br>
  delete the de-normalization
- **code reformat for plot_DVH**

## 2021.08.31

the performance of config_1.yaml<br>
Dose score: 2.6377, improvement: + 0.0033<br>
DVH score: 1.8182, improvement: + 0.0417<br>