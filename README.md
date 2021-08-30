# Experiment Doc

## 2021.08.30 <br>

Dose score: 2.6410,<br>
DVH score: 1.8329 <br>
work:

- **DVH曲线几乎不衰减**<br>
  原因:<br>
  it is because the original code denormalized the prediction dose again.<br>
  解决方案：<br>
  delete the de-normalization
- **code reformat for plot_DVH**