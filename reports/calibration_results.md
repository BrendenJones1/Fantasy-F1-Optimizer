Prediction-interval coverage for Ridge on 90 races (1834 driver-race rows). Each method is refit before every race on earlier races only, after 20 warm-up races. Coverage is the mean over races with a 95% bootstrap CI resampling races; width and interval score are in Fantasy points (lower is better).

|   level | method           | coverage             |   width |   interval_score |
|--------:|:-----------------|:---------------------|--------:|-----------------:|
|     0.5 | Binned quantiles | 0.478 [0.444, 0.511] |    10.8 |             29.5 |
|     0.5 | Gaussian         | 0.600 [0.566, 0.633] |    16   |             30.4 |
|     0.5 | Pooled quantiles | 0.485 [0.452, 0.519] |    10.7 |             29.9 |
|     0.8 | Binned quantiles | 0.782 [0.756, 0.807] |    32.6 |             43.3 |
|     0.8 | Gaussian         | 0.790 [0.764, 0.815] |    30.5 |             45.6 |
|     0.8 | Pooled quantiles | 0.794 [0.771, 0.815] |    30.9 |             44   |
|     0.9 | Binned quantiles | 0.876 [0.858, 0.894] |    38.8 |             49.3 |
|     0.9 | Gaussian         | 0.873 [0.854, 0.890] |    39.1 |             54.7 |
|     0.9 | Pooled quantiles | 0.895 [0.877, 0.911] |    39.7 |             52.9 |
