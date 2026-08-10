# Selective prediction operating points

All values are from 6,387 training out-of-fold predictions. Random values are means over 5,000 equal-coverage uniform random rankings.

| Method | Coverage | Accepted n | Error | MCC | BACC | SN | SP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen uncertainty ranking | 100% | 6387 | 0.0512 | 0.8733 | 0.9324 | 0.8944 | 0.9705 |
| Random rejection | 100% | 6387 | 0.0512 | 0.8733 | 0.9324 | 0.8944 | 0.9705 |
| Frozen uncertainty ranking | 95% | 6068 | 0.0366 | 0.9082 | 0.9505 | 0.9216 | 0.9795 |
| Random rejection | 95% | 6068 | 0.0512 | 0.8733 | 0.9324 | 0.8944 | 0.9705 |
| Frozen uncertainty ranking | 90% | 5748 | 0.0292 | 0.9257 | 0.9585 | 0.9315 | 0.9854 |
| Random rejection | 90% | 5748 | 0.0512 | 0.8733 | 0.9324 | 0.8943 | 0.9705 |
| Frozen uncertainty ranking | 85% | 5429 | 0.0241 | 0.9387 | 0.9650 | 0.9411 | 0.9889 |
| Random rejection | 85% | 5429 | 0.0512 | 0.8733 | 0.9324 | 0.8944 | 0.9705 |
| Frozen uncertainty ranking | 80% | 5110 | 0.0205 | 0.9481 | 0.9705 | 0.9508 | 0.9903 |
| Random rejection | 80% | 5110 | 0.0512 | 0.8733 | 0.9324 | 0.8944 | 0.9705 |
