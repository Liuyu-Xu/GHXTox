# Selective prediction operating points

All values are from 6,387 training out-of-fold predictions. Random values are means over 5,000 equal-coverage uniform random rankings.

| Method | Coverage | Accepted n | Error | MCC | BACC | SN | SP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen uncertainty ranking | 100% | 6387 | 0.0517 | 0.8721 | 0.9319 | 0.8938 | 0.9700 |
| Random rejection | 100% | 6387 | 0.0517 | 0.8721 | 0.9319 | 0.8938 | 0.9700 |
| Frozen uncertainty ranking | 95% | 6068 | 0.0366 | 0.9081 | 0.9503 | 0.9210 | 0.9797 |
| Random rejection | 95% | 6068 | 0.0517 | 0.8722 | 0.9319 | 0.8938 | 0.9700 |
| Frozen uncertainty ranking | 90% | 5748 | 0.0292 | 0.9257 | 0.9585 | 0.9316 | 0.9854 |
| Random rejection | 90% | 5748 | 0.0517 | 0.8721 | 0.9319 | 0.8938 | 0.9700 |
| Frozen uncertainty ranking | 85% | 5429 | 0.0241 | 0.9387 | 0.9649 | 0.9410 | 0.9889 |
| Random rejection | 85% | 5429 | 0.0517 | 0.8721 | 0.9319 | 0.8938 | 0.9700 |
| Frozen uncertainty ranking | 80% | 5110 | 0.0202 | 0.9490 | 0.9710 | 0.9514 | 0.9906 |
| Random rejection | 80% | 5110 | 0.0517 | 0.8722 | 0.9319 | 0.8938 | 0.9700 |
