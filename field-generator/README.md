Feeding 100 docs benchmark:

CPU
1 parallel - 8 connections - 3723 seconds
5 parallel - 8 connections - 1871 seconds
10 parallel - 8 connections - 2077 seconds

GPU
1 parallel - 8 connections - 275 - some errors
1 parallel - 1 connections - 234
5 parallel - 1 connection - 156 seconds
5 parallel - 8 connections - 131 second
5 parallel - 3 connections - 133 seconds
10 parallel - 8 connection - 140 seconds

2 GPU
2 connection - 85 seconds
3 conn  - 75 seconds
4 connection - 70 seconds

3 GPU
8 connection - 46 seconds

4 GPU
12 connection - 34 seconds
