
# Benchmarcking results

This is a comparison of the benchmarking results for indexed prefix search with fast-search and streaming prefix search.
The benchmark tests were run with one query and a variable number of clients.
Both benchmarking runs used the same corpus containing 80 000 documents,
with the documents being one to two terms long.
The benchmarking was run with a duration of 60 seconds.

## Indexed prefix search with fast-search
time = 60s \
Query used: select * from sources term where term contains ({prefix:true}\"doc\")

| Clients  | Min        | Max        | Avg        | Q/s        | Request failed |
| :-------:|:----------:|:----------:|:----------:|:----------:|:--------------:|
| 1        | 1.75 ms    | 86.87 ms   | 2.14 ms    | 455.09     | 0%             |
| 5        | 1.59 ms    | 210.19 ms  | 2.23 ms    | 2233.44    | 0%             |
| 10       | 1.69 ms    | 115.18 ms  | 2.59 ms    | 3842.93    | 0%             |
| 25       | 1.88 ms    | 260.61 ms  | 4.89 ms    | 5104.91    | 0%             |
| 50       | 2.14 ms    | 493.97 ms  | 8.88 ms    | 5625.67    | 0%             |
| 75       | 2.81 ms    | 735.22 ms  | 13.87 ms   | 5403.19    | 0%             |
| 100      | 2.50 ms    | 994.26 ms  | 19.37 ms   | 5158.39    | 0%             |

With single digit clients we can see that indexed prefix search is able to have an average response time in the low 2 ms, we can also see that the query rate drastically goes up when more clients are used under the benchmarking. With 50 clients the indexed prefix search is still able to have good performance with the average response time being below 10 ms and having a query rate at mid 5000 Q/s. At 100 concurrent clients the query rate drops down to 5158.39 Q/s and increases to an average response time 19.37 ms. It's important to note that under all the benchmarkings zero of the query requests failed to give a result. This points to indexed prefix search being able to hadle and increase in concurrent clients.

## Streaming prefix search

time = 60s \
Query: select * from sources term where term contains “doc”

| Clients  | Min        | Max        | Avg        | Q/s        | Request failed |
| :-------:|:----------:|:----------:|:----------:|:----------:|:--------------:|
| 1        | 34.12 ms   | 75.20 ms   | 41.18 ms   | 24.27      | 0%             |
| 5        | 120.39 ms  | 317.01 ms  | 191.72 ms  | 26.08      | 0%             |
| 10       | 249.84 ms  | 1395.74 ms | 385.80 ms  | 25.92      | 0%             |
| 25       | 292.55 ms  | 771.40 ms  | 518.83 ms  | 48.18      | 90%            |
| 50       | 345.13 ms  | 991.34 ms  | 515.12 ms  | 97.05      | 93%            |
| 75       | 357.97 ms  | 1287.02 ms | 516.99 ms  | 145.07     | 95%            |
| 100      | 376.51 ms  | 1518.95 ms | 519.68 ms  | 192.42     | 89%            |

With streaming prefix search we can see that the performance has drastically decreased, the response time with 1 client is on average around 40 ms and the query rate is about 5.4% of the indexed prefix search under the same circumstances. This difference in average response time and query rate only increases as the number of clients goes up, as average response time drastically increases and the query rate minimally increases. It is important to note that when the number of clients increases there is a point where query requests start to fail as the server gets overloaded and starts returning status codes 504 and 503. That is also most likely the reason why we see the query rate increase as the requests start failing. It is not processing those requests and therefore returns a response faster, making it seem as the query rate goes up. The big increase in average response time and the increase in failed query requests as the number of clients increases points out that streaming prefix search is not sustainable when corpus size is large and the number of clients grows.   

One thing to note is that streaming search uses less memory as it is stored on disk, while the corpus for indexed prefix search has to be stored in memory at all times. This is because prefix search in index mode is only available for attributes and attributes are stored in memory.

## Use cases
Streaming prefix search is suitable if the corpus size is small and the number of concurrent users is also low. Such a case can for example be personal indexes where only the user's data is searched. You can see [reference](https://docs.vespa.ai/en/streaming-search.html) for more detailed information on streaming search

Indexed prefix search is suitable if the corpus is larger and needs to handle cases where there are several concurrent users. You can see [reference](https://docs.vespa.ai/en/attributes.html) for more on how attributes affect memory.
