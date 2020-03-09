library(ggplot2)

data <- read.csv("data/msmarco/train_test_set/training_data_weakAND_title_body_bert_collect_rank_features_embeddings_99_random_samples.csv")


ggplot(data, aes(x=bm25.title. + bm25.body., y = stat(count) / sum(count))) + 
  geom_histogram(fill = "blue", bins=100, alpha=0.5) + 
  geom_histogram(data = subset(data, relevant == 1), fill = "red", bins=100, alpha = 0.5) +
  labs(x = "bm25(title) + bm25(body)", y = "Document frequency")

ggplot(data, aes(x=rankingExpression.dot_product_title_bert. + rankingExpression.dot_product_body_bert., y = stat(count) / sum(count))) + 
  geom_histogram(fill = "blue", bins=100, alpha=0.5) + 
  geom_histogram(data = subset(data, relevant == 1), fill = "red", bins=100, alpha = 0.5) +
  labs(x = "dot-product(title, query) + dot-product(body, query)", y = "Document frequency")

ggplot(data, aes(x=rankingExpression.dot_product_title_bert. + rankingExpression.dot_product_body_bert., 
                 y=bm25.title. + bm25.body.)) + 
  geom_point(alpha = 0.2) + 
  geom_point(data = subset(data, relevant == 1), colour = "red") + 
  labs(x = "dot-product(title, query) + dot-product(body, query)", y = "bm25(title) + bm25(body)")



ggplot(data) + geom_histogram(aes(x=rankingExpression.dot_product_title_bert.), fill = "white", colour="black")
ggplot(data) + geom_histogram(aes(x=rankingExpression.dot_product_body_bert.), fill = "white", colour="black")

ggplot(data) + geom_histogram(aes(x=bm25.title.), fill = "white", colour="black", bins=100) 

  
ggplot(data, aes(x=bm25.body., y = stat(count) / sum(count))) + 
  geom_histogram(fill = "blue", bins=100, alpha=0.5) + 
  geom_histogram(data = subset(data, relevant == 1), fill = "red", bins=100, alpha = 0.5)

ggplot(data, aes(x=bm25.title., y = stat(count) / sum(count))) + 
  geom_histogram(fill = "blue", bins=100, alpha=0.5) + 
  geom_histogram(data = subset(data, relevant == 1), fill = "red", bins=100, alpha = 0.5)


ggplot(data, aes(x=rankingExpression.dot_product_title_bert., y = stat(count) / sum(count))) + 
  geom_histogram(fill = "blue", bins=100, alpha=0.5) + 
  geom_histogram(data = subset(data, relevant == 1), fill = "red", bins=100, alpha = 0.5)

ggplot(data, aes(x=rankingExpression.dot_product_body_bert., y = stat(count) / sum(count))) + 
  geom_histogram(fill = "blue", bins=100, alpha=0.5) + 
  geom_histogram(data = subset(data, relevant == 1), fill = "red", bins=100, alpha = 0.5)


ggplot(subset(data, bm25.title. > 0)) + 
  geom_point(aes(x=rankingExpression.dot_product_title_bert., y=bm25.title.), alpha = 0.2) + 
  geom_point(data = subset(data, relevant == 1), aes(x=rankingExpression.dot_product_title_bert., y=bm25.title.), colour = "red")

ggplot(subset(data, bm25.body. > 0)) + geom_point(aes(x=rankingExpression.dot_product_body_bert., y=bm25.body.), alpha = 0.2)

ggplot(data) + 
  geom_point(aes(x=bm25.title., 
                 y=bm25.body.), alpha = 0.2) +
  geom_point(data = subset(data, relevant == 1), aes(x=bm25.title., 
                                                     y=bm25.body.), colour = "red", alpha = 0.2)
