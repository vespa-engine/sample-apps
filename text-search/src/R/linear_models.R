library(ggplot2)
library(reshape2)
library(arm)

#
# Load data
#
data <- read.csv("data/training_data_collectAll.csv")
drops <- c("docid", 
           "qid", 
           "relevant", 
           "rankingExpression.averagefieldTermMatchBodyFirstPosition.", 
           "rankingExpression.averagefieldTermMatchTitleFirstPosition.",
           "rankingExpression.maxfieldTermMatchBodyFirstPosition.",
           "rankingExpression.maxfieldTermMatchTitleFirstPosition.",
           "rankingExpression.averagefieldTermMatchBodyOcurrences.",
           "rankingExpression.maxfieldTermMatchBodyOcurrences.",
           "rankingExpression.maxfieldTermMatchTitleOcurrences.",
           "rankingExpression.averagefieldTermMatchTitleOcurrences.",
           "elementCompleteness.body..elementWeight",
           "elementCompleteness.title..elementWeight",
           "matches.body.",
           "matches.title."
)

#
# Select only features to use
#
features <- data[, !(names(data) %in% drops)]
target <- "relevant"

#
# Lasso
#
library(glmnet)

set.seed(1)
idx <- sample(x = 1:nrow(features), nrow(features), replace = FALSE)
x=model.matrix(relevant~.,cbind(data.frame(relevant = data$relevant), features))[,-1]
x <- x[idx,]
y <- data$relevant[idx]

#grid=10^seq(10,-2,length=100)
lasso.mod=cv.glmnet(x,y,alpha=1)
plot(lasso.mod)
lasso.mod$lambda.min
#
# Basic linear models
#
r = glm(formula = as.formula(paste(target, paste(names(features), collapse=" + "), sep=" ~ ")), family = binomial(), data = data)
r_b = bayesglm(formula = as.formula(paste(target, paste(names(features), collapse=" + "), sep=" ~ ")), family = "binomial", data = data)

coefs <- r_b$coefficients
variables <- names(coefs)
expression <- as.character(coefs[1])
for (i in 2:length(r_b$coefficients)){
  expression <- paste(expression, paste(coefs[i], variables[i], sep = " * "), sep = " + ")
}

r = glm(formula = relevant ~ 1 + bm25.body. + bm25.title., family = binomial(), data = data)

logistic <- function(x){
  return(1/(1+exp(-x)))
}
# data <- data %>% mutate(bm25.body. = scale(bm25.body.),
#                         bm25.title. = scale(bm25.title.),
#                         matchCount.body. = scale(matchCount.body.),
#                         matchCount.title. = scale(matchCount.title.))

drops <- c("docid", 
           "qid", 
           "relevant", 
           "rankingExpression.averagefieldTermMatchBodyFirstPosition.", 
           "rankingExpression.averagefieldTermMatchTitleFirstPosition.",
           "rankingExpression.maxfieldTermMatchBodyFirstPosition.",
           "rankingExpression.maxfieldTermMatchTitleFirstPosition.",
           "rankingExpression.averagefieldTermMatchBodyOcurrences.",
           "rankingExpression.maxfieldTermMatchBodyOcurrences.",
           "rankingExpression.maxfieldTermMatchTitleOcurrences.",
           "rankingExpression.averagefieldTermMatchTitleOcurrences.",
           "elementCompleteness.body..elementWeight",
           "elementCompleteness.title..elementWeight",
           "matches.body.",
           "matches.title."
           )

features <- data[, !(names(data) %in% drops)]

# cleared <- c("bm25.body.",
#              "bm25.title.",
#              "matchCount.body.",
#              "matchCount.title.")
# 
# features <- features[, !(names(features) %in% cleared)]

# features <- features[, names(features)[61:]]
             
melted <- melt(data = features)

ggplot(melted) + geom_boxplot(aes(x = variable, y = value))



as.character(unique(melted[melted["value"] > 12, "variable"]))