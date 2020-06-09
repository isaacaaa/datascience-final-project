library(xgboost)
library(argparser, quietly=TRUE)
library(Matrix)
library(caret)
# library(FeatureSelection)
# data(agaricus.train, package='xgboost')
p <- arg_parser("Process input and output csv")
p <- add_argument(p, "--train",help="iutput result csv file")
p <- add_argument(p, "--test", help="output result csv file", default="test.csv")
args <- parse_args(p, commandArgs(trailingOnly = TRUE))
train <- read.csv('train.csv')
test <- read.csv(args$test, header=T)
# train
for(i in 1:ncol(train)){
  train[is.na(train[,i]), i] <- mean(train[,i], na.rm = TRUE)
}
train$slope_label <- sapply(train$slope, function(x) {ifelse(x==1, 1,0)})
test$slope_label <- sapply(test$slope, function(x) {ifelse(x==1, 1,0)})
# train
# train<-train[,-8]
# train<-train[,-9]
# test<-test[,-8]
# test<-test[,-9]
# iris2 = iris
# head(train)
# print(typeof(train))
# train=read_csv('train.csv')
# train
# test
# 增加一個 factor 變數當作解釋變數, 否則 iris 資料檔的
# 解釋變數都是數值變數, 不符合實務上常見的資料形態
# manually add a categorical variable as one of the features
# to reflect real-world situation.
# iris2$id = factor(sample(c("A","B"),150,replace=T))
 
# Use sparse matrix in Matrix package

# 把所有解釋變數都轉為矩陣型態
# convert all feature variables to a sparse matrix
# xdata = sparse.model.matrix(heart_disease ~ .-1, data = train)
#  xdata
# Species 分類數目
# number of categories in response variable
# m =2

# 把目標變數 Species 三個分類轉成 0,1,2. 必須從 0 開始
# recode Y as 0,1,2,...,m-1
# Y = as.integer(train$heart_disease) - 1
#  print(Y) 
# set random seed
# set.seed(12345)
 
# xgboost 參數設定 (xgboost parameters setup)
# param = list("objective" = "multi:softprob",
# "eval_metric" = "mlogloss",
# "num_class" = m
# )
dtrain<-train[1:200,]
dtrain<-train[1:200,]
dtest<-train[201:228,]
# print(train[1:11,])
dtrain$heart_disease <- NULL
# head(Data)
# train
#  train$heart_disease
# vec_y = sparse.model.matrix(train$heart_disease)
#  mat_y
# build the model
# train1=as.matrix(train)
# print(typeof(train1))
# Y=train1$heart_disease
# print(typeof(Y))
params <- list(booster = "gbtree", objective = "binary:logistic", 
eta=0.3, gamma=0, max_depth=5, min_child_weight=1, subsample=1, colsample_bytree=1)
dtest$heart_disease<-NULL
dtrain<-as.matrix(dtrain)
dtest<-as.matrix(dtest)
id<-test$id
test<-as.matrix(test)
dtrain <- xgb.DMatrix(dtrain, label =train[1:200,]$heart_disease)
bst = xgboost(data=dtrain, max_depth=5, eta=1,nfold=5, objective='binary:logistic', nrounds=5)

# print(names(train[,15]))
imp = xgb.importance(names(train[,-15]),model=bst)
print(imp)
# dtest <- xgb.DMatrix(agaricus.test$data, label = agaricus.test$label)
# bst
# test
# 計算預測值 (get prediction)
dev_pred = predict(bst,dtest)
# heart_disease
# train[201:228,15]
# confusionMatrix(table(Argument 1, Argument 2)) 
table(dev_pred,train[201:228,15])

heart_disease = predict(bst,test)
# table[]   
# heart_disease<-data.frame(heart_disease)
# print(typeof(heart_disease))

# print(test$id)
# df = data.frame(id,heart_disease)
# df
# df$heart_disease <- ifelse(df$heart_disease>=0.5, 1, 0)
# df
# write.csv(df, "predict.csv", row.names=FALSE)


# train with feature selection

train<-train[,-8]
# train<-train[,-9]
# train<-train[,-8]
test<-test[,-8]
# test<-test[,-9]
# test<-test[,-8]
dtrain<-train[1:200,]

dtest<-train[201:228,]
# print(train[1:11,])
dtrain$heart_disease <- NULL

dtest$heart_disease<-NULL
dtrain<-as.matrix(dtrain)
dtest<-as.matrix(dtest)
id<-test[,1]
test<-as.matrix(test)
dtrain <- xgb.DMatrix(dtrain, label =train[1:200,]$heart_disease)

xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, nfold = 5, showsd = T, stratified = T, 
print.every.n = 10, early.stop.round = 20, maximize = F)
bst = xgboost(data=dtrain, max_depth=5, eta=1,nfold=5, objective='binary:logistic', nrounds=5)
# imp = xgb.importance(names(train[,-15]),model=bst)
# print(imp)

# dev_pred = predict(bst,dtest)

# table(dev_pred,train[201:228,15])

heart_disease = predict(bst,test)

df = data.frame(id,heart_disease)
# df
df$heart_disease <- ifelse(df$heart_disease>=0.5, 1, 0)
# df
write.csv(df, "predict.csv", row.names=FALSE)










































































# Ypred = t(matrix(Ypred,m,length(Ypred)/m))
# # colnames(Ypred) = levels(iris2$Species)
# Ypred = levels(train$heart_disease)[max.col(Ypred)]
# Ypred = factor(Ypred,levels=levels(train$heart_disease))
 
# # 混淆矩陣 (confusion matrix)
# t0 = table(iris2$Species,Ypred)
# t0
 
# 預測正確率 (accuracy)
# sum(diag(t0))/sum(t0)
 
# 解釋變數重要性 (variable importance)
# imp = xgb.importance(names(iris2[,-5]),model=result)
# print(imp)
 
# library(Ckmeans.1d.dp)
# xgb.plot.importance(imp)
# library(DiagrammeR)
# xgb.plot.tree(feature_names=names(iris[,-5]),model=result, n_first_tree=2)

