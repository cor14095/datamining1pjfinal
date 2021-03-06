# Proyecto final 
# Grupo 3
# Estuardo Umana
# Mario Morales 
# Alejandro Cortes

# Librerias
library(readr)

library(dplyr)
library(ggplot2)

library(rpart)
library(rpart.plot)

library(corrplot)

library(vcd)
library(vcdExtra)

library(caret)
library(weights)
library(e1071)
library(pROC)

library(randomForest)

library(tidyverse)
library(hrbrthemes)
library(viridis)

# -----------------------------------------------------------------------------
# |                              Cargar data                                  |
# -----------------------------------------------------------------------------
trainInsure <- read_csv("Data/train.csv", col_types = cols(id = col_number(), 
                                                   Age = col_number(), 
                                                   Driving_License = col_number(), 
                                                   Region_Code = col_number(), 
                                                   Previously_Insured = col_number(), 
                                                   Annual_Premium = col_number(), 
                                                   Policy_Sales_Channel = col_number(), 
                                                   Vintage = col_number(),
                                                   Response = col_number()
                                                   )
                 )

# -----------------------------------------------------------------------------
# |                            Exploracion datos                              |
# -----------------------------------------------------------------------------
View(trainInsure)

# Explorando un par de datos
head(trainInsure)
summary(trainInsure)

# Exploremos un poco cada variable para encontrar nulls/nas
table(trainInsure$id)
table(trainInsure$Gender)
table(trainInsure$Age)
table(trainInsure$Driving_License)
table(trainInsure$Region_Code)
table(trainInsure$Previously_Insured)
table(trainInsure$Vehicle_Age)
table(trainInsure$Vehicle_Damage)
table(trainInsure$Annual_Premium)
table(trainInsure$Policy_Sales_Channel)
table(trainInsure$Vintage)
table(trainInsure$Response)

# Estamos libres de pecado, amen /\
for (i in trainInsure) {
  View(trainInsure[is.na(i),])
}

# -----------------------------------------------------------------------------
# |                            Dummificar datos                               |
# -----------------------------------------------------------------------------

trainInsure$GenderDummy =  ifelse(trainInsure$Gender == "Male", 1 , 0)

#   < 1 Year    0
#   1-2 Year    1 
#   > 2 Years   2
trainInsure$Vehicle_AgeDummy =  ifelse (trainInsure$Vehicle_Age == "< 1 Year",0, ifelse(trainInsure$Vehicle_Age == "> 2 Years", 2 , 1))

trainInsure$VehiculeDamage_Dummy =  ifelse(trainInsure$Vehicle_Damage == "Yes", 1 , 0)

# Subset trainInsure
trainSinCategorico = trainInsure %>% select(
  - id,
  - Gender,
  - Vehicle_Age,
  - Vehicle_Damage
)

# -----------------------------------------------------------------------------
# |                         Matriz de correlacion                             |
# -----------------------------------------------------------------------------
matriz_cor = cor(trainSinCategorico)
corrplot(matriz_cor)

rm(matriz_cor)
rm(trainSinCategorico)

# Preparamos el set
trainInsure$Gender <- as.factor(trainInsure$Gender)
trainInsure$Vehicle_Age <- as.factor(trainInsure$Vehicle_Age)
trainInsure$Vehicle_Damage <- as.factor(trainInsure$Vehicle_Damage)
trainInsure$Previously_Insured<- as.factor(trainInsure$Previously_Insured)
# trainInsure$Response <- as.factor(trainInsure$Response)
trainInsure$Region_Code <- as.factor(trainInsure$Region_Code)
trainInsure$Driving_License <- as.factor(trainInsure$Driving_License)

# Como se divide el valor de Response
table(trainInsure$Response)
# Respuesta positiva
(46710/(334399+46710)) * 100
# Respuesta negativa
(334399/(334399+46710)) * 100

# Revisamos todas las variables
variables <- c(
  trainInsure[2],
  trainInsure[4],
  trainInsure[6],
  trainInsure[7],
  trainInsure[8]
  )

# -----------------------------------------------------------------------------
# |                                  Graficas                                 |
# -----------------------------------------------------------------------------
varNameCount <- 1
varPos <- c(2,4,6,7,8)
for (i in variables) {
  mosaicplot(
    Response ~ i,
    data = trainInsure,
    ylab = names(trainInsure)[varPos[varNameCount]],
    color = trainInsure$Response,
    shade = TRUE
  )
  varNameCount <- varNameCount + 1
}

# Revisamos variables cuantitativas
trainInsure %>% 
  ggplot() +
  aes(x = Gender, y = Age, color = Response) +
  geom_boxplot() +
  ggtitle("Por Genero y Edad")

trainInsure %>% 
  ggplot() +
  aes(x = Age, y = Annual_Premium, color = Response) +
  geom_histogram(stat='identity') +
  ggtitle("Por Edad y Monto acumulado de poliza")

trainInsure %>% 
  ggplot() +
  aes(x = Previously_Insured, y = as.numeric(Response), color = Response) +
  geom_bar(stat='identity') +
  ggtitle("Previamente asegurados acumulado por respuesta") +
  scale_y_continuous(breaks = seq(0, 300000, 50000), 
                     limits=c(0, 300000))

# -----------------------------------------------------------------------------
# |                       Tablas de contingencias                             |
# -----------------------------------------------------------------------------
# Contingencia de genero
genderTable <- table(trainInsure$Gender, trainInsure$Response)
genderTable

rowSums(genderTable)
colSums(genderTable)

rm(genderTable)

# Contingencia por edad del vehiculo
vehicleAgeTable <- table(trainInsure$Vehicle_Age, trainInsure$Response)
vehicleAgeTable

rowSums(vehicleAgeTable)
colSums(vehicleAgeTable)

rm(vehicleAgeTable)

# Contingencia por danio previo al vehiculo
vehicleDamageTable <- table(trainInsure$Vehicle_Damage, trainInsure$Response)
vehicleDamageTable

rowSums(vehicleDamageTable)
colSums(vehicleDamageTable)

rm(vehicleDamageTable)

# Contingencia por canal de comunicacion
channelTable <- table(trainInsure$Policy_Sales_Channel, trainInsure$Response)
channelTable <- cbind(channelTable, (((channelTable[,1] - channelTable[,2])/ (channelTable[,1] + channelTable[,2])) - 1) * -100)
channelTable

rowSums(channelTable)
colSums(channelTable)

rm(channelTable)

# Contingencia por previo asegurado
prevEnsureTable <- table(trainInsure$Previously_Insured, trainInsure$Response)
prevEnsureTable

rowSums(prevEnsureTable)
colSums(prevEnsureTable)

rm(prevEnsureTable)

# Contingencia por previo asegurado
ageTable <- table(trainInsure$Age, trainInsure$Response)
ageTable <- cbind(ageTable, (((ageTable[,1] - ageTable[,2])/ (ageTable[,1] + ageTable[,2])) - 1) * -100)
ageTable

rowSums(ageTable)
colSums(ageTable)

rm(ageTable)

# -----------------------------------------------------------------------------
# |                                  Hipotesis                                |
# -----------------------------------------------------------------------------
# Las variables de Age, Previoulsy Ensured, Sale Channel, Vehicle Age, Vehicle Damage
# y Gender tienen una relacion que explica la variable Response

# -----------------------------------------------------------------------------
# |                           PREGUNTAS A RESONDER PARA LA HIPOTESIS                               |
# -----------------------------------------------------------------------------

trainInsure %>% 
  ggplot() +
  aes(x = as.factor(Response), y = Age, fill=Response) +
  geom_boxplot() +
  theme_ipsum() +
  theme(
    legend.position="none",
    plot.title = element_text(size=11)
  ) +
  ggtitle("Edad x Response")


mosaicplot(~as.factor(Response) + Previously_Insured, 
           data = trainInsure, 
           color = 2:7, 
           las = 1) 

mosaicplot(~as.factor(Response) + Gender, 
           data = trainInsure, 
           color = 2:14, 
           las = 1) 

# -----------------------------------------------------------------------------
# |                         Generacion Train y test                           |
# -----------------------------------------------------------------------------
# Filas para trainInsure y test (utilizando paquete de CARET)
inTrain <- createDataPartition(y = trainInsure$Response, p = 0.7, list = FALSE)
test <- trainInsure[-inTrain,]
train <- trainInsure[inTrain,]

# Es necesario hacer una particion especial para RF, LR y DT
# Primero sacamos un DF de solo positivas y solo negativas
inTrainTemp1 <- trainInsure %>% filter(Response == 1)
inTrainTemp0 <- trainInsure %>% filter(Response == 0)

# reducimos la proporcion de las negativas a la misma de las positivas
inTrainTemp0_sort = sort(sample(nrow(inTrainTemp0), nrow(inTrainTemp0)*.14))
# Sacamos el 14% de las negativas que equivale a la misma cantidad de las positivas
inTrainTemp0_reduced <- inTrainTemp0[inTrainTemp0_sort,]

# Unimos ambos DFs
newTrain <- rbind(inTrainTemp1, inTrainTemp0_reduced)

# La proporcion de NewTrain es ~50% para Response
table(newTrain$Response)

rm(trainInsure)
rm(inTrain)
rm(variables)
rm(i)
rm(varNameCount)
rm(varPos)
rm(inTrainTemp1)
rm(inTrainTemp0)
rm(inTrainTemp0_sort)
rm(inTrainTemp0_reduced)

# -----------------------------------------------------------------------------
# |                             Naive Bayes                                   |
# -----------------------------------------------------------------------------

# --------------------------------- Hip 1 -------------------------------------
modeloBayes <- naiveBayes(Response ~ ., data = train)
predNB_Test_1<-predict(modeloBayes, newdata=test, type="raw")

predictionNB_Test_1 <- as.data.frame(predNB_Test_1)
predictionNB_Test_1$Result <- ifelse(predictionNB_Test_1[,1]>predictionNB_Test_1[,2],0,1)
predictionNB_Test_1$ResultChar = ifelse(predictionNB_Test_1$Result==1,"SI","NO")
resultsNB_Test_1 <- table(test$Response, predictionNB_Test_1$ResultChar)
resultsNB_Test_1

# Acurracy
accuracyNB_Test_1 <- sum(diag(resultsNB_Test_1))/sum(resultsNB_Test_1)
accuracyNB_Test_1

# Recall
recall_Test_1 <- (resultsNB_Test_1[2,2]/(resultsNB_Test_1[2,1]+resultsNB_Test_1[2,2]))
recall_Test_1

# Precision
precision_Test_1 <- (resultsNB_Test_1[2,2]/(resultsNB_Test_1[1,2]+resultsNB_Test_1[2,2]))
precision_Test_1

# Clean Memory
rm(modeloBayes)
rm(predictionNB_Test_1)
rm(predNB_Test_1)
# -------------------------------- Fin Hip 1 ----------------------------------

# --------------------------------- Hip 2 -------------------------------------
modeloBayes2 <- naiveBayes(Response ~ Age + Gender + Previously_Insured +
                             Vehicle_Age + Vehicle_Damage + Policy_Sales_Channel, 
                           method="class", 
                           data=train
                           )

predNB_Test_2 <- predict(modeloBayes2, newdata=test, type="raw")

predictionNB_Test_2 <- as.data.frame(predNB_Test_2)
predictionNB_Test_2$Result <- ifelse(predictionNB_Test_2[,1]>predictionNB_Test_2[,2],0,1)
predictionNB_Test_2$ResultChar = ifelse(predictionNB_Test_2$Result==1,"SI","NO")
resultsNB_Test_2 <- table(test$Response, predictionNB_Test_2$ResultChar)
resultsNB_Test_2

# Acurracy
accuracyNB_Test_2 <- sum(diag(resultsNB_Test_2))/sum(resultsNB_Test_2)
accuracyNB_Test_2

# Recall
recall_Test_2<-(resultsNB_Test_2[2,2]/(resultsNB_Test_2[2,1]+resultsNB_Test_2[2,2]))
recall_Test_2

# Precision
precision_Test_2<-(resultsNB_Test_2[2,2]/(resultsNB_Test_2[1,2]+resultsNB_Test_2[2,2]))
precision_Test_2

# Clean Memory
rm(modeloBayes2)
rm(predictionNB_Test_2)
rm(predNB_Test_2)
# -------------------------------- Fin Hip 2 ----------------------------------

# -----------------------------------------------------------------------------
# |                           Decision Tree                                   |
# -----------------------------------------------------------------------------

# --------------------------------- Hip 1 -------------------------------------
modeloDT<-rpart(Response ~ ., method="class", data=newTrain)
prp(modeloDT) #gráfica básica 
rpart.plot(modeloDT) #Gráfica recomendada

predictionDT_Test_1<-predict(modeloDT, newdata=test, type="class")
predictionDT_Test_1 = as.character(predictionDT_Test_1)
predictionDT_Test_1 = ifelse(predictionDT_Test_1 == 1, "SI", "NO")
resultsDT_Test_1 <- table(test$Response, predictionDT_Test_1)
resultsDT_Test_1
prop.table(resultsDT_Test_1)

# Acurracy
accuracyDT_Test_3<-sum(diag(resultsDT_Test_1))/sum(resultsDT_Test_1)
accuracyDT_Test_3

# Recall
recall_Test_3<-(resultsDT_Test_1[2,2]/(resultsDT_Test_1[2,1]+resultsDT_Test_1[2,2]))
recall_Test_3

# Precision
precision_Test_3<-(resultsDT_Test_1[2,2]/(resultsDT_Test_1[1,2]+resultsDT_Test_1[2,2]))
precision_Test_3

# Clean Memory
rm(modeloDT)
rm(predictionDT_Test_1)
rm(resultsDT_Test_1)
# -------------------------------- Fin Hip 1 ----------------------------------

# --------------------------------- Hip 2 -------------------------------------
modeloDT2<-rpart(Response ~ Age + Gender + Previously_Insured +
                  Vehicle_Age + Vehicle_Damage + Policy_Sales_Channel, 
                method="class", data=newTrain)
prp(modeloDT2) #gráfica básica 
rpart.plot(modeloDT2) #Gráfica recomendada

predictionDT_Test_2<-predict(modeloDT2, newdata=test, type="class")
predictionDT_Test_2 = as.character(predictionDT_Test_2)
predictionDT_Test_2 = ifelse(predictionDT_Test_2 == 1, "SI", "NO")
resultsDT_Test_2 <- table(test$Response, predictionDT_Test_2)
resultsDT_Test_2
prop.table(resultsDT_Test_2)

# Acurracy
accuracyDT_Test_4<-sum(diag(resultsDT_Test_2))/sum(resultsDT_Test_2)
accuracyDT_Test_4

# Recall
recall_Test_4<-(resultsDT_Test_2[2,2]/(resultsDT_Test_2[2,1]+resultsDT_Test_2[2,2]))
recall_Test_4

# Precision
precision_Test_4<-(resultsDT_Test_2[2,2]/(resultsDT_Test_2[1,2]+resultsDT_Test_2[2,2]))
precision_Test_4

# Clean Memory
rm(modeloDT2)
rm(predictionDT_Test_2)
rm(resultsDT_Test_2)
# -------------------------------- Fin Hip 2 ----------------------------------

# -----------------------------------------------------------------------------
# |                              Random Forest                                |
# -----------------------------------------------------------------------------

# --------------------------------- Hip 1 -------------------------------------
modeloRF <- randomForest(Response ~ ., data=newTrain, importance = F, type = "class")
modeloRF

predictionRF_Test_1 <- predict(modeloRF, newdata = test, type = "class")
predictionRF_Test_1 = ifelse(predictionRF_Test_1 > 0.5, "SI", "NO")
resultsRF_Test_1<-table(test$Response, predictionRF_Test_1)
resultsRF_Test_1
prop.table(resultsRF_Test_1)

# Acurracy
accuracyRF_Test_5<-sum(diag(resultsRF_Test_1))/sum(resultsRF_Test_1)
accuracyRF_Test_5

# Recall
recall_Test_5<-(resultsRF_Test_1[2,2]/(resultsRF_Test_1[2,1]+resultsRF_Test_1[2,2]))
recall_Test_5

# Precision
precision_Test_5<-(resultsRF_Test_1[2,2]/(resultsRF_Test_1[1,2]+resultsRF_Test_1[2,2]))
precision_Test_5

# Clean Memory
rm(modeloRF)
rm(predictionRF_Test_1)
rm(resultsRF_Test_1)
# -------------------------------- Fin Hip 1 ----------------------------------

# --------------------------------- Hip 2 -------------------------------------
modeloRF2 <- randomForest(Response ~ Age + Gender + Previously_Insured +
                            Vehicle_Age + Vehicle_Damage + Policy_Sales_Channel, 
                          data=newTrain, importance = F, type = "class")
modeloRF2

predictionRF_Test_2 <- predict(modeloRF2, newdata = test, type = "class")
predictionRF_Test_2 = ifelse(predictionRF_Test_2 > 0.5, "SI", "NO")
resultsRF_Test_2<-table(test$Response, predictionRF_Test_2)
resultsRF_Test_2
prop.table(resultsRF_Test_2)

# Acurracy
accuracyRF_Test_6<-sum(diag(resultsRF_Test_2))/sum(resultsRF_Test_2)
accuracyRF_Test_6

# Recall
recall_Test_6<-(resultsRF_Test_2[2,2]/(resultsRF_Test_2[2,1]+resultsRF_Test_2[2,2]))
recall_Test_6

# Precision
precision_Test_6<-(resultsRF_Test_2[2,2]/(resultsRF_Test_2[1,2]+resultsRF_Test_2[2,2]))
precision_Test_6

# Clean Memory
rm(modeloRF2)
rm(predictionRF_Test_2)
rm(resultsRF_Test_2)
# -------------------------------- Fin Hip 2 ----------------------------------

# -----------------------------------------------------------------------------
# |                            Logic Regression                               |
# -----------------------------------------------------------------------------

# --------------------------------- Hip 1 -------------------------------------
modeloLR = glm(Response ~ ., data = newTrain, family = binomial)

#Predicciones
predictionLR_Test_1 <-  predict(modeloLR, newdata=test, "response")
predictionLR_Test_1 = as.data.frame(predictionLR_Test_1)
predictionLR_Test_1$respuesta = ifelse(predictionLR_Test_1 > 0.5,"SI","NO")
resultsLR_Test_1<-table(test$Response, predictionLR_Test_1$respuesta)
resultsLR_Test_1
prop.table(resultsLR_Test_1)

# Acurracy
accuracyLR_Test_7<-sum(diag(resultsLR_Test_1))/sum(resultsLR_Test_1)
accuracyLR_Test_7

# Recall
recall_Test_7<-(resultsLR_Test_1[2,2]/(resultsLR_Test_1[2,1]+resultsLR_Test_1[2,2]))
recall_Test_7

# Precision
precision_Test_7<-(resultsLR_Test_1[2,2]/(resultsLR_Test_1[1,2]+resultsLR_Test_1[2,2]))
precision_Test_7

# Clean Memory
rm(modeloLR)
rm(predictionLR_Test_1)
rm(resultsLR_Test_1)
# -------------------------------- Fin Hip 1 ----------------------------------

# --------------------------------- Hip 2 -------------------------------------
modeloLR2 = glm(Response ~ Age + Gender + Previously_Insured +
                  Vehicle_Age + Vehicle_Damage + Policy_Sales_Channel, 
                data = newTrain, family = binomial)

#Predicciones
predictionLR_Test_2 <-  predict(modeloLR2, newdata=test, "response")
predictionLR_Test_2 = as.data.frame(predictionLR_Test_2)
predictionLR_Test_2$respuesta = ifelse(predictionLR_Test_2 > 0.5,"SI","NO")
resultsLR_Test_2<-table(test$Response, predictionLR_Test_2$respuesta)
resultsLR_Test_2
prop.table(resultsLR_Test_2)

# Acurracy
accuracyLR_Test_8<-sum(diag(resultsLR_Test_2))/sum(resultsLR_Test_2)
accuracyLR_Test_8

# Recall
recall_Test_8<-(resultsLR_Test_2[2,2]/(resultsLR_Test_2[2,1]+resultsLR_Test_2[2,2]))
recall_Test_8

# Precision
precision_Test_8<-(resultsLR_Test_2[2,2]/(resultsLR_Test_2[1,2]+resultsLR_Test_2[2,2]))
precision_Test_8

# Clean Memory
rm(modeloLR2)
rm(predictionLR_Test_2)
rm(resultsLR_Test_2)
# -------------------------------- Fin Hip 2 ----------------------------------


#Tabla de comparacion Accuracy, Recall y Precision

modelo = c(
  "Modelo NB 1", 
  "Modelo NB 2", 
  "Modelo DT 1", 
  "Modelo DT 2", 
  "Modelo RF 1", 
  "Modelo RF 2", 
  "Modelo LR 1", 
  "Modelo LR 2"
  )
accuracy = c(
  accuracyNB_Test_1,
  accuracyNB_Test_2,
  accuracyDT_Test_3,
  accuracyDT_Test_4,
  accuracyRF_Test_5,
  accuracyRF_Test_6,
  accuracyLR_Test_7,
  accuracyLR_Test_8
)

precision = c(
  precision_Test_1,
  precision_Test_2,
  precision_Test_3,
  precision_Test_4,
  precision_Test_5,
  precision_Test_6,
  precision_Test_7,
  precision_Test_8
)

recall = c(
  recall_Test_1,
  recall_Test_2,
  recall_Test_3,
  recall_Test_4,
  recall_Test_5,
  recall_Test_6,
  recall_Test_7,
  recall_Test_8
)

resumen = data.frame(modelo, accuracy, precision, recall)

resumen
