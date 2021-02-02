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

# Set de data
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
# Ver informacion
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

#Preparando para la matrix de correlacion

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

# Matriz
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

# Revisamos categoricas
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

# Tablas de contingencia
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

# Hipopotamo
# Las variables de Age, Previoulsy Ensured, Sale Channel, Vehicle Age, Vehicle Damage
# y Gender tienen una relacion que explica la variable Response

# Generamos nuestro subset para trainInsure y test
# Filas para trainInsure y test (utilizando paquete de CARET)
inTrain <- createDataPartition(y = trainInsure$Response, p = 0.7, list = FALSE)
test <- trainInsure[-inTrain,]
train <- trainInsure[inTrain,]

rm(trainInsure)
rm(inTrain)
rm(variables)
rm(i)
rm(varNameCount)
rm(varPos)

# Naive Bayes
# Hip 1
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
# TP/(TP+FN)
recall_Test_1 <- (resultsNB_Test_1[2,2]/(resultsNB_Test_1[2,1]+resultsNB_Test_1[2,2]))
recall_Test_1

# Precision
# TP/(TP+FP)
precision_Test_1 <- (resultsNB_Test_1[2,2]/(resultsNB_Test_1[1,2]+resultsNB_Test_1[2,2]))
precision_Test_1

# Clean Memory
rm(modeloBayes)
rm(predictionNB_Test_1)
rm(predNB_Test_1)

# Hip 2
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
#TP/(TP+FN)
recall_Test_2<-(resultsNB_Test_2[2,2]/(resultsNB_Test_2[2,1]+resultsNB_Test_2[2,2]))
recall_Test_2

# Precision
#TP/(TP+FP)
precision_Test_2<-(resultsNB_Test_2[2,2]/(resultsNB_Test_2[1,2]+resultsNB_Test_2[2,2]))
precision_Test_2

# Clean Memory
rm(modeloBayes2)
rm(predictionNB_Test_2)
rm(predNB_Test_2)

# Decision Tree

# Random Forest

# Logic Regression


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
  accuracyNB_Test_2
)

precision = c(
  precision_Test_1,
  precision_Test_2
)

recall = c(
  recall_Test_1,
  recall_Test_2
)

resumen = data.frame(modelo, accuracy, precision, recall)

resumen
