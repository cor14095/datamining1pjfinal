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
train <- read_csv("Data/train.csv", col_types = cols(id = col_number(), 
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
View(train)

# Explorando un par de datos
head(train)
summary(train)

# Exploremos un poco cada variable para encontrar nulls/nas
table(train$id)
table(train$Gender)
table(train$Age)
table(train$Driving_License)
table(train$Region_Code)
table(train$Previously_Insured)
table(train$Vehicle_Age)
table(train$Vehicle_Damage)
table(train$Annual_Premium)
table(train$Policy_Sales_Channel)
table(train$Vintage)
table(train$Response)

# Estamos libres de pecado, amen /\
for (i in train) {
  View(train[is.na(i),])
}

#Preparando para la matrix de correlacion

train$GenderDummy =  ifelse(train$Gender == "Male", 1 , 0)

#   < 1 Year    0
#   1-2 Year    1 
#   > 2 Years   2
train$Vehicle_AgeDummy =  ifelse (train$Vehicle_Age == "< 1 Year",0, ifelse(train$Vehicle_Age == "> 2 Years", 2 , 1))

train$VehiculeDamage_Dummy =  ifelse(train$Vehicle_Damage == "Yes", 1 , 0)

# Subset train
trainSinCategorico = train %>% select(
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
train$Gender <- as.factor(train$Gender)
train$Vehicle_Age <- as.factor(train$Vehicle_Age)
train$Vehicle_Damage <- as.factor(train$Vehicle_Damage)
train$Previously_Insured<- as.factor(train$Previously_Insured)
# train$Response <- as.factor(train$Response)
train$Region_Code <- as.factor(train$Region_Code)
train$Driving_License <- as.factor(train$Driving_License)

# Como se divide el valor de Response
table(train$Response)
# Respuesta positiva
(46710/(334399+46710)) * 100
# Respuesta negativa
(334399/(334399+46710)) * 100

# Revisamos todas las variables
variables <- c(
  train[2],
  train[4],
  train[6],
  train[7],
  train[8]
  )

# Revisamos categoricas
varNameCount <- 1
varPos <- c(2,4,6,7,8)
for (i in variables) {
  mosaicplot(
    Response ~ i,
    data = train,
    ylab = names(train)[varPos[varNameCount]],
    color = train$Response,
    shade = TRUE
  )
  varNameCount <- varNameCount + 1
}

# Revisamos variables cuantitativas
train %>% 
  ggplot() +
  aes(x = Gender, y = Age, color = Response) +
  geom_boxplot() +
  ggtitle("Por Genero y Edad")

train %>% 
  ggplot() +
  aes(x = Age, y = Annual_Premium, color = Response) +
  geom_histogram(stat='identity') +
  ggtitle("Por Edad y Monto acumulado de poliza")

train %>% 
  ggplot() +
  aes(x = Previously_Insured, y = as.numeric(Response), color = Response) +
  geom_bar(stat='identity') +
  ggtitle("Previamente asegurados acumulado por respuesta") +
  scale_y_continuous(breaks = seq(0, 300000, 50000), 
                     limits=c(0, 300000))

# Tablas de contingencia
# Contingencia de genero
genderTable <- table(train$Gender, train$Response)
genderTable

rowSums(genderTable)
colSums(genderTable)

rm(genderTable)

# Contingencia de genero
genderTable <- table(train$Gender, train$Response)
genderTable

rowSums(genderTable)
colSums(genderTable)

rm(genderTable)
