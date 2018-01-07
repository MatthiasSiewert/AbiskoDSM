#######
#author: Matthias Siewert
# matthias.siewert@natgeo.su.se
# Started 2016-01-01
# This script is to read in GIS data for digital soil mapping purposes.


######### GIS libraries
library(maptools)# autoloads sp
library(rgdal)
library(sp)
library(raster)
#library(Hmisc)
#library(mice)
library(rgeos)

library(RColorBrewer)
library(ggplot2)
library(reshape2)

#machine learning packages
library(snow) #for multicore usage
library(doMC) #for multicore usage in caret

library(randomForest)
library(e1071) #svm
library(nnet) #neural network
require(caret) #neural network optimization
#library(pls)
library(ithir) # Digital soil mapping #install.packages("ithir", repos="http://R-Forge.R-project.org")
library(MASS)


##########################################################################################
# Load raster 
# use brick instead of raster to load all layers

ABcomposite <- stack("/home/masi/Data/Abisko/GIS_RS_data/DSM/temp/ABcomposite1x1.tif")
#ABcomposite <- stack("/home/masi/Data/Abisko/GIS_RS_data/DSM/temp/ABcomposite2x2.tif")
#ABcomposite <- stack("/home/masi/Data/Abisko/GIS_RS_data/DSM/temp/ABcomposite10x10.tif")
#ABcomposite <- stack("/home/masi/Data/Abisko/GIS_RS_data/DSM/temp/ABcomposite30x30.tif")
#ABcomposite <- stack("/home/masi/Data/Abisko/GIS_RS_data/DSM/temp/ABcomposite100x100.tif")
#ABcomposite <- stack("/home/masi/Data/Abisko/GIS_RS_data/DSM/temp/ABcomposite1000x1000.tif")
names(ABcomposite) <- c('DEM','Orthophoto Red','Orthophoto Green','Orthophoto Blue',
                        'SPOT5 Green','SPOT5 RED','SPOT5 NIR','SPOT5 SWIR',
                        'SPOT5 NDVI','SPOT5 SAVI','SPOT5 NIR/SWIR','Slope','Aspect',
                        'Terrain Ruggedness Index','Topographic Position Index',
                        'Tophographic Wetness Index','Profile Curvature','Plan Curvature','Landform',
                        'Geology','Vegetation','Quartenary deposits','Land cover classification')
rm(ABcomposite)
ABcomposite <- writeRaster(ABcomposite,  '/home/masi/Data/Abisko/GIS_RS_data/DSM/test/ABcomposite1x1Fix.grd' ,overwrite=TRUE, format="raster")

#################################################
# Different ways to plot the composite
dev.off()
plot(ABcomposite)
plot(ABcomposite,13:23)

##################################################
#fix problem with extrem pixel values
# aspect
dev.off()
ABcomposite$Aspect[ABcomposite$Aspect<=0.1]=0
ABcomposite$Aspect[ABcomposite$Aspect>=359.9]=0
# Plan curvature
ABcomposite$Plan.Curvature[ABcomposite$Plan.Curvature>=0.460991]=0
ABcomposite$Plan.Curvature[ABcomposite$Plan.Curvature<=-0.547044]=0
plot(ABcomposite)
plot(ABcomposite,13:23)

#################################################
### Extract sampling points

#read points
ABpoints <- readOGR(dsn="/home/masi/Data/Abisko/GIS_RS_data/GPS/ABsitesSWEREF.shp", layer="ABsitesSWEREF")
names(ABpoints)# read columns

# transform into data.frame
ABpointsDF <- as.data.frame(ABpoints)


# Extract values from environmental data sets
ABpointsDF <- cbind(ABpointsDF,extract(ABcomposite, ABpoints))#, buffer = 4)) ## buffer didnt work for some reasion
# merge in the SOC data 
temp <- data.frame(pedon     = AB4pedon$pedon,
                   ped_type  = AB4pedon$ped_type,
                   OLdepth   = AB4pedon$OLdepth,
                   SOCOL     = AB4pedon$SOCOL,
                   SOCPF     = AB4pedon$SOCPF,
                   SOC0to30  = AB4pedon$SOC0to30,
                   SOC0to100 = AB4pedon$SOC0to100,
                   SOCTot    = AB4pedon$SOCTot)
ABpointsDF <- merge(ABpointsDF, temp, by.x="Pedon",by.y="pedon", all=F)

# remove all points where there is no SOC data 
ABpointsDF <- subset(ABpointsDF, !is.na(ABpointsDF$SOC0to100))
# if there is no value for OL, the set 0
ABpointsDF$SOCOL<- ifelse(is.na(ABpointsDF$SOCOL),0,ABpointsDF$SOCOL)

# remove all points outside area
ABpointsDF <- subset(ABpointsDF, !is.na(ABpointsDF$Orthophoto.Red))

# Save control file Promote ABpointsDF to spatialpointsdataframe
temp <- ABpointsDF[,c(3,4)]
ABcontrol<-SpatialPointsDataFrame(coords = temp, data = ABpointsDF,
                                   proj4string = CRS("+proj=utm +zone=33 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs "))


###################################################################
## Prediction
#################################################################

######################################
# Compare different prediction models

ABptsPredND           <- ABpointsDF  #no pseudo sampling points
ABptsPredND <- subset(ABptsPredND,ABptsPredND$ped_type != "Dummy") #no pseudo sampling points
# delete unnessesary stuff
ABptsPredND$Pedon     <- NULL
ABptsPredND$ped_type     <- NULL
ABptsPredND$Elevation <- NULL
ABptsPredND$coords.x1 <- NULL
ABptsPredND$coords.x2 <- NULL


### Prediction model for Total
temp <- ABptsPredND
drops <- c("OLdepth","SOCOL","SOCPF", "SOC0to30", "SOC0to100")
temp <- temp[ , !(names(temp) %in% drops)]
head(temp)

##to address multicolinearity after Kuhn in Building models with caret paper.
tempMCL <- temp[,-24]
ncol(tempMCL)
tempCOR <- cor(tempMCL)
highCorr <- findCorrelation(tempCOR, 0.90)
tempMCL <- tempMCL[, -highCorr]
ncol(tempMCL)
temp <- cbind(tempMCL, SOCTot = temp[,24])

# Drop colinear layers from rasterstack
ABcomposite <- dropLayer(ABcomposite, highCorr)
#ABcompositeColin <- ABcomposite # save the orginal raster stack

### define training split
set.seed(2)
training <- sample(nrow(temp), 1 * nrow(temp))  #subset if you want an external validation
training

#### Define parameters for controling model training. 10 fold resampling with 5 repeats
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5, savePredictions = T)

#### Train models. Four models will be trained - linear regression, random forest, support vector machines,
#### and Stochastic Gradient Boosting. Seed is set to ensure same samples are used in each model

# multi cores for caret
registerDoMC(cores = 7)

#### Train Linear Regression Model
set.seed(2)
ABlmSOCTot <- train(SOCTot~., data = temp[training,], method="lm", trControl=ctrl)
ABlmSOCTot

#### Train Artifical Neural Network
temp2 <- expand.grid(.decay=exp(seq(log(0.00001), log(1), length.out = 6)), .size=seq(2,20,1))#looks ok in validation
set.seed(2)
ABnnSOCTot <- train(SOCTot~., data = temp, method="nnet", trControl=ctrl,tuneGrid=temp2, verbose=F,trace=F, linout = 1)
ABnnSOCTot

#### Train Support Vector Machines
temp2 <- expand.grid( .C=seq(0.5,4,0.5), .sigma=seq(0.01, 0.1, 0.01))#optimization grid
set.seed(2)
ABsvmSOCTot <- train(SOCTot~., data = temp[training,], method="svmRadial", trControl=ctrl,tuneGrid=temp2, verbose=F,trace=F, linout = 1,
                    preProc = c("center","scale"))
ABsvmSOCTot

#### Train Random Forest
temp2 <- expand.grid(.mtry=c(7))#optimization grid
set.seed(2)
ABrfSOCTot <- train(SOCTot~. , data=temp[training,], method="rf",tuneGrid=temp2, trace=F, linout = 1, importance=T, trControl=ctrl,
                    corr.bias = T,ntree = 100, replace = T, nodesize = 3)
ABrfSOCTot

### Access  crossvalidation values
rm(ABcrossVal)
ABcrossVal <- as.data.frame(ABlmSOCTot$results[which.min(ABlmSOCTot$results[, "RMSE"]), 2:3,1])
ABcrossVal <- rbind(ABcrossVal, as.data.frame(ABnnSOCTot$results[which.min(ABnnSOCTot$results[, "RMSE"]), 3:4,1]))
ABcrossVal <- rbind(ABcrossVal, as.data.frame(ABsvmSOCTot$results[which.min(ABsvmSOCTot$results[, "RMSE"]), 3:4,1]))
ABcrossVal <- rbind(ABcrossVal, as.data.frame(ABrfSOCTot$results[which.min(ABrfSOCTot$results[, "RMSE"]), 2:3,1]))
rownames(ABcrossVal) <- c("LM", "ANN","SVM", "RF")
ABcrossVal

#####################################
### Plots for comparision
### Prediction for plots
ABlmSOCTotC <- predict  (ABlmSOCTot, temp[training,])
ABnnSOCTotC <- predict  (ABnnSOCTot, temp[training,])
ABsvmSOCTotC <- predict (ABsvmSOCTot,temp[training,])
ABrfSOCTotC <- predict  (ABrfSOCTot, temp[training,])

#########################
#############using ggplot adjust values to svm,lm and rf
temp3 <- data.frame(SOCTot = temp$SOCTot[training], "Multiple Linear Regression" = ABlmSOCTotC,
              "Artifical Neural Network" = ABnnSOCTotC[,1], "Support Vector Machine" =ABsvmSOCTotC,
              "Random Forest"=ABrfSOCTotC,
              check.names = FALSE) # avoids dots in name
temp2 <- melt(temp3, id= "SOCTot", variable.name = "Model")
temp3

# error labels
temp4 <- ABcrossVal
temp4 <- t(temp4)
temp4 <-  rbind(temp4[c(2,1),], CCC = ABlin[1:4,1]) # merge test result for Lin's CCC from ABlin
temp4 <-  round(temp4,3)
temp4

### ggplot
ABmodcompplot <- ggplot(temp2, aes(x = SOCTot, y=value, group =Model, color=Model))
ABmodcompplot <- ABmodcompplot +
  facet_wrap(~ Model, labeller = label_value) +
  annotate(geom="text", x=20, y=85, label=paste("atop('R2: '*",temp4[1,],",' CCC: '*",temp4[3,],",'RMSE: '*",temp4[2,],")"), parse=T) +
  annotate(geom="text", x=20, y=70, label=paste('RMSE: ',temp4[2,]), parse=T) +
  geom_point(aes(color=Model)) +
  geom_abline(col = "black")  +
  stat_smooth(method = lm,se = FALSE) +
  coord_cartesian(ylim = c(0, 100),xlim = c(0, 100)) +
  theme(legend.position="right") +
  xlab("Sampled SOC") +
  ylab("Predicted SOC") +
  theme(legend.position="no",axis.text = element_text(colour = "black",angle=0,hjust = 0.5,vjust = 0.5 ,size = 12),
        axis.title.y = element_text(size = 12),
        panel.background = element_blank(),
        panel.grid.major = element_line(colour = "grey", size = 0.5),
        aspect.ratio = 1) # make plots square
ABmodcompplot


#### pdf plot ###################
pdf(file="Plots/Fig5_modelcompRev2.pdf", width=7, height=7)
ABmodcompplot
dev.off()

png("Plots/Fig5_modelcompRev2.png",width=600,height=600)
ABmodcompplot
dev.off()
###########################################################################
##############################
# predict the models and for visual comparision
beginCluster(7)
system.time(ABlmSOCTotRast   <- clusterR(ABcomposite, predict, args=list(ABlmSOCTot)))
system.time(ABnnSOCTotRast   <- clusterR(ABcomposite, predict, args=list(ABnnSOCTot)))
system.time(ABsvmSOCTotRast  <- clusterR(ABcomposite, predict, args=list(ABsvmSOCTot)))
system.time(ABrfSOCTotRast   <- clusterR(ABcomposite, predict, args=list(ABrfSOCTot)))
endCluster()

## set values below zero to zero
dev.off()
ABlmSOCTotRast[ABlmSOCTotRast < 1] <- 0
ABnnSOCTotRast[ABnnSOCTotRast < 1] <- 0
ABsvmSOCTotRast[ABsvmSOCTotRast<1] <- 0
ABrfSOCTotRast[ABrfSOCTotRast < 1] <- 0

#### Mask out
dev.off()
ABlmSOCTotRast  <- mask(ABlmSOCTotRast  , ABmask,inverse = T)
ABnnSOCTotRast  <- mask(ABnnSOCTotRast  , ABmask,inverse = T)
ABsvmSOCTotRast <- mask(ABsvmSOCTotRast , ABmask,inverse = T)
ABrfSOCTotRast  <- mask(ABrfSOCTotRast  , ABmask,inverse = T)

### Write to disk to save memory
writeRaster(ABlmSOCTotRast,  '/home/masi/Data/Abisko/GIS_RS_data/DSM/out/ABlmSOCTot.tif' ,overwrite=TRUE)
writeRaster(ABnnSOCTotRast,  '/home/masi/Data/Abisko/GIS_RS_data/DSM/out/ABnnSOCTot.tif' ,overwrite=TRUE)
writeRaster(ABsvmSOCTotRast, '/home/masi/Data/Abisko/GIS_RS_data/DSM/out/ABsvmSOCTot.tif',overwrite=TRUE)
writeRaster(ABrfSOCTotRast,  '/home/masi/Data/Abisko/GIS_RS_data/DSM/out/ABrfSOCTot.tif' ,overwrite=TRUE)
#ABlmSOCTotRast  <- raster('/home/masi/Data/Abisko/GIS_RS_data/DSM/out/ABlmSOCTot.tif')
#ABnnSOCTotRast  <- raster('/home/masi/Data/Abisko/GIS_RS_data/DSM/out/ABnnSOCTot.tif')
#ABsvmSOCTotRast <- raster('/home/masi/Data/Abisko/GIS_RS_data/DSM/out/ABsvmSOCTot.tif')
#ABrfSOCTotRast  <- raster('/home/masi/Data/Abisko/GIS_RS_data/DSM/out/ABrfSOCTot.tif')


### Plot maps:
# define color palette
rgb.palette <- colorRampPalette(c("white", "orange", "red"),
                                space = "rgb")
### View results
plot(ABlmSOCTotRast,col = rgb.palette(20))
plot(ABnnSOCTotRast,col = rgb.palette(20))
plot(ABsvmSOCTotRast,col = rgb.palette(20))
plot(ABrfSOCTotRast,col = rgb.palette(20))


