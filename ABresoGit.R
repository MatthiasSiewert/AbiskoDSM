####
#######
#author: Matthias Siewert
# matthias.siewert@umu.se
# Started 2016-01-01
# This script is to apply the random forest model to different depth intervals and different resolutions.
# See Siewert() biogeosciences for more details.


# ######### GIS libraries
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
require(caret) #neural network optimization
library(snow) #for multicore usage
library(doMC) #for multicore usage in caret


####
####  Loop the following through each raster resolution. Need to comment in the appropriate resolution.
####  If I have time, I will make a nice function out of this :)

#ABcomposite <- stack("/home/masi/Data/Abisko/GIS_RS_data/DSM/temp/ABcomposite1x1.tif")
#ABcomposite <- stack("/home/masi/Data/Abisko/GIS_RS_data/DSM/temp/ABcomposite2x2.tif")
#ABcomposite <- stack("/home/masi/Data/Abisko/GIS_RS_data/DSM/temp/ABcomposite10x10.tif")
#ABcomposite <- stack("/home/masi/Data/Abisko/GIS_RS_data/DSM/temp/ABcomposite30x30.tif")

ABcomposite <- stack("ABcomposite30x30f.gri")

### lower resolutions can be generated using system call under Unix:
#gdal_translate -tr 250 250 ABcomposite1x1.tif ABcomposite250x250.tif

#ABcomposite <- stack("/home/masi/Data/Abisko/GIS_RS_data/DSM/temp/ABcomposite100x100.tif")
#ABcomposite <- stack("/home/masi/Data/Abisko/GIS_RS_data/DSM/temp/ABcomposite250x250.tif")
#ABcomposite <- stack("/home/masi/Data/Abisko/GIS_RS_data/DSM/temp/ABcomposite1000x1000.tif")

# 
#
#################################################
### Extract sampling points

#read points
ABpoints <- readOGR(dsn="/home/masi/Data/Abisko/GIS_RS_data/GPS/ABsitesSWEREF.shp", layer="ABsitesSWEREF")
names(ABpoints)# read columns

# transform into data.frame
ABpointsDF <- as.data.frame(ABpoints)
points(ABpoints)
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
ABptsPred           <- ABpointsDF
# delete unnessesary stuff
ABptsPred$Pedon     <- NULL
ABptsPred$ped_type  <- NULL
ABptsPred$Elevation <- NULL
ABptsPred$coords.x1 <- NULL
ABptsPred$coords.x2 <- NULL


### Prediction model for certain depth. Comment out unneeded depths. 
temp <- ABptsPred
#drops <- c("SOCOL","SOC0to30","SOCPF", "SOC0to100", "SOCTot")   # OLdepth
#drops <- c("OLdepth","SOC0to30","SOCPF", "SOC0to100", "SOCTot") # SOCOL
##drops <- c("OLdepth","SOCOL","SOC0to30", "SOC0to100", "SOCTot") # SOCPF
#drops <- c("OLdepth","SOCOL","SOCPF", "SOC0to100", "SOCTot")    # SOC0to30
#drops <- c("OLdepth","SOCOL","SOCPF", "SOC0to30", "SOCTot")     # SOC0to100
drops <- c("OLdepth","SOCOL","SOCPF", "SOC0to30", "SOC0to100")  # SOCTot
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

### define training split
set.seed(2)
training <- sample(nrow(temp), 1 * nrow(temp))

#### Define parameters for controling model training. 10 fold resampling with 5 repeats
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5, savePredictions = T)

# multi cores for caret
registerDoMC(cores = 7)

### Run the random forest model
temp2 <- expand.grid(.mtry=c(7))#optimization grid
set.seed(2)
ABrfSOCmod <- train(SOCTot~. , data=temp[training,], method="rf",tuneGrid=temp2, trace=F, linout = 1, importance=T, trControl=ctrl,
                    corr.bias = T,ntree = 100, replace = T, nodesize = 3)#, metric = "Rsquared")
ABrfSOCmod
plot(varImp(ABrfSOCmod))

### save model, comment out as needed
#ABrfOLdepth  <- ABrfSOCmod
#ABrfSOCOL    <- ABrfSOCmod
##ABrfSOCPF   <- ABrfSOCmod
#ABrfSOC30    <- ABrfSOCmod
##ABrfSOC100   <- ABrfSOCmod
#ABrfSOCTot    <- ABrfSOCmod

#ABrfSOCTot1x1mod    <- ABrfSOCmod
#ABrfSOCTot2x2mod    <- ABrfSOCmod
#ABrfSOCTot10x10mod    <- ABrfSOCmod
ABrfSOCTot30x30mod    <- ABrfSOCmod
#ABrfSOCTot100x100mod    <- ABrfSOCmod
#ABrfSOCTot250x250mod    <- ABrfSOCmod
#ABrfSOCTot1000x1000mod    <- ABrfSOCmod


######################################
## Prediction
# as multicore function
beginCluster(7)
# system.time(ABrfOLdepthRast<- clusterR(ABcomposite, predict, args=list(ABrfOLdepth)))
# system.time(ABrfSOCOLRast  <- clusterR(ABcomposite, predict, args=list(ABrfSOCOL)))
# system.time(ABrfSOCPFRast  <- clusterR(ABcomposite, predict, args=list(ABrfSOCPF)))
# system.time(ABrfSOC30Rast  <- clusterR(ABcomposite, predict, args=list(ABrfSOC30)))
# system.time(ABrfSOC100Rast <- clusterR(ABcomposite, predict, args=list(ABrfSOC100)))
# system.time(ABrfSOCTotRast <- clusterR(ABcomposite, predict, args=list(ABrfSOCTot)))

##system.time(ABrfSOCTotRast1x1 <- clusterR(ABcomposite, predict, args=list(ABrfSOCTot1x1mod))) # same as ABrfSOCTot
# system.time(ABrfSOCTotRast2x2 <- clusterR(ABcomposite, predict, args=list(ABrfSOCTot2x2mod)))
# system.time(ABrfSOCTotRast10x10 <- clusterR(ABcomposite, predict, args=list(ABrfSOCTot10x10mod)))
 system.time(ABrfSOCTotRast30x30 <- clusterR(ABcomposite, predict, args=list(ABrfSOCTot30x30mod)))
# system.time(ABrfSOCTotRast100x100 <- clusterR(ABcomposite, predict, args=list(ABrfSOCTot100x100mod)))
# system.time(ABrfSOCTotRast250x250 <- clusterR(ABcomposite, predict, args=list(ABrfSOCTot250x250mod)))
# system.time(ABrfSOCTotRast1000x1000 <- clusterR(ABcomposite, predict, args=list(ABrfSOCTot1000x1000mod)))
endCluster()

#################################
## set values below zero to zero, comment out as needed
#ABrfOLdepthRast[ABrfOLdepthRast<1] <- 0
#ABrfSOCOLRast[ABrfSOCOLRast    <1] <- 0
##ABrfSOCPFRast[ABrfSOCPFRast   <1] <- 0
#ABrfSOC30Rast[ABrfSOC30Rast    <1] <- 0
#ABrfSOC100Rast[ABrfSOC100Rast  <1] <- 0
#ABrfSOCTotRast[ABrfSOCTotRast  <1] <- 0

# ABrfSOCTotRast2x2[ABrfSOCTotRast2x2              <1] <- 0
# ABrfSOCTotRast10x10[ABrfSOCTotRast10x10          <1] <- 0
 ABrfSOCTotRast30x30[ABrfSOCTotRast30x30          <1] <- 0
# ABrfSOCTotRast100x100[ABrfSOCTotRast100x100      <1] <- 0
# ABrfSOCTotRast250x250[ABrfSOCTotRast250x250      <1] <- 0
# ABrfSOCTotRast1000x1000[ABrfSOCTotRast1000x1000  <1] <- 0

### Mask out water and artifical areas to NA
# ABrfOLdepthRast      <- mask(ABrfOLdepthRast, ABmask,inverse = T)
# ABrfSOCOLRast        <- mask(ABrfSOCOLRast  , ABmask,inverse = T)
# #ABrfSOCPFRast       <- mask(ABrfSOCPFRast  , ABmask,inverse = T)
# ABrfSOC30Rast        <- mask(ABrfSOC30Rast  , ABmask,inverse = T)
# ABrfSOC100Rast       <- mask(ABrfSOC100Rast , ABmask,inverse = T)
# ABrfSOCTotRast       <- mask(ABrfSOCTotRast , ABmask,inverse = T)

# ABrfSOCTotRast2x2         <- mask(ABrfSOCTotRast2x2 , ABmask,inverse = T, updatevalue=0)
# ABrfSOCTotRast10x10       <- mask(ABrfSOCTotRast10x10 , ABmask,inverse = T, updatevalue=0)  
 ABrfSOCTotRast30x30       <- mask(ABrfSOCTotRast30x30 , ABmask,inverse = T, updatevalue=0)   
# ABrfSOCTotRast100x100     <- mask(ABrfSOCTotRast100x100 , ABmask,inverse = T, updatevalue=0)  
# ABrfSOCTotRast250x250     <- mask(ABrfSOCTotRast250x250 , ABmask,inverse = T, updatevalue=0)
# ABrfSOCTotRast1000x1000   <- mask(ABrfSOCTotRast1000x1000 , ABmask,inverse = T, updatevalue=0)# replace with 0 if wanted


# ####
# ABrfSOCTotRast2x2 <- raster::resample(ABrfSOCTotRast2x2, ABrfSOCTotRast,method ="ngb")
# ABrfSOCTotRast10x10 <- raster::resample(ABrfSOCTotRast10x10, ABrfSOCTotRast,method ="ngb")
# ABrfSOCTotRast30x30 <- raster::resample(ABrfSOCTotRast30x30, ABrfSOCTotRast,method ="ngb")
# ABrfSOCTotRast100x100 <- raster::resample(ABrfSOCTotRast100x100, ABrfSOCTotRast,method ="ngb")
# ABrfSOCTotRast250x250 <- raster::resample(ABrfSOCTotRast250x250, ABrfSOCTotRast,method ="ngb")
# ABrfSOCTotRast1000x1000 <- raster::resample(ABrfSOCTotRast1000x1000, ABrfSOCTotRast,method ="ngb")
# 
# ABrfSOCTotRast2x2         <- mask(ABrfSOCTotRast2x2 , ABmask,inverse = T)
# ABrfSOCTotRast10x10       <- mask(ABrfSOCTotRast10x10 , ABmask,inverse = T)  
 ABrfSOCTotRast30x30       <- mask(ABrfSOCTotRast30x30 , ABmask,inverse = T)   
# ABrfSOCTotRast100x100     <- mask(ABrfSOCTotRast100x100 , ABmask,inverse = T)  
# ABrfSOCTotRast250x250     <- mask(ABrfSOCTotRast250x250 , ABmask,inverse = T)
# ABrfSOCTotRast1000x1000   <- mask(ABrfSOCTotRast1000x1000 , ABmask,inverse = T)# replace with 0 if wanted
# 

#### Write raster to disk
# writeRaster(ABrfOLdepthRast, '/home/masi/Data/Abisko/GIS_RS_data/DSM/out/ABrfOLdepth.tif',overwrite=TRUE)
# writeRaster(ABrfSOCOLRast,   '/home/masi/Data/Abisko/GIS_RS_data/DSM/out/ABrfSOCOL.tif',overwrite=TRUE)
# #writeRaster(ABrfSOCPFRast,   '/home/masi/Data/Abisko/GIS_RS_data/DSM/out/ABrfSOCPF.tif',overwrite=TRUE)
# writeRaster(ABrfSOC30Rast,   '/home/masi/Data/Abisko/GIS_RS_data/DSM/out/ABrfSOC30.tif',overwrite=TRUE)
#writeRaster(ABrfSOC100Rast,  '/home/masi/Data/Abisko/GIS_RS_data/DSM/out/ABrfSOC100.tif',overwrite=TRUE)
# writeRaster(ABrfSOCTotRast,  '/home/masi/Data/Abisko/GIS_RS_data/DSM/out/ABrfSOCTot.tif',overwrite=TRUE)
# 
# writeRaster(ABrfSOCTotRast2x2,  '/home/masi/Data/Abisko/GIS_RS_data/DSM/out/reso/ABrfSOCTot2x2.tif',overwrite=TRUE)
# writeRaster(ABrfSOCTotRast10x10,  '/home/masi/Data/Abisko/GIS_RS_data/DSM/out/reso/ABrfSOCTot10x10.tif',overwrite=TRUE)
 writeRaster(ABrfSOCTotRast30x30,  '/home/masi/Data/Abisko/GIS_RS_data/DSM/out/reso/ABrfSOCTot30x30.tif',overwrite=TRUE)
# writeRaster(ABrfSOCTotRast100x100,  '/home/masi/Data/Abisko/GIS_RS_data/DSM/out/reso/ABrfSOCTot100x100.tif',overwrite=TRUE)
# writeRaster(ABrfSOCTotRast250x250,  '/home/masi/Data/Abisko/GIS_RS_data/DSM/out/reso/ABrfSOCTot250x250.tif',overwrite=TRUE)
# writeRaster(ABrfSOCTotRast1000x1000,  '/home/masi/Data/Abisko/GIS_RS_data/DSM/out/reso/ABrfSOCTot1000x1000.tif',overwrite=TRUE)

# ### Read rasters
# ABrfOLdepthRast          <- raster('/home/masi/Data/Abisko/GIS_RS_data/DSM/out/ABrfOLdepth.tif')
# ABrfSOCOLRast            <- raster('/home/masi/Data/Abisko/GIS_RS_data/DSM/out/ABrfSOCOL.tif')
# ABrfSOC30Rast            <- raster('/home/masi/Data/Abisko/GIS_RS_data/DSM/out/ABrfSOC30.tif')
# ABrfSOC100Rast           <- raster('/home/masi/Data/Abisko/GIS_RS_data/DSM/out/ABrfSOC100.tif')
# ABrfSOCTotRast           <- raster('/home/masi/Data/Abisko/GIS_RS_data/DSM/out/ABrfSOCTot.tif')
# 
# ABrfSOCTotRast2x2        <- raster('/home/masi/Data/Abisko/GIS_RS_data/DSM/out/reso/ABrfSOCTot2x2.tif')
# ABrfSOCTotRast10x10      <- raster('/home/masi/Data/Abisko/GIS_RS_data/DSM/out/reso/ABrfSOCTot10x10.tif')
 ABrfSOCTotRast30x30      <- raster('/home/masi/Data/Abisko/GIS_RS_data/DSM/out/reso/ABrfSOCTot30x30.tif')
# ABrfSOCTotRast100x100    <- raster('/home/masi/Data/Abisko/GIS_RS_data/DSM/out/reso/ABrfSOCTot100x100.tif')
# ABrfSOCTotRast250x250    <- raster('/home/masi/Data/Abisko/GIS_RS_data/DSM/out/reso/ABrfSOCTot250x250.tif')
# ABrfSOCTotRast1000x1000  <- raster('/home/masi/Data/Abisko/GIS_RS_data/DSM/out/reso/ABrfSOCTot1000x1000.tif')

dev.off()
# plot(ABrfOLdepthRast,col = rgb.palette(20))
# plot(ABrfSOCOLRast,col = rgb.palette(20))
# #plot(ABrfSOCPFRast,col = rgb.palette(20))
# plot(ABrfSOC30Rast,col = rgb.palette(20))
# plot(ABrfSOC100Rast,col = rgb.palette(20))
# plot(ABrfSOCTotRast,col = rgb.palette(20))
# 
# plot(ABrfSOCTotRast2x2,col = rgb.palette(20))
# plot(ABrfSOCTotRast10x10,col = rgb.palette(20))
 plot(ABrfSOCTotRast30x30,col = rgb.palette(20))
# plot(ABrfSOCTotRast100x100,col = rgb.palette(20))
# plot(ABrfSOCTotRast250x250,col = rgb.palette(20))
# plot(ABrfSOCTotRast1000x1000,col = rgb.palette(20))




######################################
#### Plot variable importance
dev.off()
varImpPlot(ABrfSOC30$finalModel , main = "Variable importance SOC-30", type =1, pch=19, col=1, cex=1)
#...


#########################################


# Define two extraction functions
LandscStats <- function(variable)
{
  SUM = cellStats(variable, stat='sum', na.rm=TRUE)
  MEAN = cellStats(variable, stat='mean', na.rm=TRUE)
  SD= cellStats(variable, stat='sd', na.rm=TRUE)
  MAX= cellStats(variable, stat='max', na.rm=TRUE)
  MIN = cellStats(variable, stat='min', na.rm=TRUE)
  return(list(sum=SUM,mean=MEAN,sd=SD, max=MAX,min=MIN))
}

UpscalResult <- function(Model, LCC)
{
  beginCluster(7)
  temp <- extract(Model, LCC,progress ="text")
  temp2 <- data.frame(LCC)
  temp2$SOCsum <-sapply(temp, sum, na.rm=T) #* 4 # adjust for resolution in m2 per pixel
  temp2$SOCmean <-sapply(temp, mean, na.rm=T)
  temp2$SOCsd <-sapply(temp, sd, na.rm=T)
  temp2$SOCmin <-sapply(temp, min, na.rm=T)
  temp2$SOCmax <-sapply(temp, max, na.rm=T)
  endCluster()
  return(list(temp=temp, temp2=temp2))
}


################################################ Example for value extraction
LandscStats(ABrfSOCTotRast30x30)
ABparti30x30     <- UpscalResult(ABrfSOCTotRast30x30,ABlcc) ; ABparti30x30$temp2

