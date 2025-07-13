#############################################################################
# CFSDB - Estimate day of burning and extract covariates
# Quinn Barber, October 2023
##############################################################################

# This code interpolates fire detection hotspots for a single fire and extracts
# environmental covariates based on the interpolated day of burn. The full Canadian
# Fire Spread Database (CFSDB) requires running this code over all NBAC fires over
# 1,000 ha in final size

# Required inputs: final fire perimeter (can be a convex/concave hull around hotspots),
# hotspots cropped to fire period, environmental covariates of interest, and a base
# raster. 

# For comprehensive code using the National Burned Area Composite, contact Quinn Barber
# quinn.barber@nrcan-rncan.gc.ca

library(terra)
library(dplyr)
library(tidyterra)
library(stringr)
library(data.table)
library(gstat)
library(rmapshaper)
library(sf)
library(stars)
library(lubridate)
library(units)
library(mgcv)

setwd("C:/Projects/CFSDB/upload_example")

# 1.0 Prepare input data --------------------------------------------------
# Preprocessing includes perimeters and hotspots. Base grid should be in a projected coordinate system such as epsg:9001
basegrid       <- rast("basegrid_180m.tif")

# 1.1 Process perimeter. Here we assume perimeter is from NBAC ------------
perimeter      <- vect("perimeter.shp")

#Make an informed decision about which field represents the fire start/end date
perimeter$STARTDATE <- perimeter$AFSDATE
perimeter$ENDDATE <- perimeter$EDATE
perimeter <- terra::project(perimeter, crs(basegrid))


# 1.2 Process hotspots. Here we assume FIRMs ------------------------------
modis_hotspots <- read.csv("modis_2021_Canada.csv")
viirs_hotspots <- read.csv("viirs-snpp_2021_Canada.csv")
names(viirs_hotspots) <- names(modis_hotspots)
hotspots       <- rbind(modis_hotspots, viirs_hotspots)

hotspots$JDAY     <- as.numeric(lubridate::yday(hotspots$acq_date))
hotspots$ACQ_TIME <- as.numeric(format(hotspots$acq_time, digits=4))/2400
hotspots$YEAR     <- lubridate::year(hotspots$acq_date)
hotspots$JDAYDEC  <- hotspots$JDAY + hotspots$ACQ_TIME

#Crop to perimeter date. In this case, maximum allowable error is 30 days.
hotspots <- hotspots %>% dplyr::filter(YEAR == lubridate::year(perimeter$STARTDATE), 
                                       JDAY >= lubridate::yday(perimeter$STARTDATE) - 30,
                                       JDAY <= lubridate::yday(perimeter$ENDDATE) + 30)
                                       
hotspots <- vect(hotspots, geom = c("longitude", "latitude"), crs = "epsg:4326")
hotspots <- terra::project(hotspots, crs(basegrid))

#Spatially crop to perimeter + 1000 m
perimeter_buf <- terra::buffer(perimeter, 1000)
hotspots      <- terra::crop(hotspots, perimeter_buf) 

# 2.0 Interpolate Fire Arrival --------------------------------------------
# Here we use the subset hotspots and pre-defined perimeter to interpolate fire arrival time using kriging
grid.fire <- terra::crop(basegrid, perimeter)
grid.fire <- terra::rasterize(perimeter, grid.fire, fun="max", field=0)

grid.pt <- st_as_stars(grid.fire)
grid.pt <- st_as_sf(grid.pt, as_points = TRUE, merge = FALSE)

#Time zone correction - here assume daylight savings time and only one time zone
timezone <- vect("C:/Projects/CFSDB/upload_example/Canada_Time_Zones.shp") %>% 
    terra::project(crs(basegrid))
timezone <- terra::intersect(centroids(perimeter), timezone) %>% dplyr::select(LST_offset, LDT_offset)
timezone_offset <- as.numeric(timezone$LDT_offset)

hotspots$JDAYDEC <- hotspots$JDAYDEC - timezone_offset/24
hotspots <- st_as_sf(hotspots) #gstat needs sf

#kriging
dob.kriged <- gstat::variogram(JDAYDEC ~ 1, hotspots)
dob.fit <- fit.variogram(dob.kriged, vgm(c("Exp", "Sph", "Mat")))
dob.kriged <- krige(JDAYDEC~1, hotspots, grid.pt, model=dob.fit, nmax=6)

# Assign values to grid, using GRID.FIRE as a base raster
grid.jday <- grid.fire
grid.jday[!is.na(grid.jday)] <- dob.kriged$var1.pred

writeRaster(grid.jday, "firearrival_decimal_krig.tif", overwrite=T)
grid.jday <- trunc(grid.jday)
writeRaster(grid.jday, "firearrival_yday_krig.tif", overwrite=T)

# 3.0 Covariate Extraction ------------------------------------------------
# The method of covariate extraction must change depending on if the covariate is
# static, varies annually, daily, or by some other period
# slope (static)
grid.jday <- rast("firearrival_yday_krig.tif")
grid.jday <- as.points(grid.jday)
names(grid.jday) <- "DOB"

#static: slope
slope <- rast("slope.tif")
grid.jday <- grid.jday %>% terra::project(crs(slope))
cov.names <- "slope"
cov.slice <- terra::extract(slope, grid.jday)[,-1]
grid.jday[,cov.names] <- cov.slice

#daily: weather
fwi.stack  <- rast(paste0("fire_weather_index_", perimeter$YEAR, ".tif"))
grid.jday <- grid.jday %>% terra::project(crs(fwi.stack))
cov.names <- "fwi"

for (x in unique(grid.jday$DOB)){
    cov.slice <- terra::extract(fwi.stack[[x]], grid.jday[grid.jday$DOB == x,])[,-1]
    names(cov.slice) <- cov.names
    grid.jday[grid.jday$DOB == x, cov.names] <- cov.slice
}

grid.jday <- terra::project(grid.jday, crs(basegrid))
writeVector(grid.jday, "firearrival_pts.shp", overwrite=TRUE)

#Daily spread estimation - assuming circular growth. Note this biases later fires day to unestimated growth
PTS <- vect("firearrival_pts.shp")
DOB <- rast("firearrival_yday_krig.tif")
DOB <- as.polygons(DOB)
DOB <- st_as_sf(DOB)
names(DOB)[1] <- "DOB"
  
DOB$firearea <- st_area(DOB)
DOB$firearea <- set_units(DOB$firearea, ha)
DOB <- vect(DOB)

DOB <- DOB %>% as.data.frame() %>% dplyr::group_by(DOB) %>% summarise(firearea = sum(as.numeric(firearea)))
DOB <- DOB %>% dplyr::mutate(firearea = ifelse(is.na(firearea), 0, firearea))
DOB$cumuarea <- cumsum(DOB$firearea)
DOB$fireday <- DOB$DOB - min(DOB$DOB) + 1
DOB$sprdistm <- 2*((DOB$cumuarea * 10000/pi)^0.5 - ((DOB$cumuarea*10000-DOB$firearea*10000)/pi)^0.5) # *2 for unidirectional growth

PTS <- left_join(PTS, DOB, by="DOB") %>% arrange(DOB)
writeVector(PTS, "firespread_pts.shp", overwrite=TRUE)
write.csv(as.data.frame(PTS), "firespread_pts.csv", row.names=FALSE)

# 4.0 Summarize and model -----------------------------------------------------
grid.pts <- read.csv("firespread_pts.csv")
grid.groups <- grid.pts %>% as.data.frame() %>% 
    dplyr::group_by(DOB) %>% 
    dplyr::summarize(fireday  = mean(fireday, na.rm=TRUE),
                     sprdistm = mean(sprdistm, na.rm=TRUE),
                     firearea = mean(firearea, na.rm=TRUE),
                     slope    = mean(slope, na.rm=TRUE),
                     fwi      = mean(fwi, na.rm=TRUE))
write.csv(grid.pts, "firespread_groups.csv", row.names=F)

# GAMs, for example
mod_fs = gam(sprdistm ~ slope + fwi, data = grid.groups)
summary(mod_fs)

# R2 = 0.356. Pretty good for 2 variables on a single fire!
