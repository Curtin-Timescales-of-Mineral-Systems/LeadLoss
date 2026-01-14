

# 
# ############################# U-Pb modeling inputs  ############################# 
library(IsoplotR)
library(dplyr)

source( "UPb_Constants_Functions_Libraries.R" )   # Read in constants and functions from the other file





### Create a synthetic dataset - this one is perfectly linear
# define line
lower.int.age <- 30
upper.int.age.max.1 <- 1850
upper.int.age.min.1 <- 1750
upper.int.age.max.2 <- 1450
upper.int.age.min.2 <- 1350
upper.int.age.max.3 <- 1150
upper.int.age.min.3 <- 1050

## Set model parameters
num.data.points <- 150 # number of data in each dataset
## number of iterations - increase after testing
num.iterations <- 2 # number of iterations

proportion.max <- 99 #how concordant can it be?
proportion.min <- 0.05 #min concordance it can be?
uncertainty.max <- 0.05 # upper limit on uncertainty
uncertainty.min <- 0.02 # lower limit on uncertainty
rho.max <- 0.98 # upper rho value allowed
rho.min <- 0.75 # lower rho value allowed

data.temp <- data.frame( "sample.name" = rep( NA, num.data.points ),
                         "ratio735"  = rep( NA, num.data.points ),
                         "err.735" = rep( NA, num.data.points ),
                         "ratio638" = rep( NA, num.data.points ),
                         "err.638" = rep( NA, num.data.points ),
                         "rho" = rep( NA, num.data.points ),
                         "age76" = rep( NA, num.data.points ) )

output.isoplotr.1 <- data.frame( lower.int.age = rep(NA, num.iterations),
                               lower.uncert = rep(NA, num.iterations),
                               upper.int.age = rep(NA, num.iterations),
                               upper.uncert = rep(NA, num.iterations),
                               MSWD = rep(NA, num.iterations) )
output.isoplotr.2 <- data.frame( lower.int.age = rep(NA, num.iterations),
                                 lower.uncert = rep(NA, num.iterations),
                                 upper.int.age = rep(NA, num.iterations),
                                 upper.uncert = rep(NA, num.iterations),
                                 MSWD = rep(NA, num.iterations) )
output.isoplotr.3 <- data.frame( lower.int.age = rep(NA, num.iterations),
                                 lower.uncert = rep(NA, num.iterations),
                                 upper.int.age = rep(NA, num.iterations),
                                 upper.uncert = rep(NA, num.iterations),
                                 MSWD = rep(NA, num.iterations) )
output.isoplotr.wm <- data.frame( mean = rep(NA, num.iterations ),
                                  err = rep(NA, num.iterations ))
output.modeling.lower.disc.sum.total <- list(NA)
output.modeling.lower.disc <- list(NA)
output.synth.data <- list(NA)

for( j in 1:num.iterations) {
  # j = 1

  
  for( k in 1:num.data.points) {
    temp.category <- sample( 1:3,1, replace=T)  # which of the three upper intercepts should it be?
    if( temp.category == 1 ) {
      temp.upper.age <- runif( 1, min = upper.int.age.min.1, max = upper.int.age.max.1 ) # randomly select an upper intercept
    } else { if( temp.category == 2 ) {
      temp.upper.age <- runif( 1, min = upper.int.age.min.2, max = upper.int.age.max.2 )
    } else {
      temp.upper.age <- runif( 1, min = upper.int.age.min.3, max = upper.int.age.max.3 )
    }
    }
    data.temp$sample.name[ k ] <- temp.category

        temp.proportion <- runif( 1, min = proportion.min, max = proportion.max ) # how far between the upper and lower intercepts will it be?
    temp.uncert <- runif( 1, min = uncertainty.min, max = uncertainty.max ) # what's the uncertainty? between 0.5% and 2% - 2SE
    
    x.values <- c( ratio735(lower.int.age*1e6), ratio735(temp.upper.age*1e6) ) # calculate 7/35 ratios
    y.values <- c( ratio638(lower.int.age*1e6), ratio638(temp.upper.age*1e6) ) # calculate 6/38 ratios
    
    data.temp$ratio735[ k ] <- ( ( x.values[2] - x.values[1] ) * ( temp.proportion / 100 ) ) + x.values[1]
    data.temp$err.735[ k ] <- data.temp$ratio735[ k ] * temp.uncert
    data.temp$ratio638[ k ] <- ( ( y.values[2] - y.values[1] ) * ( temp.proportion / 100 ) ) + y.values[1]
    data.temp$err.638[ k ] <- data.temp$ratio638[ k ] * temp.uncert
    data.temp$rho[ k ] <- runif( n = 1, min = rho.min, max = rho.max ) # randomly select the rho value
    data.temp$age76[ k ] <- age76( data.temp$ratio735[ k ] * (1 / data.temp$ratio638[ k ] ) * (1 / 137.88) )
  }
  
  data.temp.concordia.all <- read.data(  data.temp[ , 2:6], ierr = 2,
                                       method = 'U-Pb', format = 1 )


  #read data into isoplotr
  # now do the isoplot calculations
  ## subset to create data
  data.temp.1 <- subset( data.temp, sample.name == 1 )
  data.temp.2 <- subset( data.temp, sample.name == 2 )
  data.temp.3 <- subset( data.temp, sample.name == 3 )
  
  data.temp.concordia.1 <- read.data(  data.temp.1[ , 2:6], ierr = 2,
                                     method = 'U-Pb', format = 1 )
  data.temp.concordia.2 <- read.data(  data.temp.2[ , 2:6], ierr = 2,
                                       method = 'U-Pb', format = 1 )
  data.temp.concordia.3 <- read.data(  data.temp.3[ , 2:6], ierr = 2,
                                       method = 'U-Pb', format = 1 )
  temp.concordia.1 <- concordia( data.temp.concordia.1, type = 1, show.age = 4  )
  temp.concordia.2 <- concordia( data.temp.concordia.2, type = 1, show.age = 4  )
  temp.concordia.3 <- concordia( data.temp.concordia.3, type = 1, show.age = 4  )
  
  ## now run our modeling reduction for the dataset
  # Define the sample name, and node spacing
  node.spacing	<-1
  
  #############################  SWITCHES #############################
  ## this should be "Y" to normalize the uncertainties to the median value
  #    otherwise it doesn't do anything
  normalize.uncertainty	 <- "N"  
  
  ## this should be "detrital" to weight against concordant analyses
  #    otherwise it should be 'single' to not weight against concordant analyses
  data.type	 <- "single"  
  
  ## If cut.data.by.ratios is Y it trim the input data by the cuts below
  cut.data.by.ratios	<- "N"
  ## These are the start and ends, in ratio space, this cuts data out of the data file
  startcut.r75        <- 0
  endcut.r75          <- 20
  startcut.r68        <- 0
  endcut.r68          <- 0.8
  
  # This zooms the plots into a certain age window
  #  		Use this to either simply zoom in on a particular age, or 
  #  		to zoom in and use a very tight node spacing to save computational time	
  #       it doesn't perform the analysis outside of the age window defined below
  zoom.analysis		<- "Y"
  ## These are the start and ends, only performs the reduction on certain nodes defined
  #	by the ages below here
  startcut.age.lower        <- 0 			## Age in Ma
  endcut.age.lower          <- 100		  ## Age in Ma
  startcut.age.upper        <- 900 		## Age in Ma
  endcut.age.upper          <- 2000		  ## Age in Ma
  
  data.test <- data.temp
  source( "UPb_Reduction_Resample.R" )  ## do the reduction
  
  ## save outputs
  output.modeling.lower.disc.sum.total[[j]] <- lowerdisc.sum.total
  output.modeling.lower.disc[[j]] <- lowerdisc
  output.synth.data[[j]] <- data.temp
  
  output.isoplotr.1$lower.int.age[ j ] <- temp.concordia.1$par[ 1 ]
  output.isoplotr.1$lower.uncert[ j ] <- temp.concordia.1$err[ 1 ]
  output.isoplotr.1$upper.int.age[ j ] <- temp.concordia.1$par[ 2 ]
  output.isoplotr.1$upper.uncert[ j ] <- temp.concordia.1$err[ 2 ]
  output.isoplotr.1$MSWD[ j ] <- temp.concordia.1$mswd
  
  output.isoplotr.2$lower.int.age[ j ] <- temp.concordia.2$par[ 1 ]
  output.isoplotr.2$lower.uncert[ j ] <- temp.concordia.2$err[ 1 ]
  output.isoplotr.2$upper.int.age[ j ] <- temp.concordia.2$par[ 2 ]
  output.isoplotr.2$upper.uncert[ j ] <- temp.concordia.2$err[ 2 ]
  output.isoplotr.2$MSWD[ j ] <- temp.concordia.2$mswd
  
  output.isoplotr.3$lower.int.age[ j ] <- temp.concordia.3$par[ 1 ]
  output.isoplotr.3$lower.uncert[ j ] <- temp.concordia.3$err[ 1 ]
  output.isoplotr.3$upper.int.age[ j ] <- temp.concordia.3$par[ 2 ]
  output.isoplotr.3$upper.uncert[ j ] <- temp.concordia.3$err[ 2 ]
  output.isoplotr.3$MSWD[ j ] <- temp.concordia.3$mswd
  
  
  weighted.mean.isoplot <- IsoplotR::weightedmean( cbind( c( temp.concordia.1$par[ 1 ], 
                                                   temp.concordia.2$par[ 1 ], 
                                                   temp.concordia.3$par[ 1 ] ),
                                                c( temp.concordia.1$err[ 1 ] * 2,
                                                   temp.concordia.2$err[ 1 ] * 2,
                                                   temp.concordia.3$err[ 1 ] * 2 ) ) )
  
  
  output.isoplotr.wm$mean[ j ] <- weighted.mean.isoplot$mean[ 1 ]
  output.isoplotr.wm$err[ j ] <- weighted.mean.isoplot$mean[ 2 ]
  
  print( j )
  
}


## FWHM function
calculate_low.half <- function(dataframe) {
  
  # dataframe = output.modeling.lower.disc.sum.total[[1]]
  # Find peak likelihood value
  peak_likelihood <- max(dataframe$normalized.sum.likelihood)
  
  # Find index of peak likelihood value
  peak_index <- which.max(dataframe$normalized.sum.likelihood)
  
  # Find half maximum value
  half_max <- peak_likelihood / 2
  
  # Find index of the point closest to half maximum on the left side of the peak
  left_index <- max( which( dataframe$normalized.sum.likelihood[1:peak_index] <= half_max ) )
  
  # Find index of the point closest to half maximum on the right side of the peak
  right_index <- min(which(dataframe$normalized.sum.likelihood[peak_index:length(dataframe$normalized.sum.likelihood)] <= half_max ) ) + peak_index
  
  # Calculate FWHM
  low.peak.half.uncert <-  dataframe$`Lower Intercept`[left_index] 
  low.peak.half.uncert / 1e6
}

## FWHM function
calculate_high.half <- function(dataframe) {
  
  # dataframe = output.modeling.lower.disc.sum.total[[1]]
  # Find peak likelihood value
  peak_likelihood <- max(dataframe$normalized.sum.likelihood)
  
  # Find index of peak likelihood value
  peak_index <- which.max(dataframe$normalized.sum.likelihood)
  
  # Find half maximum value
  half_max <- peak_likelihood / 2
  
  # Find index of the point closest to half maximum on the left side of the peak
  left_index <- max( which( dataframe$normalized.sum.likelihood[1:peak_index] <= half_max ) )
  
  # Find index of the point closest to half maximum on the right side of the peak
  right_index <- min(which(dataframe$normalized.sum.likelihood[peak_index:length(dataframe$normalized.sum.likelihood)] <= half_max ) ) + peak_index
  
  # Calculate FWHM
  high.peak.half.uncert <-  dataframe$`Lower Intercept`[right_index] 
  high.peak.half.uncert / 1e6
}

#peak finding function
list.maxposition <- lapply(output.modeling.lower.disc.sum.total,function(df)which.max(df$normalized.sum.likelihood))
list.maxlikelihood <- lapply(output.modeling.lower.disc.sum.total,function(df)max(df$normalized.sum.likelihood))
data.max.age <- data.frame( max.position=unlist(list.maxposition),
                            max.likelihood=unlist(list.maxlikelihood))
data.max.age$age <- data.max.age$max.position * node.spacing + startcut.age.lower
data.max.age$min.ratio735.analysis <- unlist( lapply( output.synth.data, function(df) min(df$ratio735)) )
data.max.age$min.ratio638.analysis <- unlist( lapply( output.synth.data, function(df) min(df$ratio638)) )
data.max.age$low.uncert <- unlist( lapply( output.modeling.lower.disc.sum.total, calculate_low.half ) )
data.max.age$high.uncert <- unlist( lapply( output.modeling.lower.disc.sum.total, calculate_high.half ) )



## plot names


## plot isoplot age vs our reduction age
data.lowerage.combined <- cbind( output.isoplotr.wm, data.max.age) 


ggplot( data.lowerage.combined, aes( x = lower.int.age, y = age ) ) +
  fte_theme_white() +
  labs( x = "IsoplotR Lower Int (Ma)", 
        y = "Discordance Lower Int (Ma)" ) +
  ylim( 10, 70 ) +
  xlim( 10, 70 ) +
  geom_hline( yintercept = 30 ) + geom_vline( xintercept = 30 ) +
  geom_errorbarh( aes( xmax = lower.int.age + err, xmin = lower.int.age - err ),
                  height = 0.2, color = "cadetblue4" ) +
  geom_errorbar( aes( ymax = high.uncert, ymin = low.uncert ),
                 color = "cadetblue4"  ) +
  geom_point( size = 4, color = "cadetblue")


### plot lower intercept age vs upper intercept age
ggplot( data.lowerage.combined, aes( x = lower.int.age, y = err ) ) +
  fte_theme_white() +
  labs( x = "IsoplotR Lower Int (Ma)", 
        y = "IsoplotR Lower Uncert (Ma)" ) +
  # ylim( 10, 25 ) +
  # xlim( 20, 40 ) +
  geom_point( size = 3, color = "cadetblue")



### plot model lower age vs the peak height as a proxy for uncertainty
ggplot( data.lowerage.combined, aes( x = age, y = max.likelihood, color = min.ratio735.analysis ) ) +
  fte_theme_white() +
  labs( x = "Discordance Max Age (Ma)", 
        y = "Max Likelihood" ) +
  scale_color_viridis( ) +
  geom_vline( xintercept = 30 ) +
  geom_errorbarh( aes( xmax = high.uncert, xmin = low.uncert ),
                  height = 0.2  ) +
  # ylim( 10, 40 ) +
  # xlim( 10, 40 ) +
  geom_point( size = 3 )


#plot discordance model outputs
ggplot(dplyr::bind_rows(output.modeling.lower.disc.sum.total, .id="data_frame"),
       aes(x=`Lower Intercept`/1e6, y=normalized.sum.likelihood, color=data_frame)) +
  geom_line( show.legend = FALSE, alpha = 0.2 ) +
  geom_vline( xintercept = lower.int.age ) +
  xlab( 'Intercept Age (Ma)' ) +
  ylab( 'Normalized Likelihood' ) +
  fte_theme_white() + 
  scale_colour_grey( start = 0.3, end = 0.31 ) 



#plot distribution of intercepts
ggplot(data = data.lowerage.combined, aes( x = lower.int.age ) ) +
  # geom_line( show.legend = FALSE, alpha = 0.2 ) +
  geom_vline( xintercept = lower.int.age ) +
  xlab( 'Intercept Age (Ma)' ) +
  ylab( 'Normalized Likelihood' ) +
  fte_theme_white() + 
  geom_density( fill = "cadetblue"  ) +
  geom_density( aes( x = age ), fill = "darkblue", alpha = 0.5 ) 



write.csv( output.modeling.lower.disc.sum.total, 
           file= paste( plot.name,"_lowerdiscsumtotal.csv", sep=""))
write.csv( output.modeling.lower.disc, 
           file= paste( plot.name,"_lowerdisc.csv", sep=""))
write.csv( output.synth.data, 
           file= paste( plot.name, "_datasets.csv", sep=""))
write.csv( output.isoplotr.wm, 
           file= paste( plot.name, "_IsoplotOutputs.csv", sep=""))
write.csv( data.lowerage.combined, 
           file= paste( plot.name, "_results.csv", sep=""))






          