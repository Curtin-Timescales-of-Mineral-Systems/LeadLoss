## Don't touch anything in here
# activate necessary libraries

library(RColorBrewer)
library(reshape2)
library(cardidates)
library(data.table)
# library(raster)
library(ggplot2)
library(IsoplotR)
library(viridis)
library(viridisLite)


#### Constants
Lambda238             <- 1.55125*10^(-10)
Lambda235             <- 9.8485*10^(-10)
u.isotope.composition		<- 137.818  ## Hiess et al 2012


##### Functions for calculating U/Pb ratios from ages
ratio.76 <- function( age.76.sample ) {
	(1/u.isotope.composition) * ( ( exp(Lambda235*age.76.sample*1e6)-1) / 
								  	( exp(Lambda238*age.76.sample*1e6)-1) )
}

aff            <- function(x1, y1, x2, y2)  {(y2 - y1)/(x2 - x1)
}
bff            <- function(x1, y1, x2, y2)  {y2 - x2*(y2 - y1)/(x2 - x1)
}
afff           <- function(t1, t2) {aff(exp(Lambda235 * t1) - 1, exp(Lambda238 * t1) - 1,
										exp(Lambda235 * t2) - 1, exp(Lambda238 * t2) - 1)
}
bfff           <- function(t1, t2) {bff(exp(Lambda235 * t1) - 1, exp(Lambda238 * t1) - 1,
										exp(Lambda235 * t2) - 1, exp(Lambda238 * t2) - 1)
}


#### Probability functions and the BigFunction that does the probability calculation
Pro  <- function(a, b, Xi, sX, Yi, sY, rho, disc) { abs(disc) * (1 / (2 * pi * sX * sY)) *
		exp((-1 / 2) * ((((b + a * Xi - Yi) / (cos(atan((2 * rho * sX * sY) /
															(sX ^ 2) - (sY ^ 2)) / 2) + a * sin(atan((2 * rho * sX * sY) /
																									 	(sX ^ 2) - (sY ^ 2)) / 2))) / sY) ^ 2 / (1 + (sX / sY * ((a * 
																									 															  	cos(atan((2 * rho * sX * sY) / (sX ^ 2) - (sY ^ 2)) / 2) -
																									 															  	sin(atan((2 * rho * sX * sY) / (sX ^ 2) - (sY ^ 2)) / 2)) /
																									 															 	(cos(atan((2 * rho * sX * sY) / (sX ^ 2) - (sY ^ 2)) / 2) + a *
																									 															 	 	sin(atan((2 * rho * sX * sY) / (sX ^ 2) - (sY ^ 2)) / 2)))) ^ 2)))
}

Prob   <- function(p1, p2) {
	p1 = as.list(p1); p2 = as.list(p2)
	Pro(p1$Slope, p1$Yintercept, p2$r75, p2$sigma75, p2$r68,
		p2$sigma68, p2$rho, p2$discordance)
}

BigFunction  <- function (x) {
	Npoint                    <- dim(Data.new)
	Npoints                   <- (Npoint[1])
	indexdisc      <- CJ(indexdisc1 = seq( nrow( DiscGridTableFinal )), 
						 indexdisc2 = seq( nrow( x )))
	sumdisc        <- indexdisc[,`:=`(resultdisc = Prob( DiscGridTableFinal[indexdisc1, ], 
														 x[indexdisc2, ]),
									  Group.1 = rep( seq( nrow( DiscGridTableFinal )), 
									  			   each = nrow( x )))][,.(sumdisc = sum( resultdisc )),
									  			   					by = Group.1]
	sumdisc                <- as.data.frame( sumdisc )
	
	colnames(sumdisc)      <- c("ID", "Likelihood")
	Resultdisc             <- merge(DiscGridTableFinal, sumdisc, by = "ID", all.x = TRUE)
	row.names(Resultdisc)  <- seq_len(nrow(Resultdisc))
	rm(indexdisc, sumdisc)
	assign(paste("Resultdisc"), Resultdisc)
}


## U-Pb equations
ratio76    = function(age) {   ## Age in years
  ( 1 / u.isotope.composition ) * ( ( exp(Lambda235 * age) - 1) / (exp(Lambda238 * age) - 1) ) 
}

ratio735    = function(age) {   ## Age in years
  exp( Lambda235 * age ) - 1 
}

ratio638    = function(age) {   ## Age in years
  exp( Lambda238 * age ) - 1 
}

## Iterative function for calculating 7/6 ages 
age76     <- function( ratio ) {
  t     <- -0.01
  tol   <- 1e-5
  repeat {
    R75    <- exp( Lambda235 * t ) - 1
    R68    <- exp( Lambda238 * t ) - 1
    func   <- R75 / R68 / u.isotope.composition
    der    <- ( ( Lambda235 * exp ( Lambda235 * t) ) - ( Lambda238 * exp( Lambda238 * t ) ) * 
                  ( R75 / R68 ) ) / R68 / u.isotope.composition
    delta  <- ( ratio - func ) / der
    t      <- t + delta
    if ( abs( delta ) < tol ) break
    
  }
  t / 1e6
}


## Using ggplot2 for the plotting

## Set the ggplot2 theme as all grey backgrounds
#   Use the same theme for all the plots!!  This is great...
fte_theme <- function() {
	
	library( RColorBrewer )
	# Generate the colors for the chart procedurally with RColorBrewer
	palette          <- brewer.pal( "Greys", n = 9 )
	color.background = palette[ 2 ]
	color.grid.major = palette[ 4 ]
	color.axis.text  = palette[ 7 ]
	color.axis.title = palette[ 8 ]
	color.title      = palette[ 9 ]
	
	# Begin construction of chart
	theme_bw( base_size = 9 ) +
		
		# Set the entire chart region to a light gray color
		theme( panel.background = element_rect( fill = color.background, color = color.background ) ) +
		theme( plot.background  = element_rect( fill = color.background, color = color.background ) ) +
		theme( panel.border     = element_rect( color = color.background ) ) +
		
		# Format the grid
		theme( panel.grid.major = element_line( color = color.grid.major, size = 0.25 ) ) +
		theme( axis.line.x      = element_line( color = color.grid.major, size = 0.35 ) ) +
		theme( axis.line.y      = element_line( color = color.grid.major, size = 0.35 ) ) +
		theme( panel.grid.minor = element_blank( ) ) +
		theme( axis.ticks       = element_blank( ) ) +
		theme( legend.key = element_rect( fill = color.background ) ) +
		
		
		# Format the legend
		# theme( legend.position = "none" ) +
		theme( legend.background = element_rect( fill = color.background ) ) +
		theme( legend.text       = element_text( size = 7, color = color.axis.title ) ) +
		
		# Set title and axis labels, and format these and tick marks
		theme( plot.title   = element_text( color = color.title, size = 25, vjust = 1.25, hjust = 0.5 ) ) +
		theme( plot.subtitle   = element_text( color = color.title, size = 16, vjust = 1.25, hjust = 0.5 ) ) +
		theme( axis.text.x  = element_text( size = 28, color = color.axis.text ) ) +
		theme( axis.text.y  = element_text( size = 28, color = color.axis.text ) ) +
		theme( axis.title.x = element_text( size = 36, color = color.axis.title, vjust = 0 ) ) +
		theme( axis.title.y = element_text( size = 36, color = color.axis.title, vjust = 1.25 ) ) +
		
		# Plot margins
		theme( plot.margin = unit( c( 0.35, 0.2, 0.3, 0.35 ), "cm" ) )
}



## Set the ggplot2 theme as all grey backgrounds
#   Use the same theme for all the plots!!  
fte_theme_black <- function() {
	
	library( RColorBrewer )
	# Generate the colors for the chart procedurally with RColorBrewer
	palette          <- brewer.pal( "Greys", n = 9 )
	color.background = palette[ 9 ]
	color.grid.major = palette[ 4 ]
	color.axis.text  = palette[ 1 ]
	color.axis.title = palette[ 1 ]
	color.title      = palette[ 1 ]
	
	
	# Begin construction of chart
	theme_bw( base_size = 9 ) +
		
		# Set the entire chart region to a light gray color
		theme( panel.background = element_rect( fill = color.background, color = color.background ) ) +
		theme( plot.background  = element_rect( fill = color.background, color = color.background ) ) +
		theme( panel.border     = element_rect( color = color.background ) ) +
		
		# Format the grid
		theme( panel.grid.major = element_line( color = color.grid.major, size = 0.25 ) ) +
		theme( axis.line.x      = element_line( color = color.grid.major, size = 0.35 ) ) +
		theme( axis.line.y      = element_line( color = color.grid.major, size = 0.35 ) ) +
		theme( panel.grid.minor = element_blank( ) ) +
		theme( axis.ticks       = element_blank( ) ) +
		theme( legend.key = element_rect( fill = color.background ) ) +
		
		
		# Format the legend
		# theme( legend.position = "none" ) +
		theme( legend.background = element_rect( fill = color.background ) ) +
		theme( legend.text       = element_text( size = 7, color = color.axis.title ) ) +
		
		# Set title and axis labels, and format these and tick marks
		theme( plot.title   = element_text( color = color.title, size = 25, vjust = 1.25, hjust = 0.5 ) ) +
		theme( axis.text.x  = element_text( size = 28, color = color.axis.text ) ) +
		theme( axis.text.y  = element_text( size = 28, color = color.axis.text ) ) +
		theme( axis.title.x = element_text( size = 36, color = color.axis.title, vjust = 0 ) ) +
		theme( axis.title.y = element_text( size = 36, color = color.axis.title, vjust = 1.25 ) ) +
		
		# Plot margins
		theme( plot.margin = unit( c( 0.35, 0.2, 0.3, 0.35 ), "cm" ) )
}


multiplot <- function(..., plotlist=NULL, cols) {
	require(grid)
	
	# Make a list from the ... arguments and plotlist
	plots <- c(list(...), plotlist)
	
	numPlots = length(plots)
	
	# Make the panel
	plotCols = cols                          # Number of columns of plots
	plotRows = ceiling(numPlots/plotCols) # Number of rows needed, calculated from # of cols
	
	# Set up the page
	grid.newpage()
	pushViewport(viewport(layout = grid.layout(plotRows, plotCols)))
	vplayout <- function(x, y)
		viewport(layout.pos.row = x, layout.pos.col = y)
	
	# Make each plot, in the correct location
	for (i in 1:numPlots) {
		curRow = ceiling(i/plotCols)
		curCol = (i-1) %% plotCols + 1
		print(plots[[i]], vp = vplayout(curRow, curCol ))
	}
	
}


fte_theme_white <- function() {
	
	library( RColorBrewer )
	# Generate the colors for the chart procedurally with RColorBrewer
	palette          <- brewer.pal( "Greys", n = 9 )
	# color.background = palette[ 2 ]
	color.grid.major = palette[ 4 ]
	color.axis.text  = palette[ 7 ]
	color.axis.title = palette[ 8 ]
	color.title      = palette[ 9 ]
	
	# Begin construction of chart
	theme_bw( base_size = 9 ) +
		
		# Set the entire chart region to a light gray color
		# theme( panel.background = element_rect( fill = color.background, color = color.background ) ) +
		# theme( plot.background  = element_rect( fill = color.background, color = color.background ) ) +
		# theme( panel.border     = element_rect( color = color.background ) ) +
		
		# Format the grid
		theme( panel.grid.major = element_line( color = color.grid.major, size = 0.25 ) ) +
		theme( axis.line.x      = element_line( color = color.grid.major, size = 0.35 ) ) +
		theme( axis.line.y      = element_line( color = color.grid.major, size = 0.35 ) ) +
		theme( panel.grid.minor = element_blank( ) ) +
		theme( axis.ticks       = element_blank( ) ) +
		# theme( legend.key = element_rect( fill = color.background ) ) +
		
		
		# Format the legend
		# theme( legend.position = "none" ) +
		# theme( legend.background = element_rect( fill = color.background ) ) +
		theme( legend.text       = element_text( size = 7, color = color.axis.title ) ) +
		
		# Set title and axis labels, and format these and tick marks
		theme( plot.title   = element_text( color = color.title, size = 25, vjust = 1.25, hjust = 0.5 ) ) +
		theme( plot.subtitle   = element_text( color = color.title, size = 16, vjust = 1.25, hjust = 0.5 ) ) +
		theme( axis.text.x  = element_text( size = 28, color = color.axis.text ) ) +
		theme( axis.text.y  = element_text( size = 28, color = color.axis.text ) ) +
		theme( axis.title.x = element_text( size = 36, color = color.axis.title, vjust = 0 ) ) +
		theme( axis.title.y = element_text( size = 36, color = color.axis.title, vjust = 1.25 ) ) +
		
		# Plot margins
		theme( plot.margin = unit( c( 0.35, 0.2, 0.3, 0.35 ), "cm" ) )
}
