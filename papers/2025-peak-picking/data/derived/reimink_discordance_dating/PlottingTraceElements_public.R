
library(ggplot2)

## set your working directory
setwd( "/Users/jessereimink/My Drive (jxr1350@psu.edu)/Research/PSU/Projects/Basin Fluid Flow/Discordance Dating Manuscript/CodeForGithub")

data.sample.2 <- read.csv( "Sample_2_All_LabeledPoints.csv" )

### Alpha dose
alpha.dose <- function( u.conc, th.conc, age ) {
  ## calculate zircon alpha dose at age in Ma
  8*(  ( u.conc * 6.022148e+23 * 0.9938 )/( 238.05078826*1000 )) * ( exp(1.55125e-10*age*1e6)-1) +
    7*( ( u.conc*6.022148e+23*0.0073)/(235.0439299 *1000)) * (exp(9.8485e-10*age*1e6)-1 )  +
    6*( ( th.conc*6.022148e+23)/(232.03805*1000))*(exp(4.948e-11*age*1e6)-1)	
}


data.sample.2$alpha.dose <- alpha.dose( data.sample.2$U,
                                        data.sample.2$Th,
                                        data.sample.2$Pb207.Pb206_age_mean )

### U vs discordance plot
ggplot( data.sample.2, aes( x = X._Discordance , 
                            y =  log( U ) , 
                            color = log( Al ) ) ) +
  viridis::scale_color_viridis() +
  geom_point( size = 4 )

### alpha dose vs discordance plot
ggplot( data.sample.2, aes( x = X._Discordance , 
                            y =  log( alpha.dose ) , 
                            color = log( Al ) ) ) +
  viridis::scale_color_viridis() +
  geom_point( size = 4 )

### alpha dose vs discordance plot
ggplot( data.sample.2, aes( x = X._Discordance , 
                            y =  Pb204 , 
                            color = log( Al ) ) ) +
  viridis::scale_color_viridis() +
  ylim( -3, 15 ) + ## remove one very low data point from plotting
  geom_point( size = 4 )


### alpha dose vs discordance plot
ggplot( data.sample.2, aes( x = X._Discordance , 
                            y =  Pb204 , 
                            color = log( Al ) ) ) +
  viridis::scale_color_viridis() +
  ylim( -3, 15 ) + ## remove one very low data point from plotting
  geom_point( size = 4 )

### alpha dose vs discordance plot
ggplot( data.sample.2, aes( x = X._Discordance , 
                            y =  Th/U , 
                            color = log( Al ) ) ) +
  viridis::scale_color_viridis() +
  geom_point( size = 4 )
ggsave("th.u.plot.pdf", width = 10, height = 7 )

          