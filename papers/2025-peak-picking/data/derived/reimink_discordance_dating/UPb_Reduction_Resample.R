#### Data reduction code for performing the UPb data reduction based on inputs
#  		defined in the ResamplingCode_Public.R file

## DO NOT TOUCH CODE BELOW HERE FOR FEAR OF BREAKING IT

if( zoom.analysis == "Y" ) {
} else {
	startcut.age.lower        <- 0 			## Age in Ma
	endcut.age.lower          <- 4500		## AGe in Ma
	startcut.age.upper        <- 0 			## Age in Ma
	endcut.age.upper          <- 4500		## Age in Ma
}

# set input variables, these all are used for the main 'grid' not the dataset. 
Tstep         = node.spacing * 1e6
# define the maximum of the y-axis
maximus       = 75
# number of data points in each block
Number        = 25

## Don't touch anything below here
colnames(data.test)     <- c( "Spot", "r75", "sigma75", "r68", "sigma68", "rho", "age76")
Data.new    <- data.test

## force numeric for Data.new
Data.new$r75		<- as.numeric( as.character( Data.new$r75 ) )
Data.new$sigma75	<- as.numeric( as.character( Data.new$sigma75 ) )
Data.new$r68		<- as.numeric( as.character( Data.new$r68 ) )
Data.new$sigma68	<- as.numeric( as.character( Data.new$sigma68 ) )
Data.new$rho		<- as.numeric( as.character( Data.new$rho ) )
Data.new$age76		<- as.numeric( as.character( Data.new$age76 ) )


## Back to old code
deltaT        = 100*10^6
concTstep     = Tstep

# f is the uncertainty interval in sigma, should be two
f             = 2   

datapoints            <- nrow(Data.new)
Npoints               <- Number

discordances           <- matrix( c( abs(1 - ( Data.new$r68 / (exp( Lambda238 * Data.new$age76 * 
                                                                    1000000) - 1)))))
Data.reduction              <- data.frame( cbind( Spot = Data.new$Spot, 
											r75 = Data.new$r75, 
											sigma75 = Data.new$sigma75, 
											r68 = Data.new$r68, 
											sigma68 = Data.new$sigma68, 
											rho = Data.new$rho, 
											discordance = as.vector( discordances ) ) )

nfiles                <- ceiling(nrow(Data.reduction)/Number)
groupings              <- c(rep(1:(nfiles - 1), each = Number), 
                                         rep(nfiles, times = nrow(Data.reduction) - 
                                               (as.integer(Number*(nfiles - 1))))) 
Data.reduction              <- data.frame( cbind( Data.reduction, 
											GROUP = as.vector( groupings ) ) )

### force numeric for Data.reduction
## force numeric
Data.reduction$r75			<- as.numeric( as.character( Data.reduction$r75 ) )
Data.reduction$sigma75		<- as.numeric( as.character( Data.reduction$sigma75 ) )
Data.reduction$r68			<- as.numeric( as.character( Data.reduction$r68 ) )
Data.reduction$sigma68		<- as.numeric( as.character( Data.reduction$sigma68 ) )
Data.reduction$rho			<- as.numeric( as.character( Data.reduction$rho ) )
Data.reduction$discordance	<- as.numeric( as.character( Data.reduction$discordance ) )
Data.reduction$GROUP		<- as.numeric( as.character( Data.reduction$GROUP ) )


data.split            <- split(Data.reduction, Data.reduction$GROUP)


a                          <- as.vector( seq( from = startcut.age.lower * 1e6, 
											  to = endcut.age.lower * 1e6 - Tstep, 
											  by = Tstep ) )
b                          <- as.vector( seq( from = startcut.age.upper * 1e6 + Tstep, 
											  to = endcut.age.upper * 1e6, 
											  by = Tstep ) )
DiscGrid                   <- expand.grid( a, b )
DiscGrid2                  <- subset( DiscGrid, Var1 < Var2)
DiscGrid3                  <- DiscGrid2[ with( DiscGrid2, order( Var1 )), ]
DiscGridTable              <- DiscGrid3
colnames(DiscGridTable)    <- c( "Lower Intercept", "Upper Intercept" )

DiscGridTableA                <- mapply(afff, DiscGridTable["Lower Intercept"], 
                                        DiscGridTable["Upper Intercept"])
colnames(DiscGridTableA)      <- "Slope" 
DiscGridTableB                <- mapply(bfff, DiscGridTable["Lower Intercept"], 
                                        DiscGridTable["Upper Intercept"])
colnames(DiscGridTableB)      <- "Yintercept" 
DiscGridTableFinal            <- cbind(DiscGridTable[1:2], DiscGridTableA, DiscGridTableB)
DiscGridTableFinal            <- DiscGridTableFinal[c(3, 4, 1, 2)]
row.names(DiscGridTableFinal) <- seq_len(nrow(DiscGridTableFinal))
DiscGridTableFinal$ID         <- seq(1, nrow(DiscGridTableFinal), 1)
Discline                      <- nrow(DiscGrid)   
Disclines                     <- Discline                                    
# rm(a, b, aff, afff, bff, bfff, DiscGrid, DiscGrid2, DiscGrid3, DiscGridTable,
#    DiscGridTableA, DiscGridTableB, discordance)

bigdata              <- by( Data.reduction[, 1:7], Data.reduction$GROUP, BigFunction)
splitfun   <- function(x) {
  bigdata[[x]]$Likelihood
}
for (i in 1:nfiles) {
  assign(paste("res", i), as.data.frame(splitfun(i)))
}


likelis               <- do.call( cbind, lapply( paste( "res", 1:nfiles, sep=" "), get))
totallikelihood       <- apply( likelis, 1, sum, na.rm = T ) 
Resultdisc            <- cbind(bigdata$`1`[, 1:5], as.data.frame(totallikelihood))
colnames(Resultdisc)  <- c("ID", "Slope", "Yintercept", "Lower Intercept", "Upper Intercept", 
                           "Likelihood")
normalized            <- Resultdisc [, "Likelihood"] / datapoints
Resultdisc            <- cbind(Resultdisc, normalized)
upperdisc             <- aggregate (Resultdisc$normalized, 
                                    by = list (Resultdisc [, "Upper Intercept"]), max)
colnames(upperdisc)   <- c("Upper Intercept", "Likelihood")

### Sum up the total probability assigned to all lines with a given lower intercept age
lowerdisc.sum.total   <- aggregate( Resultdisc$normalized, 
                                    by = list(Resultdisc[, "Lower Intercept"]), sum )
colnames( lowerdisc.sum.total )   <- c("Lower Intercept", "Likelihood")

## Calculate the number of lines that have a given lower intercept age
lowerdisc.sum.total$n.lines <-  aggregate( Resultdisc$normalized, 
                                           by = list(Resultdisc[, "Lower Intercept"]), length )[ ,2]
lowerdisc.sum.total$normalized.sum.likelihood <- lowerdisc.sum.total$Likelihood /
  lowerdisc.sum.total$n.lines  ## Calculate the Likelihood normalized by the number of lines

lowerdisc             <- aggregate( Resultdisc$normalized, 
                                     by = list( Resultdisc[, "Lower Intercept" ] ), max )
colnames(lowerdisc)   <- c( "Lower Intercept", "Likelihood" )
# rm(likelis, grouping, bigdata, data.split)















