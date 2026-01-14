library(data.table)
library(reshape2)
library(lattice)
library(boot)
library(pastecs)
library(raster)

# Data should have format as follows
# sample name, ratio 7/5, 1sigma75 (abs), ratio 6/8, 1sigma 68 (abs), rho, age76 in Ma

# download data file and view and define Npoints for future calculations
DataTitle  <- "Lucy"
Data.raw <- read.csv("/Users/lucymathieson/Desktop/1a_reimink.csv")

# set input variables, these all are used for the main 'grid' not the dataset. 
Tstep         = 10*10^6
# define the maximum of the y-axis
maximus       = 75
# number of data points in each block
Number        = 25


## Don't touch anything below here
# activate necessary libraries



Data.real   <- Data.raw
colnames(Data.real)    <- c("Spot", "r75", "sigma75", "r68", "sigma68", "rho", "age76")
Data.new    <- Data.raw
colnames(Data.raw)     <- c("Spot", "r75", "sigma75", "r68", "sigma68", "rho", "age76")

colnames(Data.new)    <- c("Spot", "r75", "sigma75", "r68", "sigma68", "rho", "age76")
sigma75mean           <- median( Data.new[ , "sigma75"])
sigma68mean           <- median( Data.new[ , "sigma68"])
Data.new["sigma75"]   <- sigma75mean
Data.new["sigma68"]   <- sigma68mean

deltaT        = 100*10^6
concTstep     = 100*10^3

f             = 2
Tmin          = 0.0;
Tmax          = 4.5*10^9;

ptm <- proc.time()

datapoints            <- nrow(Data.new)
Npoints               <- Number
concNlines            <- Tmax/concTstep + 1;

Lambda238             <- 1.55125*10^(-10)
Lambda235             <- 9.8485*10^(-10)

discordance           <- matrix( c( abs(1 - (Data.new$r68 / (exp( Lambda238 * Data.new$age76 * 
                                                                    1000000) - 1)))))
Data.new              <- data.frame(cbind(Data.new$Spot, Data.new$r75, Data.new$sigma75, 
                                          Data.new$r68, 
                                          Data.new$sigma68, Data.new$rho, discordance))
colnames(Data.new)    <- c("Spot", "r75", "sigma75", "r68", "sigma68", "rho", "discordance")

nfiles                <- ceiling(nrow(Data.new)/Number)
grouping              <- as.data.frame(c(rep(1:(nfiles - 1), each = Number), 
                                         rep(nfiles, times = nrow(Data.new) - 
                                               (as.integer(Number*(nfiles - 1))))))
Data.new              <- cbind(Data.new, grouping)
colnames(Data.new)    <- c("Spot", "r75", "sigma75", "r68", "sigma68", "rho", "discordance", "GROUP")

data.split            <- split(Data.new, Data.new$GROUP)

a                          <- as.vector( seq( from = 0, to = Tmax - Tstep, by = Tstep ))
b                          <- as.vector( seq( from = Tmin + Tstep, to = Tmax, by = Tstep))
DiscGrid                   <- expand.grid( a, b )
DiscGrid2                  <- subset( DiscGrid, Var1 < Var2)
DiscGrid3                  <- DiscGrid2[ with( DiscGrid2, order( Var1 )), ]
DiscGridTable              <- DiscGrid3
colnames(DiscGridTable)    <- c( "Lower Intercept", "Upper Intercept" )

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
rm(a, b, aff, afff, bff, bfff, DiscGrid, DiscGrid2, DiscGrid3, DiscGridTable,
   DiscGridTableA, DiscGridTableB, discordance)



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
bigdata              <- by(Data.new[, 1:7], Data.new$GROUP, BigFunction)
splitfun   <- function(x) {
  bigdata[[x]]$Likelihood
}
for (i in 1:nfiles) {
  assign(paste("res", i), as.data.frame(splitfun(i)))
}


likelis               <- do.call(cbind, lapply(paste("res", 1:nfiles, sep=" "), get))
totallikelihood       <- apply(likelis, 1, sum) 
Resultdisc            <- cbind(bigdata$`1`[, 1:5], as.data.frame(totallikelihood))
colnames(Resultdisc)  <- c("ID", "Slope", "Yintercept", "Lower Intercept", "Upper Intercept", 
                           "Likelihood")
normalized            <- Resultdisc [, "Likelihood"] / datapoints
Resultdisc            <- cbind(Resultdisc, normalized)
upperdisc             <- aggregate (Resultdisc$normalized, 
                                    by = list (Resultdisc [, "Upper Intercept"]), max)
colnames(upperdisc)   <- c("Upper Intercept", "Likelihood")

lowerdisc             <- aggregate (Resultdisc$normalized, 
                                    by = list(Resultdisc[, "Lower Intercept"]), max)
colnames(lowerdisc)   <- c("Lower Intercept", "Likelihood")
rm(likelis, grouping, bigdata, data.split)

## EXTRA PLOTTING CODE
Title    <- paste(DataTitle, "\nn =", datapoints, "\nNode Spacing (myr) =", Tstep/1000000)
res1     <- dcast(Resultdisc, `Upper Intercept` ~ `Lower Intercept`, 
                  value.var = "normalized")[-1]
res2     <- as.matrix(res1)
rf       <- colorRampPalette(c("White", "grey85", "tan1", "darkorange", "royalblue1", "royalblue4" ),
                             bias = 1)
r              <- rf(32)
Lambda238      <- 1.55125*10^(-10)
Lambda235      <- 9.8485*10^(-10)
age68    = function(age) {
  exp(Lambda238 * age) - 1
}
age75    = function(age) {
  exp(Lambda235 * age) - 1
}
ageX     <- c(seq(0.5e09, 4.5e09, 0.5e09))
ratio75  <- matrix(c(age75(ageX)))
ratio68  <- matrix(c(age68(ageX)))
ages     <- matrix(c(seq(0.5, 4.5, 0.5)))
ratios   <- cbind(ratio75, ratio68, ages)
eq1      <- function(x, Xo, Yo, sigx, sigy, rho, f) {
  1 / sigx^2 * (Yo * sigx^2 + x * rho * sigx * sigy - Xo * rho * sigx * sigy -
                  sqrt( -x^2 * sigx^2 * sigy^2 + 2 * x * Xo * sigx^2 * sigy^2 - 
                          Xo^2 * sigx^2 * sigy^2 + x^2 * rho^2 * sigx^2 * sigy^2 - 
                          2 * x * Xo * rho^2 * sigx^2 * sigy^2 + Xo^2 * rho^2 * sigx^2 * 
                          sigy^2 + 2 * f^2 * sigx^4 * sigy^2 - 2 * f^2 * rho^2 * sigx^4 
                        * sigy^2))
}
eq2   <- function(x, Xo, Yo, sigx, sigy, rho, f) {
  1 / sigx^2 * (Yo * sigx^2 + x * rho * sigx * sigy - Xo * rho * sigx * sigy +
                  sqrt( -x^2 * sigx^2 * sigy^2+ 2 * x * Xo * sigx^2 * sigy^2 - 
                          Xo^2 * sigx^2 * sigy^2 + x^2 * rho^2 * sigx^2 * sigy^2 - 
                          2 * x * Xo * rho^2 * sigx^2 * sigy^2 + Xo^2 * rho^2 * sigx^2 * 
                          sigy^2 + 2 * f^2 * sigx^4 * sigy^2 - 2 * f^2 * rho^2 * sigx^4 
                        * sigy^2))
}
conc   <- function(x) {
  ((x +1) ^ 0.15751129613647) - 1
}
xtable      <- matrix(c(seq(0.005, 50, 0.03)))
ytable      <- matrix(c(conc(xtable)))
concordia   <- cbind(xtable, ytable)

xtable      <- matrix(c(seq(0.005, 50, 0.0006)))
pointsA      <- matrix(c(eq1(xtable, Data.real[, 2], Data.real[, 4], Data.real[, 3], 
                             Data.real[, 5], Data.real[, 6], 2)))
pointsAA     <- cbind(xtable, pointsA)

pointsB      <- matrix(c(eq2(xtable, Data.real[, 2], Data.real[, 4], Data.real[, 3], 
                             Data.real[, 5], Data.real[, 6], 2)))
pointsBB     <- cbind(xtable, pointsB)

pointsC      <- rbind(pointsAA, pointsBB)
pointsCC     <- subset(pointsC, pointsC[, 2] > 0 )
fig.conc   <- function() {
  par(mar = c(5, 5, 0, 1))
  plot(concordia, pch = 20, cex = 0.05, xlim = c(0, max(Data.real[, 2] + 1)), 
       ylim = c(0, max(Data.real[, 4] + 0.05)), xaxs = "i", yaxs = "i", 
       xlab = expression(paste(" " ^ "207", "Pb", " /", " "^"235", "U")), 
       ylab = expression(paste(" " ^ "206", "Pb", " /", " "^"238", "U")))
  par(new = TRUE)
  plot(ratios, pch = 20, cex = 1, xlim = c(0, max(Data.real[, 2] + 1)), 
       ylim = c(0, max(Data.real[, 4] + 0.05)), xaxs = "i", yaxs = "i", 
       axes = F, ann = F)
  par(new = TRUE)
  text(ratios[, 1], ratios[, 2], ratios[, 3], cex = 0.8, pos = 3)
  par(new = TRUE)
  #   plot(pointsCC, pch = 20, cex = 0.05, col = "red", xlim = c(0, max(Data.real[, 2] + 1)), 
  #        ylim = c(0, max(Data.real[, 4] + 0.05)), xaxs = "i", yaxs = "i", 
  #        axes = F, ann = F)
  
  plot(pointsCC, pch = 20, cex = 0.05, col = "Black", xlim = c(0, max(Data.real[, 2] + 1)), 
       ylim = c(0, max(Data.real[, 4] + 0.05)), xaxs = "i", yaxs = "i", 
       axes = F, ann = F)
}
fig.2dhist  <- function() {
  par(mar = c(4, 4, 1, 1))
  image( res2, col = r, axes = F, xlab = NA, ylab = NA, xaxs = "i", yaxs =  "i", useRaster = TRUE)
  axis( 1, at = seq( 0, 1, 0.111111111111 ), 
        labels = c( 0, NA, 1000, NA, 2000, NA, 3000, NA, 4000, NA))
  axis( 2, at = seq( 0, 1, 0.11111111111 ), 
        labels = c( 0, NA, 1000, NA, 2000, NA, 3000, NA, 4000, NA))
  title(xlab = "Upper Intercept (Ma)", ylab = "Lower Intercept (Ma)", cex = 2 )
  grid( nx = 45, ny = 45, lty = 3, lwd = 0.5, col = "Grey")
  grid( nx = 9, ny = 9, lty = 2, lwd = 0.5, col = "Black")
}

fig.xyplot  <- function() {
  par(mar = c(5, 4, 1, 2))
  plot(lowerdisc[1:nrow(lowerdisc), "Likelihood"] ~ lowerdisc[1:nrow(lowerdisc), "Lower Intercept"],
       type ="n", axes = F, ann = F, xlim = c(0, Tmax), ylim = c(0, max(lowerdisc$Likelihood) * 1.2),
       xaxs = "i", yaxs =  "i")
  axis(1, at = seq(0, Tmax, 5e+08), labels = seq(0, 4500, 500), 
       tcl = -0.3)
  axis(2, at = seq(0, max(lowerdisc$Likelihood),
                   ceiling(max(lowerdisc$Likelihood)/10 )),
       tcl = -0.3, las = 2)
  #   axis(2, at = seq(0, max(lowerdisc$Likelihood) * 1.2,
  #                    round(max(lowerdisc$Likelihood) * 1.2 /10 , 2)),
  #        tcl = -0.3, las = 2)
  #   points(upperdisc[1:nrow(upperdisc), "Likelihood"] ~ upperdisc[1:nrow(upperdisc), "Upper Intercept"],
  #          pch = 20, cex = .5, col = "blue")
  #   points(lowerdisc[1:nrow(lowerdisc), "Likelihood"] ~ lowerdisc[1:nrow(lowerdisc), "Lower Intercept"],
  #          pch = 20, cex = .5, col = "orange")
  
  points(upperdisc[1:nrow(upperdisc), "Likelihood"] ~ upperdisc[1:nrow(upperdisc), "Upper Intercept"],
         pch = 20, cex = .5, col = "Black")
  points(lowerdisc[1:nrow(lowerdisc), "Likelihood"] ~ lowerdisc[1:nrow(lowerdisc), "Lower Intercept"],
         pch = 20, cex = .5, col = "Grey")
  
  grid( nx = 45, ny = NA, lty = 3, lwd = 0.5, col = "Grey")
  grid( nx = 9, ny = NA, lty = 2, lwd = 0.5, col = "Black")
  title(xlab = "Age (Ma)", ylab = "Likelihood")
  
  #   legend("topright", pch = c(19, 19),
  #          col = c("orange", "blue"),
  #          legend = c("Lower Intercept", "Upper Intercept"))
  
  legend("topright", pch = c(19, 19),
         col = c("Grey", "Black"),
         legend = c("Lower Intercept", "Upper Intercept"))
}

fig.title   <- function() {
  par(mar = c(0, 0, 0, 0))
  plot(c(0, 1), c(0, 1), ann = F, bty = 'n', type = 'n', xaxt = 'n', yaxt = 'n')
  text(x = 0.5, y = 0.5, paste(Title), cex = 1.5)
}

fig.xyplotnorm  <- function() {
  par(mar = c(4, 4, 1, 1))
  plot(lowerdisc[1:nrow(lowerdisc), "Likelihood"] ~ lowerdisc[1:nrow(lowerdisc), "Lower Intercept"],
       type ="n", axes = F, ann = F, xlim = c(0, Tmax), ylim = c(0, 75), xaxs = "i", yaxs =  "i")
  axis(1, at = seq(0, Tmax, 5e+08), labels = seq(0, 4500, 500), 
       tcl = -0.3)
  axis(2, at = seq(0, maximus, round(maximus/5)),
       tcl = -0.3, las = 2)
  #   points(upperdisc[1:nrow(upperdisc), "Likelihood"] ~ upperdisc[1:nrow(upperdisc), "Upper Intercept"],
  #          pch = 20, cex = .5, col = "blue")
  #   points(lowerdisc[1:nrow(lowerdisc), "Likelihood"] ~ lowerdisc[1:nrow(lowerdisc), "Lower Intercept"],
  #          pch = 20, cex = .5, col = "orange")
  points(upperdisc[1:nrow(upperdisc), "Likelihood"] ~ upperdisc[1:nrow(upperdisc), "Upper Intercept"],
         pch = 20, cex = .5, col = "Black")
  points(lowerdisc[1:nrow(lowerdisc), "Likelihood"] ~ lowerdisc[1:nrow(lowerdisc), "Lower Intercept"],
         pch = 20, cex = .5, col = "Grey")
  grid( nx = 45, ny = NA, lty = 3, lwd = 0.5, col = "Grey")
  grid( nx = 9, ny = NA, lty = 2, lwd = 0.5, col = "Black")
  title(main = DataTitle, xlab = "Age (Ma)", ylab = "Likelihood")
  #   legend("topright", pch = c(19, 19),
  #          col = c("orange", "blue"),
  #          legend = c("Lower Intercept", "Upper Intercept"))
  legend("topright", pch = c(19, 19),
         col = c("Grey", "Black"),
         legend = c("Lower Intercept", "Upper Intercept"))
}
pdf(file = c(paste(DataTitle, "compare.pdf")))
layout(matrix(c(0, 0, 0, 1, 1, 2, 0, 0, 0), 3, 3, byrow = TRUE))
fig.xyplotnorm()
fig.2dhist()
dev.off()

pdf(file = c(paste(DataTitle, "plate.pdf")))
layout(matrix(c(4, 4, 4, 2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1, 1, 
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), 6, 6, byrow = TRUE))
fig.xyplot()
fig.2dhist()
fig.conc()
fig.title()
dev.off()

pdf(file = c(paste(DataTitle, "XYInterceptsnorm.pdf")))
fig.xyplotnorm()
dev.off()

pdf(file = c(paste(DataTitle, "2DHistogram.pdf")))
fig.2dhist()
dev.off()

pdf(file = c(paste(DataTitle, "XYIntercepts.pdf")))
fig.xyplot()
dev.off()

pdf(file = c(paste(DataTitle, "Concordia.pdf")), useDingbats = TRUE)
fig.conc()
dev.off()

write.table(upperdisc, file = paste(DataTitle, "upper intercept.csv"), row.names = FALSE,
            quote = FALSE, sep = ",")

write.table(lowerdisc, file = paste(DataTitle, "lower intercept.csv"), row.names = FALSE,
            quote = FALSE, sep = ",")

write.table(Resultdisc, file = paste(DataTitle, "Results.csv"), row.names = FALSE,
            quote = FALSE, sep = ",")

# Stop the clock
runtime = proc.time() - ptm
runfile  <- matrix( c( "Data set Title", DataTitle, "Data points", datapoints,
                       "Node Spacing (myr)", Tstep, "Number of gridlines", Discline,
                       "206/238 Bandwidth", sigma68mean,
                       "207/235 Bandwidth", sigma75mean,
                       "Run time (sec)", runtime[3]), 7, 2, byrow = TRUE)
colnames(runfile)    <- c( "Information", "This run")

write.table(runfile, file = paste(DataTitle, ".csv"), row.names = FALSE,
            col.names = FALSE, quote = FALSE, sep = ",")
rm(ages, concordia, pointsA, pointsAA, pointsB, pointsBB, pointsC, pointsCC, ratio68, ratio75,
   ratios, res1, res2, xtable, ytable)

### Uncertainty analysis

# Gaussian fit function
fitG =
  function(x,y,mu,sig,scale){
    
    f = function(p){
      d = p[3] * dnorm( x, mean = p[ 1 ], sd = p[ 2 ] )
      sum( ( d - y ) ^ 2)
    }
    optim( c( mu, sig, scale ), f )
  }

# iterative function that starts with 100 data points on either side of the peak 
# and moves inward until either: Sum of squared deviates < 0.5 and points > 11 or
# sum of squared deviates < 1 and points used is between 3 and 11 or 
fitG2   <- function (x, y) {
  i <- 1
  repeat {
    i  <- i + 1
    y1 <- y - 100 + i
    y2 <- y + 100 - i
    if (y1 < 1) {
      y1 <- 1
    }
    newdata <- x[y1 : y2, ]
    newfit  <- fitG( newdata$Intercept, newdata$Likelihood, 1.0e9, 200e7, 1)
    pred.likelihood  <- data.frame( Intercept = newdata$Intercept,
                                    Predicted = newfit$par[3]* dnorm(newdata$Intercept, 
                                                                     newfit$par[1], newfit$par[2]))
    deviates    <- pred.likelihood$Predicted - newdata$Likelihood
    deviates.2  <- deviates ^ 2
    sum.dev.2   <- sum(deviates.2) / working[y, 2]
    n           <- nrow(newdata)
    if (sum.dev.2 < 0.05 & n > 11 |
        sum.dev.2 < 1 & n > 3 & n < 11 |
        i > 98)
      break
    # print(n)
  }
  list(mean = newfit$par[1], sigma = newfit$par[2], 
       points =  n, deviation = sum.dev.2)
}


# Fit the Gaussian curves to the peaks identified in the 'lowerpeak' file
aa           <- acast(Resultdisc, `Upper Intercept`~`Lower Intercept`, value.var = "normalized", 
                      fun.aggregate = mean)

# filter by peak height, only using data that are > 1/3 the max value
aaa          <- ifelse( aa < max(aa / 3, na.rm = TRUE), NA, aa)
colnames(aa) <- c(seq(1, ncol(aa), 1))
rownames(aa) <- c(seq(1, nrow(aa), 1))

## Convert it to a raster object
r         <- raster(aaa)
extent(r) <- extent(c( 0, ncol(aaa), 0, ncol(aaa) ) + 0.5)

## Find the maximum value within the 9-cell neighborhood of each cell and put it in the 
# group of 9 cells
f          <- function( X ) {max (X, na.rm = FALSE)}
localmax   <- focal( r, w = matrix( 1, 3, 3 ), fun = f, pad = TRUE, padValue = NA)

## Does each cell have the maximum value in its neighborhood?
r2         <- r == localmax

## Get x-y coordinates of those cells that are local maxima
maxXY            <- data.frame(xyFromCell( r2, Which(r2==1, cells=TRUE)))
colnames(maxXY)  <- c("A", "B")
heights          <- rep(0, nrow(maxXY))
for (m in 1:nrow(maxXY))  {
  heights[m]  <- aa[nrow(aa) - maxXY$B[m], maxXY$A[m]]
}
maxXY             <- data.table(cbind(maxXY, heights))
maxXY             <- setorder(maxXY, -`heights`)

# try to fit the gaussian fit function to the data array 
# x is data array, y is the lower intercept location, 
# z is the upper intercept location
fitG.lower   <- function (x, y, z) {
  i <- 1
  repeat {
    i                 <- i + 1
    y1                <- y - 100 + i
    y2                <- y + 100 - i
    if (y1 < 1) {
      y1 <- 1
    }
    newdata           <- data.frame( x[ z, y1:y2 ] )
    rows              <- c( seq( y1 * Tstep, y2 * Tstep, Tstep ))
    newdata           <- data.frame( cbind( rows, newdata ))
    n                 <- nrow( newdata )
    colnames(newdata) <- c( "Intercept", "Likelihood" )
    newfit            <- fitG( newdata$Intercept, newdata$Likelihood, 1.0e9, 200e7, 1)
    pred.likelihood   <- data.frame( Intercept = newdata$Intercept,
                                     Predicted = newfit$par[3]* dnorm(newdata$Intercept, 
                                                                      newfit$par[1], newfit$par[2]))
    deviates          <- pred.likelihood$Predicted - newdata$Likelihood
    deviates.2        <- deviates ^ 2
    sum.dev.2         <- sum(deviates.2) / x[z, y]
    if (sum.dev.2 < 0.05 & n > 11 |
        sum.dev.2 < 1 & n > 3 & n < 11 |
        i > 98)
      break
  }
  list(mean = newfit$par[1], width = round(n * Tstep / 1e6), 
       points =  n, deviation = round(sum.dev.2, 6))
}

uncert  <- data.frame(       Upeak = rep(0, nrow(maxXY)), 
                             Uwidth = rep(0, nrow(maxXY)),
                             Upoints = rep(0, nrow(maxXY)),
                             Udeviation = rep(0, nrow(maxXY)),
                             Lpeak = rep(0, nrow(maxXY)), 
                             Lwidth = rep(0, nrow(maxXY)),
                             Upoints = rep(0, nrow(maxXY)),
                             Udeviation = rep(0, nrow(maxXY)),
                             `normalized likelihood` = rep(0, nrow(maxXY)))

for (i in 1:nrow(maxXY)) {
  uncert[i, 5:8] <- fitG.lower( aa, maxXY$A[i], nrow(aa) - maxXY$B[i])
  uncert$normalized.likelihood[i]  <- aa[nrow(aa) - maxXY$B[i], maxXY$A[i]]
}
uncert$Lpeak             <- round(uncert$Lpeak / 1e6)



# try to fit the gaussian fit function to the data array 
# x is data array, y is the lower intercept location, 
# z is the upper intercept location
fitG.upper   <- function (x, y, z) {
  i <- 1
  repeat {
    i                 <- i + 1
    z1                <- z - 100 + i
    z2                <- z + 100 - i
    if (z2 > nrow(aa)) {
      z2 <- nrow(aa)
    }
    newdata           <- data.frame( x[ z1:z2, y] )
    rows              <- c( seq( z1 * Tstep, z2 * Tstep, Tstep ))
    newdata           <- data.frame( cbind( rows, newdata ))
    n                 <- nrow( newdata )
    colnames(newdata) <- c( "Intercept", "Likelihood" )
    newfit            <- fitG( newdata$Intercept, newdata$Likelihood, 3.0e9, 200e7, 1)
    pred.likelihood   <- data.frame( Intercept = newdata$Intercept,
                                     Predicted = newfit$par[3]* dnorm(newdata$Intercept, 
                                                                      newfit$par[1], newfit$par[2]))
    deviates          <- pred.likelihood$Predicted - newdata$Likelihood
    deviates.2        <- deviates ^ 2
    sum.dev.2         <- sum(deviates.2) / x[z, y]
    if (sum.dev.2 < 0.05 & n > 11 |
        sum.dev.2 < 1 & n > 3 & n < 11 |
        i > 98)
      break
  }
  list(mean = newfit$par[1], width = round(n * Tstep / 1e6),
       points =  n, deviation = round(sum.dev.2, 6))
}


for (i in 1:nrow(maxXY)) {
  uncert[i, 1:4] <- fitG.upper( aa, maxXY$A[i], nrow(aa) - maxXY$B[i])
  uncert$normalized.likelihood[i]  <- round(aa[nrow(aa) - maxXY$B[i], maxXY$A[i]], 2)
}
uncert$Upeak       <- round(uncert$Upeak / 1e6)
colnames(uncert)   <- c("Upper Age (Ma)", "Upper Peak Width (Ma)", "Upper Number of Points Used",
                        "Upper Sum of Squared Deviates","Lower Age (Ma)", "Lower Peak Width (Ma)", 
                        "Lower Number of Points Used", "Lower Sum of Squared Deviates", 
                        "Normalized Likelihood")

# Sort by likelihood
uncert   <- setorder(data.table(uncert), -`Normalized Likelihood`)

write.table(uncert, file = paste(DataTitle, "Peak Location and Width.csv"), 
            row.names = FALSE, quote = FALSE, sep = ",")


# 3) At the very end, add a friendly message:
message("✅ run_discordance.R finished.  All outputs should be in ", getwd())