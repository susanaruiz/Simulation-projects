# Busqueda Local
# la funcion se modifica
g <- function(x, y) {
  f<- (((x + 0.5)^4 - 25 * x^2 - 20 * x - 5 * x^2 + (y + 0.5)^4 - 30 * y^2 - 20 * y + 5 * y)/100)
  return(f)
}

low <- -3
high <- 3
step <- 0.01
replicas <- 100
replica <- function(t){
  curr <- c(runif(1, low, high), runif(1, low, high))
  best <- curr
  for (tiempo in 1:t) {
    delta <- runif(1, 0, step)
    #delta.x <- runif(1, 0, step)
    #delta.y <- runif(1, 0, step)
    xl <- curr + c(-delta,0)
    xr <- curr + c(delta,0)
    yl <- curr + c(0,-delta)
    yr <- curr + c(0,delta)
    coordenadas <- c(xl,xr,yl,yr)
    for(k in 1:8){
      if(coordenadas[k] < (-3)){
        coordenadas[k] <- coordenadas[k]+6 
      }
      if(coordenadas[k] > 3){
        coordenadas[k] <- coordenadas[k]-6
      }
    }
    mejor1 <- c()
    mejor2 <- c()
    for(p in 1:8){
      if(p %% 2 == 0){
        mejor2 <- c(mejor2,coordenadas[p])
      }else{
        mejor1 <- c(mejor1,coordenadas[p])
      }
    }
    val <- c()
    for(q in 1:4){
      val <- c(val, g(mejor1[q], mejor2[q]) )
    }
    maximo <- which.max(val)
    curr <- c(mejor1[maximo], mejor2[maximo])
    if(g(curr[1],curr[2]) > g(best[1],best[2])){ #Maximizar
      best <- curr
    }
  }
  return(best)
}

suppressMessages(library(doParallel))
registerDoParallel(makeCluster(detectCores() - 1))


for (pot in 2:4) {
  tmax <- 10^pot
  resultados <- foreach(i = 1:replicas, .combine=c) %dopar% replica(tmax)
  
  mejor1 <- c()
  mejor2 <- c()
  repl <- (2*replicas)
  for(p in 1:repl){
    if(p %% 2 == 0){
      mejor2 <- c(mejor2,resultados[p])
    }else{
      mejor1 <- c(mejor1,resultados[p])
    }
  }
  
  valores <- c()
  for(q in 1:replicas){
    valores <- c(valores, g(mejor1[q], mejor2[q]))
  }
  mejor <- which.max(valores)
  #Figuras
  x <- seq(-6, 5, 0.25)
  y <-  x
  z <- outer(x, y, g)
  dimnames(z) <- list(x, y)
  library(reshape2) 
  d <- melt(z)
  names(d) <- c("x", "y", "z")
  library(lattice) 
  png(paste0("t7", tmax, ".png", sep=""), width=500, height=500)
  plot(levelplot(z ~ x * y, data = d, col.regions=colorRampPalette(c("blue", "yellow","red", "black")), xlab="x", ylab="y", main="Representacion de la busqueda local"))
  trellis.focus("panel", 1, 1, highlight=FALSE)
  lpoints(mejor1, mejor2, pch=1, col="blue", cex=1)
  trellis.unfocus()
  trellis.focus("panel"[1], 1, 1, highlight=FALSE)
  lpoints(mejor1[mejor], mejor2[mejor], pch=19, col="chartreuse", cex=1)
  trellis.unfocus()
  
  graphics.off() 
}
stopImplicitCluster()