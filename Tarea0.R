s <- seq(2, 30, 3)
x <- 14
z <- c(2, 4, 10, 6, 5, 16, 8, 13)
w <- c(3, 6, 13, 7, 9, 11, 21, 8, 9)

if (x > 5) { print("Es mayor que 5") } else { print("Es menor que 5") }

for (y in 1:9) { print(2**y) }

while (x < 10) { print("Valor menor que 10") }

boxplot(z, w)
hist(w)

rnorm(5, mean = 2, sd = 0.5)
runif(10) < 0.5
