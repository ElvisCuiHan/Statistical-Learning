library(boot)
set.seed(1234)

live_days <- c(4,2,4,7,1,5,3,2,2,4,5,2,5,3,1,4,3,1,1,3)

CI <- t.test(live_days,alternative = "two.sided")$conf.int

mean_value <- function(data,indices){
  d <- data[indices]
  return(mean(d))}
results <- boot(data = live_days, statistic = mean_value, R=1000)
boot.ci(results, conf = 0.95,type = "perc")

options(digits=3)
log_live_days <- log(live_days)

log_CI <- t.test(log_live_days)$conf.int
