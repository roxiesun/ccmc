# Results of marginal density / histogram estimation in reproducing the illustrative example of the CCMC paper

## 1. On the wiggle issue in estimating the marginal density of $x_2 \sim \frac{1}{3}N(-5,8^2) +\frac{2}{3}N(25, 0.5^2)$ 

For the CCMC alogrithm, the gain factor $\delta_t$ is set as eq (2.4), the bandwidth $h_t$ is set as eq (3.3), 
the parameters $\{\rho, \kappa, M,\gamma\}$, the random walk MH, and the lattice size are set as those in the illustrative example. 
I tried running 6e5 iterations, but it seems that the wiggle issue still remains as shown in the Figure below.


![ccmc6e5g245](https://github.com/roxiesun/ccmc/blob/main/images/cmc1e6m244.gif)

Similar results were obtained if the lattice size is increased to 489 (Figure below). 
I'm not sure if this wiggling problem will always occur as the iteration goes up and the bandwidth gets increasingly small.
