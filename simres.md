## Results of marginal density / histogram estimation in reproducing the illustrative example of the CCMC paper

### 1. On the wiggle issue in estimating the marginal density of $x_2 \sim \frac{1}{3}N(-5,8^2) +\frac{2}{3}N(25, 0.5^2)$ 

For the CCMC alogrithm, the gain factor $\delta_t$ is set as eq (2.4), the bandwidth $h_t$ is set as eq (3.3), 
the parameters $\{\rho, \kappa, M,\gamma\}$, the random walk MH, and the lattice size are set as those in the illustrative example. 
I tried running 6e5 iterations, but it seems that the wiggle issue still remains as shown in the Figure below.


![ccmc6e5g245](https://github.com/roxiesun/ccmc/blob/main/images/ccmc6e5g245new.gif)

Similar results were obtained if the lattice size is increased to 489 (Figure below). 
I'm not sure if this wiggling problem will always occur as the iteration goes up and the bandwidth gets increasingly small.

![ccmc6e5g489](https://github.com/roxiesun/ccmc/blob/main/images/ccmc6e5g489new.gif)


Another notable problem is that, although the histogram of the $x_2$ samples seems close to uniform, samples near the boundaries were drawn with slightly higher frequency even if the desired sampling distribution $\mathbf{\pi}$ was set as uniform. 


### 2. CMC for estimating the histogram of the marginal density of $x_2$
I also tried the CMC algorithm to get a histogram estimate for the marginal density of $x_2$. Figure (a) below shows the true histogram of this marginal density together with the CMC estimates across 1e6 iterations while Figure (b) is the CMC estimates only. The number of subregions is set to 244 according to the lattice used by ccmc, and parameters like $\delta_t, M, \rho,$ and $\kappa$ are all set the same. I add an additional assumption $\sum_i\widehat{g_i}^{(str)} = 1$ to control the size of $\widehat{g_i}^{(itr)}$.

It seems that  $\widehat{g_i}^{(itr)} \propto \int_{E_i}\psi(\mathbf{x})d\mathbf{x}$ holds when $itr$ gets large, and the histogram of $x_2$ samples are closed to uniform as shown in (c) below.


![cmc1e6m244](https://github.com/roxiesun/ccmc/blob/main/images/cmc1e6m244.gif)



If the additional contraint is set as $\sum_i\widehat{g_i}^{(itr)} = 4$, then the true and estimated histogram in (a) gets much closer:


![cmc1e6m244c4](https://github.com/roxiesun/ccmc/blob/main/images/cmc1e6m244c4.gif)



### 3.SAMC for estimating the histogram of the marginal density of $x_2$
I'm not sure if I'm doing this right, but it seems to me that the SAMC and CMC algorithm differs only in the number of samples $\mathbf{x}_k^{(t)}$ drawn in the sampling step (i.e., $M = 10$ or $1$) in estimating histogram of the marginal distribution. The figure below gives the true vs. estimated histogram, and the histogram of the $x_2$ samples.

The estimated $\widehat{g_i}^{(t)}$ seems less proportional to the true histogram than those obtained from CMC as shown in (a), but histogram of the $x_2$ samples is close to uniform as shown in (c). 


![samc1e6m244](https://github.com/roxiesun/ccmc/blob/main/images/samc1e6m244.gif)
