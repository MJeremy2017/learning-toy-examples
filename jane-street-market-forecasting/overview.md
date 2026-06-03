# Description
When approaching modeling problems in modern financial markets, there are many reasons to believe that the problems you are trying to solve are impossible. Even if you put aside the beliefs that the prices of financial instruments rationally reflect all available information, you’ll have to grapple with time series and distributions that have properties you don’t encounter in other sorts of modeling problems. Distributions can be famously fat-tailed, time series can be non-stationary, and data can generally fail to satisfy a lot of the underlying assumptions on which very successful statistical approaches rely. Layer on all of this the fact that the financial markets are ultimately a human endeavor involving a large number of individuals and institutions that are constantly changing with advances in technology and shifts in society, and responding to economic and geopolitical issues as they arise - and you can start to get a sense of the difficulties involved!

In this challenge, we ask you to build a model using real-world data derived from some of our production systems. This data gives a very close picture of some of the things we have to do every day to be successful at trading in modern financial markets. We’ve assembled a collection of features and responders related to markets where we run automated trading strategies and are concerned about having good underlying models. To balance crafting a challenging, relevant problem that ties into our business while respecting the proprietary and highly competitive nature of our trading, you will notice that we have anonymized and lightly obfuscated some of the features and responders we present in the data. These modifications don’t change the essence of the problem at hand, but they do allow us to give you a difficult task that meaningfully illustrates the work we do at Jane Street.

Jane Street has spent decades relentlessly innovating on all aspects of our trading, and building machine learning models to aid our decision-making. These models help us actively trade thousands of financial products each day across 200+ trading venues around the world. While this challenge only presents a tiny fraction of the quantitative problems Jane Streeters work on daily, we are very interested in seeing how the Kaggle community will approach this challenge, and in engaging with you about your solutions to the problem!

# Evaluation
Submissions are evaluated on a scoring function defined as the sample weighted zero-mean R-squared score ($R^2$) of `responder_6`. The formula is given by:

$$
\mathrm{R}^2 = 1 - \frac{\sum_i w_i (y_i - \hat{y}_i)^2}{\sum_i w_i y_i^2}
$$


where $y$ and $\hat{y}$ are the ground-truth and predicted value vectors of `responder_6`, respectively; $w_i$ is the sample weight vector.

