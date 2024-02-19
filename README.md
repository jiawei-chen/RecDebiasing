# Recommendation Debiasing

This website collects recent works and datasets on recommendation debiasing and their codes. We hope this website could help you do search on this topic.


### Contents
* [1. Survey Papers](#1-survey-papers)
* [2. Datasets](#2-datasets)
* [3. Debiasing Strategies](#3-debiasing-strategies) 
	* [3.1 Universal](#31-universal)
    * [3.2 Selection Bias](#32-selection-bias)
    * [3.3 Conformity Bias](#33-conformity-bias)
    * [3.4 Exposure Bias](#34-exposure-bias)
    * [3.5 Position Bias](#35-position-bias)
    * [3.6 Popularity Bias](#36-popularity-bias)
    * [3.7 Unfairness](#37-unfairness)
    * [3.8 Loop Effect](#38-loop-effect)
    * [3.9 Other Bias](#39-other-bias)

* [Tips](#tips)


## 1. Survey Papers
1. **A Survey on the Fairness of Recommender Systems**. TOIS 2023. [[pdf](https://dl.acm.org/doi/pdf/10.1145/3547333)]
1. **Bias and Debias in Recommender System: A Survey and Future Directions**. TOIS 2023. [[pdf](https://arxiv.org/pdf/2010.03240.pdf)]
2. **Bias Issues and Solutions in Recommender System**. WWW 2021,Recsys 2021. [[pdf](http://staff.ustc.edu.cn/~hexn/papers/recsys21-tutorial-bias.pdf)]

### More works about ***Introduction of bias*** can be found [here](https://github.com/jiawei-chen/RecDebiasing/blob/main/Introduction%20of%20bias.md)


## 2. Datasets
We collect some datasets which include unbiased data and are often used in the research of recommendation debiasing.
1. **Yahoo!R3: Collaborative Prediction and Ranking with Non-Random Missing Data**. Recsys 2009. [[pdf](https://www.cs.toronto.edu/~zemel/documents/acmrec2009-MarlinZemel.pdf)][[data](https://webscope.sandbox.yahoo.com/catalog.php?datatype=r)]
2. **Coat: Recommendations as Treatments: Debiasing Learning and Evaluation**. ICML 2016. [[pdf](https://arxiv.org/abs/1602.05352)][[data](https://www.cs.cornell.edu/~schnabts/mnar/)]
3. **KuaiRec: A Fully-observed Dataset for Recommender Systems**. CIKM 2022. [[pdf](https://arxiv.org/abs/2202.10842)][[data](https://chongminggao.github.io/KuaiRec/)]
4. **KuaiRand: An Unbiased Sequential Recommendation Dataset
with Randomly Exposed Videos**. CIKM 2022.[[pdf](https://arxiv.org/pdf/2208.08696.pdf)][[data](https://kuairand.com/)]


## 3. Debiasing Strategies

### 3.1 Universal 
1. **Balancing Unobserved Confounding with a Few Unbiased Ratings in Debiased Recommendations**. WWW 2023.[[pdf](https://dl.acm.org/doi/10.1145/3543507.3583495)] 

1. **Transfer Learning in Collaborative Recommendation for Bias Reduction**. Recsys 2021.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3460231.3478860)] [[code](http://csse.szu.edu.cn/staff/panwk/publications/TJR/)]
1. **AutoDebias: Learning to Debias for Recommendation**. SIGIR 2021.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3404835.3462919)] [[code](https://github.com/DongHande/AutoDebias)]
1. **A General Knowledge Distillation Framework for Counterfactual Recommendation via Uniform Data**. SIGIR 2020.[[pdf](https://dgliu.github.io/files/SIGIR20_KDCRec.pdf)] [[code](https://github.com/dgliu/SIGIR20_KDCRec)]
1. **Causal Embeddings for Recommendation**. Recsys 2018.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3240323.3240360)] [[code](https://github.com/criteo-research/CausE)]



### 3.2 Selection Bias
1. **Reconsidering Learning Objectives in Unbiased Recommendation A Distribution Shift Perspective**. KDD 2023.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3580305.3599487)] 
1. **Propensity Matters Measuring and Enhancing Balancing for Recommendation**. ICML 2023.[[pdf](https://dl.acm.org/doi/10.5555/3618408.3619239)] 
1. **A Generalized Propensity Learning Framework for Unbiased Post-Click Conversion Rate Estimation**. CIKM 2023.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3583780.3614760)] [[code](https://github.com/yuqing-zhou/GPL)]
1. **CDR: Conservative Doubly Robust Learning for Debiased Recommendation**. CIKM 2023.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3583780.3614805)] [[code](https://github.com/crazydumpling/CDR_CIKM2023)]
1. **UKD: Debiasing Conversion Rate Estimation via Uncertainty-regularized Knowledge Distillation**. WWW 2022.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3485447.3512081)] 
1. **Practical Counterfactual Policy Learning for Top-𝐾 Recommendations**. KDD 2022.[[pdf](https://dl.acm.org/doi/abs/10.1145/3534678.3539295)] 
1. **Debiasing Neighbor Aggregation for Graph Neural Network in Recommender Systems**. CIKM 2022.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3511808.3557576)] 
1. **Representation Matters When Learning From Biased Feedback in Recommendation**. CIKM 2022.[[pdf](https://dl.acm.org/doi/10.1145/3511808.3557431)] 
1. **Hard Negatives or False Negatives: Correcting Pooling Bias in Training Neural Ranking Models**. CIKM 2022.[[pdf](https://dl.acm.org/doi/abs/10.1145/3511808.3557343)] 
1. **Be Causal: De-biasing Social Network Confounding in Recommendation**. TKDD 2022.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3533725)] 
1. **Debiased recommendation with neural stratification**. AI OPEN 2022.[[pdf](https://arxiv.org/abs/2208.07281)] 


1. **ESCM2: Entire Space Counterfactual Multi-Task Model for Post-Click Conversion Rate Estimation**. SIGIR 2022.[[pdf](https://arxiv.org/abs/2204.05125)] 
1. **Generalized Doubly Robust Learning Framework for Debiasing Post-Click Conversion Rate Prediction**. KDD 2022.[[pdf](https://dl.acm.org/doi/10.1145/3534678.3539270)] 
1. **Combating Selection Biases in Recommender Systems with a Few Unbiased Ratings**. WSDM 2021.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3437963.3441799)] 
1. **Doubly Robust Estimator for Ranking Metrics with Post‐Click Conversions**. RecSys 2020.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3383313.3412262)] [[code](https://github.com/usaito/dr-ranking-metric)]
1. **Asymmetric tri-training for debiasing missing-not-at-random explicit feedback**. SIGIR 2020.[[pdf]((https://arxiv.org/pdf/1910.01444.pdf))] 
1. **Recommendations as treatments: Debiasing learning and evaluation**. ICML 2016.[[pdf](http://proceedings.mlr.press/v48/schnabel16.pdf)] [[code](https://www.cs.cornell.edu/~schnabts/mnar/)]
1. **Doubly robust joint learning for recommendation on data missing not at random**. ICML 2019.[[pdf](http://proceedings.mlr.press/v97/wang19n/wang19n.pdf)] 
1. **The deconfounded recommender: A causal inference approach to recommendation**. arXiv 2018.[[pdf](https://arxiv.org/pdf/1808.06581.pdf)] 
1. **Social recommendation with missing not at random data**. ICDM 2018.[[pdf](https://ieeexplore.ieee.org/abstract/document/8594827)] 
1. **Recommendations as treatments: Debiasing learning and evaluation**. .[[pdf](http://proceedings.mlr.press/v48/schnabel16.pdf)] 
1. **Boosting Response Aware Model-Based Collaborative Filtering**. TKDE 2015.[[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7045598)] 
1. **Probabilistic matrix factorization with non-random missing data**. JMLR 2014.[[pdf](http://proceedings.mlr.press/v32/hernandez-lobatob14.pdf)] [[code](https://jmhl.org/)]
1. **Bayesian Binomial Mixture Model for Collaborative Prediction With Non-Random Missing Data**. RecSys 2014.[[pdf](https://dl.acm.org/doi/pdf/10.1145/2645710.2645754)] 
1. **Evaluation of recommendations: rating-prediction and ranking**. RecSys 2013.[[pdf](https://dl.acm.org/doi/pdf/10.1145/2507157.2507160)] 
1. **Training and testing of recommender systems on data missing not at random**. KDD 2010.[[pdf](https://dl.acm.org/doi/abs/10.1145/1835804.1835895)] 
1. **Collaborative prediction and ranking with non-random missing data**. RecSys 2009.[[pdf](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.154.7692&rep=rep1&type=pdf)] 
1. **Collaborative filtering and the missing at random assumption**. UAI 2007.[[pdf](https://arxiv.org/ftp/arxiv/papers/1206/1206.5267.pdf)] [[code](https://github.com/rfenzo/ProyectoRecomendadores)]


### 3.3 Conformity Bias

1. **Popularity Bias Is Not Always Evil: Disentangling Benign and Harmful Bias for Recommendation**. TKDE 2022.[[pdf](https://arxiv.org/pdf/2109.07946.pdf)] 
1. **Disentangling user interest and Conformity for recommendation with causal embedding**. WWW 2021.[[pdf](https://arxiv.org/pdf/2006.11011.pdf)] [[code](https://github.com/tsinghua-fib-lab/DICE)]
1. **Learning personalized preference of strong and weak ties for social recommendation**. WWW 2017.[[pdf](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?referer=https://scholar.google.com.hk/&httpsredir=1&article=4657&context=sis_research)] 
1. **Are you influenced by others when rating?: Improve rating prediction by conformity modeling**. RecSys 2016.[[pdf](https://dl.acm.org/doi/10.1145/2959100.2959141)] 
1. **Xgboost: A scalable tree boosting system**. KDD 2016.[[pdf](https://arxiv.org/pdf/1603.02754.pdf?__hstc=133736337.1bb630f9cde2cb5f07430159d50a3c91.1513641600097.1513641600098.1513641600099.1&__hssc=133736337.1.1513641600100&__hsfp=528229161)] [[code](https://github.com/dmlc/xgboost)]

1. **A probabilistic model for using social networks in personalized item recommendation**. RecSys 2015.[[pdf](http://www.cs.columbia.edu/~blei/papers/ChaneyBleiEliassi-Rad2015.pdf)] [[code](https://github.com/ajbc/spf)]
1. **Mtrust: discerning multi-faceted trust in a connected world**. WSDM 2012.[[pdf](http://www.public.asu.edu/~huanliu/papers/wsdm12.pdf)] 
1. **Learning to recommend with social trust ensemble**. SIGIR 2009.[[pdf](https://www.researchgate.net/profile/Michael-Lyu/publication/221299915_Learning_to_Recommend_with_Social_Trust_Ensemble/links/5461d7dd0cf27487b4530caa/Learning-to-Recommend-with-Social-Trust-Ensemble.pdf)] 



### 3.4 Exposure Bias
1. **uCTRL Unbiased Contrastive Representation Learning via Alignment and Uniformity for Collaborative Filtering**. SIGIR 2023.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3539618.3592076)] [[code](https://github.com/Jaewoong-Lee/sigir_2023_uCTRL)]
1. **Empowering Collaborative Filtering with Principled Adversarial Contrastive Loss**. NIPS 2023.[[pdf](https://papers.nips.cc/paper_files/paper/2023/file/13f1750b825659394a6499399e7637fc-Paper-Conference.pdf)] [[code](https://github.com/LehengTHU/AdvInfoNCE)]
1. **Debiasing Sequential Recommenders through Distributionally Robust Optimization over System Exposure**. WSDM 2023.[[pdf](https://arxiv.org/abs/2312.07036)] [[code](https://github.com/nancheng58/DebiasedSR_DRO)]
1. **Debiasing the Cloze Task in Sequential Recommendation with Bidirectional Transformers**. KDD 2022.[[pdf](https://dl.acm.org/doi/abs/10.1145/3534678.3539430)] 
1. **Debiasing Neighbor Aggregation for Graph Neural Network in Recommender Systems**. CIKM 2022.[[pdf](https://arxiv.org/abs/2208.08847)] 
1. **Non-Clicks Mean Irrelevant? Propensity Ratio Scoring As a Correction**. WSDM 2021.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3437963.3441798)] 
1. **Propensity-Independent Bias Recovery in Offline Learning-to-Rank Systems**. SIGIR 2021.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3404835.3463097)] 
1. **Clicks can be Cheating: Counterfactual Recommendation for Mitigating Clickbait Issue**. SIGIR 2021.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3404835.3462962)] [[code](https://github.com/WenjieWWJ/Clickbait/)]
1. **Mitigating Confounding Bias in Recommendation via Information Bottleneck**. Recsys 2021.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3460231.3474263)] [[code](https://github.com/dgliu/RecSys21_DIB)]

1. **Debiased Explainable Pairwise Ranking from Implicit Feedback**. Recsys 2021.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3460231.3474274)] [[code](https://github.com/KhalilDMK/EBPR)]
1. **Top-N Recommendation with Counterfactual User Preference Simulation**. CIKM 2021.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3459637.3482305)] 
1. **SamWalker++: recommendation with informative sampling strategy**. TKDE 2021.[[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9507306)] [[code](https://github.com/jiawei-chen/SamWalker)]
1. **Deconfounded Causal Collaborative Filtering**. Arxiv 2021/TORS 2023.[[pdf](https://dlnext.acm.org/doi/pdf/10.1145/3606035)] 
1. **Unbiased recommender learning from missing-not-at-random implicit feedback**. WSDM 2020.[[pdf](https://arxiv.org/pdf/1909.03601.pdf)] [[code](https://github.com/usaito/unbiased-implicit-rec)]
1. **Reinforced negative sampling over knowledge graph for recommendation**. WWW 2020.[[pdf](https://arxiv.org/pdf/2003.05753.pdf)] [[code](https://github.com/xiangwang1223/kgpolicy)]
1. **Fast adaptively weighted matrix factorization for recommendation with implicit feedback**. AAAI 2020.[[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/5751)] 
1. **Correcting for selection bias in learning-to-rank systems**. WWW 2020.[[pdf](https://arxiv.org/pdf/2001.11358.pdf)] [[code](https://github.com/edgeslab/heckman_rank)]
1. **Large-scale causal approaches to debiasing post-click conversion rate estimation with multi-task learning**. WWW 2020.[[pdf](https://arxiv.org/pdf/1910.09337.pdf)] 
1. **Entire space multi-task modeling via post-click behavior decomposition for conversion rate prediction**. SIGIR 2020.[[pdf](https://arxiv.org/pdf/1910.07099.pdf)] 

1. **Unbiased Implicit Recommendation and Propensity Estimation via Combinational Joint Learning**. Recsys 2020.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3383313.3412210)] [[code](https://github.com/Zziwei/Unbiased-Propensity-and-Recommendation)]
1. **Debiasing Item-to-Item Recommendations With Small Annotated Datasets**. Recsys 2020.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3383313.3412265)] 
1. **Reinforced negative sampling for recommendation with exposure data**. IJCAI 2019.[[pdf](https://www.ijcai.org/Proceedings/2019/0309.pdf)] [[code](https://github.com/dingjingtao/ReinforceNS)]
1. **Samwalker: Social recommendation with informative sampling strategy**. WWW 2019.[[pdf](https://jiawei-chen.github.io/paper/SamWalker.pdf)] [[code](https://github.com/jiawei-chen/Samwalker)]
1. **Collaborative filtering with social exposure: A modular approach to social recommendation**. AAAI 2018.[[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/11835)] [[code](https://github.com/99731/SERec)]
1. **An improved sampler for bayesian personalized ranking by leveraging view data**. WWW 2018.[[pdf](http://staff.ustc.edu.cn/~hexn/papers/www18-improvedBPR.pdf)] 
1. **Unbiased offline recommender evaluation for missing-not-at-random implicit feedback**. RecSys 2018.[[pdf](https://vision.cornell.edu/se3/wp-content/uploads/2018/08/recsys18_unbiased_eval.pdf)] [[code](https://github.com/ylongqi/unbiased-offline-recommender-evaluation)]
1. **Entire space multi-task model: An effective approach for estimating post-click conversion rate**. SIGIR 2018.[[pdf](https://arxiv.org/pdf/1804.07931.pdf)] 
1. **Modeling users’ exposure with social knowledge influence and consumption influence for recommendation**. CIKM 2018.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3269206.3271742)] 
1. **Selection of negative samples for one-class matrix factorization**. SDM 2017.[[pdf](https://www.csie.ntu.edu.tw/~cjlin/papers/one-class-mf/biased-mf-sdm-with-supp.pdf)] [[code](https://www.csie.ntu.edu.tw/~cjlin/papers/one-class-mf/)]
1. **Learning to rank with selection bias in personal search**. SIGIR 2016.[[pdf](https://research.google/pubs/pub45286.pdf)] 
1. **Modeling user exposure in recommendation**. WWW 2016.[[pdf](https://arxiv.org/pdf/1510.07025.pdf)] [[code](https://github.com/dawenl/expo-mf)]
1. **Collaborative denoising auto-encoders for top-n recommender systems (CDAE)**. WSDM 2016.[[pdf](https://www.datascienceassn.org/sites/default/files/Collaborative%20Denoising%20Auto-Encoders%20for%20Top-N%20Recommender%20Systems.pdf)] [[code](https://github.com/jasonyaw/CDAE)]
1. **Fast matrix factorization for online recommendation with implicit feedback**. SIGIR 2016.[[pdf](https://arxiv.org/pdf/1708.05024.pdf)] [[code](https://github.com/hexiangnan/sigir16-eals)]
1. **Dynamic matrix factorization with priors on unknown values**. KDD 2015.[[pdf](https://arxiv.org/pdf/1507.06452.pdf)] [[code](https://github.com/rdevooght/MF-with-prior-and-updates)]
1. **Logistic matrix factorization for implicit feedback data**. NIPS 2014.[[pdf](http://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf)] 
1. **Improving one-class collaborative filtering by incorporating rich user information**. CIKM 2010.[[pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.228.7135&rep=rep1&type=pdf)] 
1. **Mind the gaps: weighting the unknown in large-scale one-class collaborative filtering**. KDD 2009.[[pdf](https://dl.acm.org/doi/pdf/10.1145/1557019.1557094)] 
1. **Collaborative filtering for implicit feedback datasets**. ICDM 2008.[[pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.167.5120&rep=rep1&type=pdf)] [[code](https://github.com/benfred/implicit)]
1. **One-class collaborative filtering**. ICDM 2008.[[pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.306.4684&rep=rep1&type=pdf)] 


### 3.5 Position Bias
1. **An Offline Metric for the Debiasedness of Click Models**. SIGIR 2023.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3539618.3591639)] [[code](https://github.com/philipphager/sigir-cmip)]
1. **A Probabilistic Position Bias Model for Short-Video Recommendation Feeds**. RecSys 2023.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3604915.3608777)] [[code](https://github.com/olivierjeunen/C-3PO-recsys-2023)]

1. **Unbiased Learning to Rank with Biased Continuous Feedback**. CIKM 2022.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3511808.3557483)] [[code](https://github.com/phyllist/ULTRA)]
1. **Scalar is Not Enough: Vectorization-based Unbiased Learning to Rank**. KDD 2022.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3534678.3539468)] [[code](https://github.com/Keytoyze/Vectorization)]
1. **Doubly Robust Off-Policy Evaluation for Ranking Policies under the Cascade Behavior Model**. WSDM 2022.[[pdf](https://dl.acm.org/doi/abs/10.1145/3488560.3498380)] [[code](https://github.com/aiueola/wsdm2022-cascade-dr)]
1. **Can Clicks Be Both Labels and Features?: Unbiased Behavior Feature Collection and Uncertainty-aware Learning to Rank**. SIGIR 2022.[[pdf](https://dl.acm.org/doi/abs/10.1145/3477495.3531948)] [[code](https://github.com/Taosheng-ty/UCBRankSIGIR2022)]
1. **A Graph-Enhanced Click Model for Web Search**. SIGIR 2021.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3404835.3462895)] [[code](https://github.com/CHIANGEL/GraphCM)]
1. **Adapting Interactional Observation Embedding for Counterfactual Learning to Rank**. SIGIR 2021.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3404835.3462901)] [[code](https://github.com/Keytoyze/Interactional-Observation-Based-Model)]
1. **When Inverse Propensity Scoring does not Work: Affine Corrections for Unbiased Learning to Rank**. CIKM 2020.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3340531.3412031)] [[code](https://github.com/AliVard/trust-bias-CIKM2020/tree/master/trust_bias)]
1. **A deep recurrent survival model for unbiased ranking**. SIGIR 2020.[[pdf](https://arxiv.org/pdf/2004.14714.pdf)] [[code](https://github.com/Jinjiarui/DRSR)]
1. **Attribute-based propensity for unbiased learning in recommender systems: Algorithm and case studies**. KDD 2020.[[pdf](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/54a3b73ea1e85e94e5d5bb5a9df821a1f32aa783.pdf)] 
1. **Debiasing grid-based product search in e-commerce**. KDD 2020.[[pdf](https://www.hongliangjie.com/publications/kdd2020_2.pdf)] 
1. **Cascade model-based propensity estimation for counterfactual learning to rank**. SIGIR 2020.[[pdf](https://arxiv.org/pdf/2005.11938.pdf))] [[code](https://github.com/AliVard/CM-IPS-SIGIR20)]
1. **Addressing Trust Bias for Unbiased Learning-to-Rank**. WWW 2019.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3308558.3313697)] 
1. **Position bias estimation for unbiased learning to rank in personal search**. WSDM 2018.[[pdf](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/3bace79f9bcead0b20dec31e2a0878346ad2fb0d.pdf)] 
1. **Offline Evaluation of Ranking Policies with Click Models**. KDD 2018.[[pdf](https://dl.acm.org/doi/abs/10.1145/3219819.3220028)] 
1. **Unbiased learning to rank with unbiased propensity estimation**. SIGIR 2018.[[pdf](https://arxiv.org/pdf/1804.05938.pdf)] [[code](https://github.com/QingyaoAi/Dual-Learning-Algorithm-for-Unbiased-Learning-to-Rank)]
1. **Unbiased learning-to-rank with biased feedback**.  WSDM 2017.[[pdf](https://arxiv.org/pdf/1608.04468.pdf)] [[code](https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html)]
1. **Multileave gradient descent for fast online learning to rank**. WSDM 2016.[[pdf](https://ilps.science.uva.nl/wp-content/uploads/sites/8/2015/11/DIR2015-proceedings.pdf#page=14)] 
1. **Learning to rank with selection bias in personal search**. SIGIR 2016.[[pdf](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45286.pdf)] 
1. **Batch learning from logged bandit feedback through counterfactual risk minimization**. JMLR 2015.[[pdf](https://www.jmlr.org/papers/volume16/swaminathan15a/swaminathan15a.pdf)] 
1. **Learning socially optimal information systems from egoistic users**. ECML PKDD 2013.[[pdf](https://link.springer.com/content/pdf/10.1007/978-3-642-40991-2_9.pdf)] 
1. **Reusing historical interaction data for faster online learning to rank for ir**. WSDM 2013.[[pdf](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.408.196&rep=rep1&type=pdf)] [[code](https://ilps.science.uva.nl/resources/online-learning-framework)]
1. **A novel click model and its applications to online advertising**. WSDM 2010.[[pdf](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/WSDM2010.pdf)] 
1. **A dynamic bayesian network click model for web search ranking**. WWW 2009.[[pdf](http://www2009.eprints.org/1/1/p1.pdf)] 
1. **Click chain model in web search**. WWW 2009.[[pdf](http://www2009.eprints.org/2/1/p11.pdf)] 
1. **A user browsing model to predict search engine click data from past observations**. SIGIR 2008.[[pdf](https://www.researchgate.net/profile/Georges-Dupret/publication/200110492_A_user_browsing_model_to_predict_search_engine_click_data_from_past_observations/links/54e4c5ea0cf29865c3351048/A-user-browsing-model-to-predict-search-engine-click-data-from-past-observations.pdf)] 
1. **An experimental comparison of click position-bias models**. WSDM 2008.[[pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.112.1288&rep=rep1&type=pdf)] 
1. **Comparing click logs and editorial labels for training query rewriting**. WWW 2007.[[pdf](https://www2007.org/workshops/paper_63.pdf)] 



### 3.6 Popularity Bias
1. **TCCM Time and Content-Aware Causal Model for Unbiased News Recommendation**. CIKM 2023.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3583780.3615272)] [[code](https://github.com/XFastDataLab/TCCM)]
1. **Rlieving Popularity Bias in Interactive Recommendation A Diversity-Novelty-Aware Reinforcement Learning Approach**. TOIS 2023.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3618107)] [[code](https://github.com/shixiaoyu0216/DNaIR)]
1. **Test-Time Embedding Normalization for Popularity Bias Mitigation**. CIKM 2023.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3583780.3615281)] [[code](https://github.com/ml-postech/TTEN)]
1. **Potential Factors Leading to Popularity Unfairness in Recommender Systems A User-Centered Analysis**. Arxiv 2023.[[pdf](https://arxiv.org/pdf/2310.02961.pdf)] 
1. **Mitigating the Popularity Bias of Graph Collaborative Filtering A Dimensional Collapse Perspective**. NIPS 2023.[[pdf](https://papers.nips.cc/paper_files/paper/2023/file/d5753be6f71fbfefaf47aa27ec41279c-Paper-Conference.pdf)] [[code](https://github.com/yifeiacc/LogDet4Rec/)]
1. **A Model-Agnostic Popularity Debias Training Framework for Click-Through Rate Prediction in Recommender System**. SIGIR 2023.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3539618.3591939)] 
1. **Popularity Debiasing from Exposure to Interaction in Collaborative Filtering**. SIGIR 2023.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3539618.3591947)] [[code](https://github.com/UnitDan/IPL)]
1. **Adaptive Popularity Debiasing Aggregator for Graph Collaborative Filtering**. SIGIR 2023.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3539618.3591635)] 
1. **HDNR A Hyperbolic-Based Debiased Approach for Personalized News Recommendation**. SIGIR 2023.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3539618.3591693)] 
1. **MELT Mutual Enhancement of Long-Tailed User and Item for Sequential Recommendation**. SIGIR 2023.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3539618.3591725)] [[code](https://github.com/rlqja1107/MELT)]

1. **Stabilized Doubly Robust Learning for Recommendation on Data Missing Not at Random**. ICLR 2023.[[pdf](https://arxiv.org/abs/2205.04701)] 
1. **Invariant Collaborative Filtering to Popularity Distribution Shift**. WWW 2023.[[pdf](https://dl.acm.org/doi/10.1145/3543507.3583461)] [[code](https://github.com/anzhang314/InvCF)]
1. **Investigating Accuracy-Novelty Performance for Graph-based Collaborative Filtering**. SIGIR 2022.[[pdf](https://dl.acm.org/doi/10.1145/3477495.3532005)] [[code](https://github.com/fuxiAIlab/r-AdjNorm)]
1. **Evolution of Popularity Bias: Empirical Study and Debiasing**. KDD 2022.[[pdf](https://arxiv.org/abs/2207.03372)] [[code](https://github.com/Zziwei/Popularity-Bias-in-Dynamic-Recommendation)]
1. **Countering Popularity Bias by Regularizing Score Differences**. RecSys 2022.[[pdf](https://dl.acm.org/doi/10.1145/3523227.3546757)] [[code](https://github.com/stillpsy/popbias)]
1. **Co-training Disentangled Domain Adaptation Network for Leveraging Popularity Bias in Recommenders**. SIGIR 2022.[[pdf](https://dl.acm.org/doi/abs/10.1145/3477495.3531952)] 
1. **Popularity bias in ranking and recommendation**. AIES 2019.[[pdf](https://d1wqtxts1xzle7.cloudfront.net/59543380/AIES2019.pdf?1559792566=&response-content-disposition=inline%3B+filename%3DPopularity_Bias_in_Ranking_and_Recommend.pdf&Expires=1617615017&Signature=X8T9bB3TRTGzFfbDfyMQANbJrYQMreC3fKdt10LzBeM2SY8hpe0ovwZHtMxe2DkJAi0OVrq24e~xu3S3GHG434z7MPhgHH7e28Jy61PSVQsy-IgxM3XzeVoPZzNVaS6R8GCVKpX1Ho4XZjfzsRXaP-50wFtFtOczmLTdDmCfFzpi9ngCIAAmjQEXaTUIRUPxPnTF20JYZLuDymUfQiC7CuZNceg9FsfqFp1ON86aQVfmNiI6VBZIi1Sy0akDTjaTujYplbl4vfAuOTbx5JfhjPDV5fwLB~M~cxFFDYCtqM8PvJO-fVZqObrW3ftZCz2LoRJyy0ve1IGPYb3jAJ7oag__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)] 
1. **Disentangling User Interest and Conformity for Recommendation with Causal Embedding**. WWW 2021.[[pdf](https://arxiv.org/pdf/2006.11011.pdf)] [[code](https://github.com/tsinghua-fib-lab/DICE)]
1. **The Unfairness of Popularity Bias in Recommendation**. SAC 2021.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3412841.3442123)] [[code](https://github.com/rcaborges/popularity-bias-vae)]
1. **Popularity Bias in Dynamic Recommendation**. KDD 2021.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3447548.3467376)] [[code](https://github.com/Zziwei/Popularity-Bias-in-Dynamic-Recommendation)]
1. **Causal Intervention for Leveraging Popularity Bias in Recommendation**. SIGIR 2021.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3404835.3462875)] [[code](https://github.com/zyang1580/PDA)]
1. **Deconfounded Recommendation for Alleviating Bias Amplification**. KDD 2021.[[pdf](https://arxiv.org/pdf/2105.10648.pdf)] [[code](https://github.com/WenjieWWJ/DecRS)]
1. **Popularity Bias Is Not Always Evil: Disentangling Benign and Harmful Bias for Recommendation**. Arxiv 2021/TKDE 2022.[[pdf](https://arxiv.org/pdf/2109.07946.pdf)]
1. **Model-agnostic counterfactual reasoning for eliminating popularity bias in recommender system**. KDD 2021.[[pdf](https://arxiv.org/pdf/2010.15363.pdf)] [[code](https://github.com/weitianxin/MACR)]
1. **ESAM: discriminative domain adaptation with non-displayed items to improve long-tail performance**. SIGIR 2020.[[pdf](https://arxiv.org/pdf/2005.10545.pdf)] [[code](https://github.com/A-bone1/ESAM.git)]
1. **Unbiased offline recommender evaluation for missing-not-at-random implicit feedback**. RecSys 2018.[[pdf](https://vision.cornell.edu/se3/wp-content/uploads/2018/08/recsys18_unbiased_eval.pdf)] [[code](https://github.com/ylongqi/unbiased-offline-recommender-evaluation)]
1. **An adversarial approach to improve long-tail performance in neural collaborative filtering**. CIKM 2018.[[pdf](http://aditk2.web.engr.illinois.edu/reports/sp0781.pdf)] 
1. **Controlling popularity bias in learning-to-rank recommendation**. RecSys 2017.[[pdf](https://www.researchgate.net/profile/Himan-Abdollahpouri/publication/318351355_Controlling_Popularity_Bias_in_Learning-to-Rank_Recommendation/links/5a1375450f7e9b1e573086d6/Controlling-Popularity-Bias-in-Learning-to-Rank-Recommendation.pdf)] 
1. **Incorporating diversity in a learning to rank recommender system**. FLAIRS 2016.[[pdf](https://aaai.org/papers/572-flairs-2016-12944/)] 
1. **The limits of popularity-based recommendations, and the role of social ties**. KDD 2016.[[pdf](https://arxiv.org/pdf/1607.04263.pdf)] [[code](https://github.com/Steven--/recommender)]
1. **Correcting popularity bias by enhancing recommendation neutrality**. RecSys 2014.[[pdf](https://www.kamishima.net/archive/2014-po-recsys-print.pdf)] 
1. **Efficiency improvement of neutrality-enhanced recommendation**. RecSys 2013.[[pdf](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.6889&rep=rep1&type=pdf#page=5)] [[code](http://www.kamishima.net/inrs/)]


### 3.7 Unfairness

1. **Providing Previously Unseen Users Fair Recommendations Using Variational Autoencoders**. RecSys 2023.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3604915.3608842)] [[code](https://github.com/BjornarVass/fair-vae-rec)]
1. **Path-Specific Counterfactual Fairness for Recommender Systems**. KDD 2023.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3580305.3599462)] [[code](https://github.com/yaochenzhu/PSF-VAE)]
1. **Towards Robust Fairness-aware Recommendation**. Recsys 2023.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3604915.3608784)] 
1. **When Fairness meets Bias a Debiased Framework for Fairness aware Top-N Recommendation**. Recsys 2023.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3604915.3608770)] 
1. **Two-sided Calibration for Quality-aware Responsible Recommendation**. Recsys 2023.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3604915.3608799)] [[code](https://github.com/THUwangcy/ReChorus/tree/RecSys23)]
1. **Rectifying Unfairness in Recommendation Feedback Loop**. SIGIR 2023.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3539618.3591754)] 
1. **Measuring Item Global Residual Value for Fair Recommendation**. SIGIR 2023.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3539618.3591724)] [[code](https://github.com/Alice1998/TaFR)]

1. **Improving Recommendation Fairness via Data Augmentation**. WWW 2023.[[pdf](https://dl.acm.org/doi/10.1145/3543507.3583341)] [[code](https://github.com/newlei/FDA)]
1. **Controllable Universal Fair Representation Learning**. WWW 2023.[[pdf](https://dl.acm.org/doi/10.1145/3543507.3583307)] 
1. **Cascaded Debiasing: Studying the Cumulative Effect of Multiple Fairness-Enhancing Interventions**. CIKM 2022.[[pdf](https://dl.acm.org/doi/10.1145/3511808.3557155)] [[code](https://github.com/bhavyaghai/Cascaded-Debiasing)]

1. **Fighting Mainstream Bias in Recommender Systems via Local Fine Tuning**. WSDM 2022.[[pdf](https://dl.acm.org/doi/10.1145/3488560.3498427)] [[code](https://github.com/Zziwei/Measuring-Mitigating-Mainstream-Bias)]
1. **CPFair: Personalized Consumer and Producer Fairness Re-ranking for Recommender Systems**. SIGIR 2022.[[pdf](https://arxiv.org/abs/2204.08085)] [[code](https://github.com/rahmanidashti/CPFairRecSys)]
1. **Fairness of Exposure in Light of Incomplete Exposure Estimation**. SIGIR 2022.[[pdf](https://arxiv.org/abs/2205.12901)] [[code](https://github.com/MariaHeuss/2022-SIGIR-FOEIncomplete-Exposure)]
1. **Explainable Fairness in Recommendation**. SIGIR 2022.[[pdf](https://arxiv.org/pdf/2204.11159.pdf)] 
1. **Joint Multisided Exposure Fairness for Recommendation**. SIGIR 2022.[[pdf](https://arxiv.org/pdf/2205.00048.pdf)] [[code](https://github.com/haolun-wu/JMEFairness)]
1. **Pareto-Optimal Fairness-Utility Amortizations in Rankings with a DBN Exposure Model**. SIGIR 2022.[[pdf](https://arxiv.org/abs/2205.07647)] [[code](https://github.com/naver/expohedron)]
1. **Optimizing generalized Gini indices for fairness in rankings**. SIGIR 2022.[[pdf](https://arxiv.org/abs/2204.06521)] 
1. **Probabilistic Permutation Graph Search: Black-Box Optimization for Fairness in Ranking**. SIGIR 2022.[[pdf](https://dl.acm.org/doi/10.1145/3477495.3532045)] [[code](https://github.com/AliVard/PPG)]
1. **Measuring Fairness in Ranked Outputs**. SIGIR 2022.[[pdf](https://arxiv.org/abs/1610.08559)] 
1. **Comprehensive Fair Meta-learned Recommender System**. KDD 2022.[[pdf](https://arxiv.org/abs/2206.04789)] [[code](https://github.com/weitianxin/CLOVER)]
1. **Fair Ranking as Fair Division: Impact-Based Individual Fairness in Ranking**. KDD 2022.[[pdf](https://arxiv.org/abs/2206.07247)] [[code](https://github.com/usaito/kdd2022-fair-ranking-nsw)]
1. **Fair Representation Learning: An Alternative to Mutual Information**. KDD 2022.[[pdf](https://dl.acm.org/doi/abs/10.1145/3534678.3539302)] [[code](https://github.com/SoftWiser-group/FairDisCo)]
1. **Leave No User Behind: Towards Improving the Utility of Recommender Systems for Non-mainstream Users**. WSDM 2021.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3437963.3441769)] [[code](https://github.com/roger-zhe-li/wsdm21-mainstream)]
1. **User-oriented Fairness in Recommendation**. WWW2021.[[pdf](https://arxiv.org/pdf/2104.10671.pdf)] [[code](https://github.com/rutgerswiselab/user-fairness)]
1. **Policy-Gradient Training of Fair and Unbiased Ranking Functions**. SIGIR 2021.[[pdf](https://arxiv.org/pdf/1911.08054.pdf)] [[code](https://github.com/him229/fultr)]
1. **Towards Long-term Fairness in Recommendation**. WSDM 2021.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3437963.3441824)] [[code](https://github.com/TobyGE/FCPO)]
1. **Towards Personalized Fairness based on Causal Notion**. SIGIR 2021.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3404835.3462966)] 
1. **Computationally Efficient Optimization of Plackett-Luce Ranking Models for Relevance and Fairness**. SIGIR 2021.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3404835.3462830)] [[code](https://github.com/HarrieO/2021-SIGIR-plackett-luce)]
1. **Learning Fair Representations for Recommendation: A Graph-based Perspective**. WWW 2021.[[pdf](https://arxiv.org/pdf/2102.09140.pdf)] [[code](https://github.com/newlei/FairGo)]
1. **Debiasing Career Recommendations with Neural Fair Collaborative Filtering**. WWW 2021.[[pdf](https://mdsoar.org/bitstream/handle/11603/21218/Islam%20%282021%29%20-%20Debiasing%20Career%20Recommendations%20with%20Neural%20Fair%20Collaborative%20Filtering%20%28WWW%29.pdf?sequence=1&isAllowed=y)] [[code](https://github.com/rashid-islam/nfcf)]
1. **Debayes: a bayesian method for debiasing network embeddings**. ICML 2020.[[pdf](http://proceedings.mlr.press/v119/buyl20a/buyl20a.pdf)] [[code](https://github.com/1njiku/1njiku.github.io)]
1. **Measuring and Mitigating Item Under-Recommendation Bias in Personalized Ranking Systems**. SIGIR 2020.[[pdf](https://dl.acm.org/doi/pdf/10.1145/3397271.3401177)] [[code](https://github.com/Zziwei/Item-Underrecommendation-Bias)]
1. **Controlling fairness and bias in dynamic learning-to-rank**. SIGIR 2020.[[pdf](https://arxiv.org/pdf/2005.14713.pdf)] [[code](https://github.com/MarcoMorik/Dynamic-Fairness)]
1. **Designing fair ranking schemes**. SIGMOD 2019.[[pdf](https://arxiv.org/pdf/1712.09752.pdf)] 
1. **Fairwalk: Towards fair graph embedding**. IJCAI 2019.[[pdf](https://www.ijcai.org/Proceedings/2019/0456.pdf)] 
1. **Fairness in recommendation ranking through pairwise comparisons**. KDD 2019.[[pdf](https://arxiv.org/pdf/1903.00780.pdf)] 
1. **Compositional fairness constraints for graph embeddings**.  ICML 2019.[[pdf](http://proceedings.mlr.press/v97/bose19a/bose19a.pdf)] [[code](https://github.com/joeybose/Flexible-Fairness-Constraints)]
1. **Fairness-aware ranking in search & recommendation systems with application to linkedin talent search**. KDD 2019.[[pdf](https://arxiv.org/pdf/1905.01989.pdf)] 
1. **Counterfactual fairness: Unidentification  bound and algorithm**. IJCAI 2019.[[pdf](https://par.nsf.gov/servlets/purl/10126321)] 
1. **Privacy-aware recommendation with private-attribute protection using adversarial learning**. WSDM 2019.[[pdf](https://arxiv.org/pdf/1911.09872.pdf)] 
1. **Policy Learning for Fairness in Ranking**. NIPS 2019.[[pdf](https://proceedings.neurips.cc/paper/2019/file/9e82757e9a1c12cb710ad680db11f6f1-Paper.pdf)] [[code](https://github.com/ashudeep/Fair-PGRank)]
1. **Fairness of exposure in rankings**. KDD 2018.[[pdf](https://arxiv.org/pdf/1802.07281.pdf)] 
1. **Fairness-aware tensor-based recommendation**. CIKM 2018.[[pdf](https://par.nsf.gov/servlets/purl/10098220)] [[code](https://github.com/Zziwei/Fairness-Aware_Tensor-Based_Recommendation)]
1. **Fairness in decision-making - the causal explanation formula**. AAAI 2018.[[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/11564)] 
1. **On discrimination discovery and removal in ranked data using causal graph**. KDD 2018.[[pdf](https://arxiv.org/pdf/1803.01901.pdf)] 
1. **A fairness-aware hybrid recommender system**. FATREC 2018.[[pdf](https://arxiv.org/pdf/1809.09030.pdf)] 
1. **Fair inference on outcomes**. AAAI 2018.[[pdf](https://ojs.aaai.org/index.php/AAAI/article/download/11553/11412)] [[code](https://github.com/raziehna/fair-inference-on-outcomes)]
1. **Equity of attention: Amortizing individual fairness in rankings**. SIGIR 2018.[[pdf](https://arxiv.org/pdf/1805.01788.pdf)] 
1. **Fa*ir: A fair top-k ranking algorithm**. CIKM 2017.[[pdf](https://arxiv.org/pdf/1706.06368.pdf)] [[code](https://github.com/MilkaLichtblau/FA-IR_Ranking)]
1. **Beyond parity: Fairness objectives for collaborative filtering**. NIPS 2017.[[pdf](https://arxiv.org/pdf/1705.08804.pdf)] 
1. **Balanced neighborhoods for fairness-aware collaborative recommendation**. RecSys 2017.[[pdf](https://scholarworks.boisestate.edu/cgi/viewcontent.cgi?article=1002&context=fatrec)] 
1. **Controlling popularity bias in learning-to-rank recommendation**. RecSys 2017.[[pdf](https://www.researchgate.net/profile/Himan-Abdollahpouri/publication/318351355_Controlling_Popularity_Bias_in_Learning-to-Rank_Recommendation/links/5a1375450f7e9b1e573086d6/Controlling-Popularity-Bias-in-Learning-to-Rank-Recommendation.pdf)] 
1. **Considerations on recommendation independence for a find-good-items task**. Recsys 2017.[[pdf](https://scholarworks.boisestate.edu/cgi/viewcontent.cgi?referer=https://scholar.google.com.hk/&httpsredir=1&article=1010&context=fatrec)] 
1. **New fairness metrics for recommendation that embrace differences**. FAT/ML 2017.[[pdf](https://arxiv.org/pdf/1706.09838.pdf)] 
1. **Fairness-aware group recommendation with pareto-efficiency**. RecSys 2017.[[pdf](http://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/p107-xiao.pdf)] 
1. **Counterfactual fairness**. arXiv 2017.[[pdf](https://arxiv.org/pdf/1703.06856.pdf)] [[code](https://github.com/Kaaii/CS7290_Fairness_Eval_Project)]
1. **Censoring representations with an adversary**. ICLR 2016.[[pdf](https://arxiv.org/pdf/1511.05897.pdf)] 
1. **Model-based approaches for independence-enhanced recommendation**. IEEE 2016.[[pdf](https://www.kamishima.net/archive/2016-ws-icdm-print.pdf)] [[code](http://www.kamishima.net/iers/)]
1. **Efficiency improvement of neutrality-enhanced recommendation.**. RecSys 2013.[[pdf](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.6889&rep=rep1&type=pdf#page=5)] [[code](http://www.kamishima.net/inrs/)]
1. **Learning fair representations**. JMLR 2013.[[pdf](http://proceedings.mlr.press/v28/zemel13.pdf)] 
1. **Enhancement of the neutrality in recommendation**. RecSys 2012.[[pdf](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.416.1022&rep=rep1&type=pdf#page=12)] 
1. **Discrimination-aware data mining**. KDD 2008.[[pdf](https://dl.acm.org/doi/abs/10.1145/1401890.1401959)] 


### 3.8 Loop Effect


1. **Toward Pareto Efficient Fairness-Utility Trade-off inRecommendation through Reinforcement Learning**. WSDM 2022.[[pdf](https://arxiv.org/abs/2201.00140)] 
1. **AutoDebias: Learning to Debias for Recommendation**. .[[pdf](https://arxiv.org/pdf/2105.04170.pdf)] [[code](https://github.com/DongHande/AutoDebias)]
1. **A general knowledge distillation framework for counterfactual recommendation via uniform data**. SIGIR 2020.[[pdf](http://csse.szu.edu.cn/staff/panwk/publications/Conference-SIGIR-20-KDCRec.pdf)] [[code](https://github.com/dgliu/SIGIR20_KDCRec)]
1. **Influence function for unbiased recommendation**. SIGIR 2020.[[pdf](https://dl.acm.org/doi/abs/10.1145/3397271.3401321)] 
1. **Jointly learning to recommend and advertise**. KDD 2020.[[pdf](https://arxiv.org/pdf/2003.00097.pdf)] 
1. **Counterfactual evaluation of slate recommendations with sequential reward interactions**. KDD 2020.[[pdf](https://arxiv.org/pdf/2007.12986.pdf)] [[code](https://github.com/spotify-research/RIPS_KDD2020)]
1. **Joint policy value learning for recommendation**. KDD 2020.[[pdf](http://adrem.uantwerpen.be/bibrem/pubs/JeunenKDD2020.pdf)] [[code](https://github.com/olivierjeunen/dual-bandit-kdd-2020)]
1. **Degenerate feedback loops in recommender systems**. AIES 2019.[[pdf](https://arxiv.org/pdf/1902.10730.pdf)] 
1. **When people change their mind: Off-policy evaluation in non-stationary recommendation environments**. WSDM 2019.[[pdf](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/jagerman-when-2019.pdf)] [[code](https://github.com/rjagerman/wsdm2019-nonstationary)]
1. **Top-k off-policy correction for a reinforce recommender system**. WSDM 2019.[[pdf](https://arxiv.org/pdf/1812.02353.pdf)] [[code](https://github.com/massquantity/DBRL)]
1. **Improving ad click prediction by considering non-displayed events**. CIKM 2019.[[pdf](https://www.csie.ntu.edu.tw/~cjlin/papers/occtr/ctr_oc.pdf)] [[code](https://www.csie.ntu.edu.tw/~cjlin/papers/occtr/)]
1. **Large-scale interactive recommendation with tree-structured policy gradient**. AAAI 2019.[[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/4204)] [[code](https://github.com/chenhaokun/TPGR)]
1. **Deep reinforcement learning for list-wise recommendations**. KDD 2019.[[pdf](https://arxiv.org/pdf/1801.00209.pdf)] [[code](https://github.com/egipcy/LIRD)]
1. **Causal embeddings for recommendation**. RecSys 2018.[[pdf](https://arxiv.org/pdf/1706.07639.pdf)] [[code](https://github.com/criteo-research/CausE)]
1. **Stabilizing reinforcement learning in dynamic environment with application to online recommendation**. KDD 2018.[[pdf](https://www.researchgate.net/profile/Qing-Da/publication/324988927_Stablizing_Reinforcement_Learning_in_Dynamic_Environment_with_Application_to_Online_Recommendation/links/5b2b4321aca27209f3797d65/Stablizing-Reinforcement-Learning-in-Dynamic-Environment-with-Application-to-Online-Recommendation.pdf)] 
1. **Recommendations with negative feedback via pairwise deep reinforcement learning**. KDD 2018.[[pdf](https://arxiv.org/pdf/1802.06501.pdf)] 
1. **Drn: A deep reinforcement learning framework for news recommendation**. WWW 2018.[[pdf](http://personal.psu.edu/~gjz5038/paper/www2018_reinforceRec/www2018_reinforceRec.pdf)] 
1. **Deep reinforcement learning for page-wise recommendations**. RecSys 2018.[[pdf](https://arxiv.org/pdf/1805.02343.pdf)] 
1. **A reinforcement learning framework for explainable recommendation**. ICDM 2018.[[pdf](https://www.microsoft.com/en-us/research/uploads/prod/2018/08/main.pdf)] 
1. **Interactive social recommendation**. CIKM 2017.[[pdf](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=4975&context=sis_research)] 
1. **Off-policy evaluation for slate recommendation**. NIPS 2017.[[pdf](https://arxiv.org/pdf/1605.04812.pdf)] [[code](https://github.com/adith387/slates_semisynth_expts)]
1. **Factorization bandits for interactive recommendation**. WWW 2016.[[pdf](https://ojs.aaai.org/index.php/AAAI/article/view/10936)] 
1. **Deconvolving feedbackloops in recommender systems**. NIPS 2016.[[pdf](https://arxiv.org/pdf/1703.01049.pdf)] 
1. **Interactive collaborative filtering**. CIKM 2013.[[pdf](https://discovery.ucl.ac.uk/id/eprint/1401363/1/ir0422-zhao_ucl.pdf)] [[code](https://github.com/h324yang/interactiveCF)]
1. **A contextual-bandit approach to personalized news article recommendation**. WWW 2010.[[pdf](https://arxiv.org/pdf/1003.0146.pdf)] [[code](https://github.com/akhadangi/Multi-armed-Bandits)]



### 3.9 Other Bias



#### 3.9.1 Attention Bias
1. **Counteracting User Attention Bias in Music Streaming Recommendation via Reward Modification**. KDD 2022.[[pdf](https://dl.acm.org/doi/abs/10.1145/3534678.3539393)] 


#### 3.9.2 Duration Bias
1. **Deconfounding Duration Bias inWatch-time Prediction for Video Recommendation**. KDD 2022.[[pdf](https://arxiv.org/abs/2206.06003)] [[code](https://github.com/MorganSQ/Ks-D2Q)]


#### 3.9.3 Sentiment Bias

1. **Causal Intervention for Sentiment De-biasing in Recommendation**. CIKM 2022.[[pdf](https://dl.acm.org/doi/10.1145/3511808.3557558)] 
1. **Mitigating Sentiment Bias for Recommender Systems**. SIGIR 2021.[[pdf](https://dl.acm.org/doi/10.1145/3404835.3462943)] 

#### 3.9.4 Multiply Bias

1. **Bounding System-Induced Biases in Recommender Systems with a Randomized Dataset**. TOIS 2023.[[pdf](https://dl.acm.org/doi/10.1145/3582002)] [[code](https://github.com/dgliu/TOIS_DUB)]

#### 3.9.5 Other Bias
1. **Debiasing Learning based Cross-domain Recommendation**. KDD 2021.[[pdf](https://dl.acm.org/doi/10.1145/3447548.3467067)] 




## Tips
### We will keep updating this list, and if you find any missing related work or have any suggestions, please feel free to contact us (cjwustc@ustc.edu.cn).

### If you find this repository useful to your research or work, it is really appreciate to star this repository. Thank you!
