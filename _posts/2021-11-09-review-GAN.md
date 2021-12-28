---
title: "밑바닥 까지 다 파보는 GAN 리뷰"
last_modified_at: 2021-11-09T16:20:02-05:00
categories:
  - deep-learning
  - paper-review
  - nlp
tags:
  - DeepLearning
  - GAN
  - review
  - paper
---

# Paper Review GAN
Generative Adversarial Nets

수식부터 코드까지 제대로 파본 GAN 리뷰

## Abstraction

>We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models

>a generative model G
that captures the data distribution, and a discriminative model D that estimates
the probability that a sample came from the training data rather than G

>The training procedure for G is to maximize the probability of D making a mistake

>In the space of arbitrary
functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere. 

> In the case where G and D are defined
by multilayer perceptrons, the entire system can be trained with backpropagation

Contribution:
>There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples

## introduction
>In the proposed adversarial nets framework, the generative model is pitted against an adversary: a
discriminative model that learns to determine whether a sample is from the model distribution or the
data distribution.

>In this article, we explore the special case when the generative model generates samples
by passing random noise through a multilayer perceptron

>the discriminative model is also a
multilayer perceptron

>We refer to this special case as adversarial nets. In this case, we can train
both models using only the highly successful backpropagation and dropout algorithms and
sample from the generative model using only forward propagation. No approximate inference or
Markov chains are necessary.

## Adversarial Nets
> the models are both
multilayer perceptrons

data space $x$

generator distribution $p_g$ over data $x$

input noize $z$

prior of input noise $p_z(z)$

multi layer perceptron function $G(z; \theta_g)$. In other words, $G: z \mapsto x$.

multi layer perceptron function $D(x; \theta_d)$. In other words,  $D: x \mapsto R$. 

>We train D to maximize the probability of assigning the
correct label to both training examples and samples from G.

>We simultaneously train G to minimize
$log(1 − D(G(z)))$

G: D가 틀리는 확률을 극대화해야 한다. D가 틀리는 상황은 $D(G(z)) \to 1$이므로, $1 - D(G(z))$가 작을 때이다. monotonic 함수 log으로 극소화되는 파라미터 $\theta_g$을 찾는다(Negative Log Liklihood).

* Cross Entropy로 해석
* 일반적인 경우: KL div가 작도록 해야 함.

$$
\begin{aligned}
\arg \min _{\theta} D_{K L}(P \| Q) &=\arg \min _{\theta} \mathbb{E}_{x \sim P}\left[-\log Q_{\theta}(X)\right]-\mathcal{H}(P(X)) \\
&=\arg \min _{\theta} \mathbb{E}_{x \sim P}\left[-\log Q_{\theta}(X)\right] \\
&=\arg \max _{\theta} \mathbb{E}_{x \sim P}\left[\log Q_{\theta}(X)\right]
\end{aligned}
$$

* 지금 상황은 KL div가 커야 한다.
  * KL div : 실제는 False인데 True으로 갈 확률이 극대화되도록 해야 한다. 실제 G가 만든 것. D가 G을 실제로 착각한 확률: $D(G(z))$. 이게 1이 되어야 True으로 예측하는 것. 바꿔서 말하면 $1 - D(G(z)) = 0$이 되어야 True으로 인식하는 것이다. 

$$
\begin{aligned}
  \arg \max D_{KL} & = \arg \max E_P[-\log Q] \\
  & = \arg \max E_P[-\log (1- D(G(z))]\\ 
  & = \arg \max E_P[-\log (1- D(G(z))]\\
  & = \arg \min E_P[\log D(G(z)] \\
\end{aligned}
$$

D는 데이터 space x에서 오는 값이 실제 x에서 오는지 sample z에서 G을 통해 G(z)인지 잘 맞춰야 한다.  이것을 표현하면,

$$
\min _{G} \max _{D} V(D, G)=\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}(\boldsymbol{x})}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]
$$

근데 실험적으로 이렇게 했더니, $1-D(G(z))$가 당연히 초기에 data 값과 다르기 때문에 D가 다 맞춰버린다. 그래서 여기에 해당하는 loss가 낮아서 학습이 안됨. 

그래서 함수를 바꿔서 $\min \log (1-D(G(z)))$ 대신에 $\max \log D(G(z))$을 썼다고 한다. 

### e.g.
기존의 식
$$
\min_G \max_D \mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]
$$

학습 초기라고 생각해보자. $G(z)$에서 $\theta_g$을 가지고 그럴 듯한 분포를 만들었다. 그런데 아직 학습 초기라서 제대로 만들리가 없음. 엉망으로 만든 상태이면, D는 실제 data x와의 차이점을 아주 잘 맞출거임. 그러면 $D(G(z)) \to 0$에 가까워 진다. 식이 $\log 1 = 0$이 되어버림. 그래서 G가 학습할 수 있는 것이 없어진다. 

그래서 

$$
\max_G \max_D \mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log D(G(\boldsymbol{z}))]
$$

이렇게 바꿈. 수식의 의미는 동일함. 하지만 학습 초기에 D가 G을 잘 맞춰서 값이 큰 값이 남고, 여기서 극대값으로 더 올리는 $\theta_g$을 찾음. 

## 4. Theoretical Results
this minimax game has a global optimum for $p_g = p_{data}$.


사용하는 수학적 성질:
For any $(a, b) \in \mathbb{R}^{2} \backslash\{0,0\}$, the function $y \rightarrow a \log (y)+b \log (1-y)$ achieves its maximum in $[0,1]$ at $\frac{a}{a+b}$. The discriminator does not need to be defined outside of $\operatorname{Supp}\left(p_{\text {data }}\right) \cup \operatorname{Supp}\left(p_{g}\right)$.

최적화 목적 함수 $V(G, D)$.

$$
\begin{aligned}
V(G, D) &=\int_{\boldsymbol{x}} p_{\text {data }}(\boldsymbol{x}) \log (D(\boldsymbol{x})) d x+\int_{\boldsymbol{z}} p_{\boldsymbol{z}}(\boldsymbol{z}) \log (1-D(g(\boldsymbol{z}))) d z \\
\end{aligned}
$$

$G:z \mapsto x$이기 때문에 $p_z(z)$은 $p_g(x)$에, $dz$은 $dz$에 사상된다. 따라서.. 

$$
=\int_{\boldsymbol{x}} p_{\text {data }}(\boldsymbol{x}) \log (D(\boldsymbol{x}))+p_{g}(\boldsymbol{x}) \log (1-D(\boldsymbol{x})) d x
$$

위의 수학적 성질에 따라서...
$$
D_{G}^{*}(\boldsymbol{x})=\frac{p_{\text {data }}(\boldsymbol{x})}{p_{\text {data }}(\boldsymbol{x})+p_{g}(\boldsymbol{x})}
$$

이것을 이용해서 D가 최대인 $V(G,D)$을 다시 나타내면

$$
\begin{aligned}
C(G) &=\max _{D} V(G, D) \\
&=\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}}\left[\log D_{G}^{*}(\boldsymbol{x})\right]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}}\left[\log \left(1-D_{G}^{*}(G(\boldsymbol{z}))\right)\right] \\
&=\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}}\left[\log D_{G}^{*}(\boldsymbol{x})\right]+\mathbb{E}_{\boldsymbol{x} \sim p_{g}}\left[\log \left(1-D_{G}^{*}(\boldsymbol{x})\right)\right] \\
&=\mathbb{E}_{\boldsymbol{x} \sim p_{\text {data }}}\left[\log \frac{p_{\text {data }}(\boldsymbol{x})}{P_{\text {data }}(\boldsymbol{x})+p_{g}(\boldsymbol{x})}\right]+\mathbb{E}_{\boldsymbol{x} \sim p_{g}}\left[\log \frac{p_{g}(\boldsymbol{x})}{p_{\text {data }}(\boldsymbol{x})+p_{g}(\boldsymbol{x})}\right]
\end{aligned}
$$

D에 대한 극댁점을 찾았으니, 이제 G에 대한 극소점을 찾는다.

Theorem: $p_{data} = p_g$일때 G가 극소점을 가진다. 그리고 이때 극소값은 $-\log 4$이다.

증명: 
$p_{data} = p_g$이라고 하자. 그러면 $D^*_G(x) = 1/2$가 된다. 

이 값을 넣으면... 
$C(G) = log 1/2 + log 1/2 = − log 4$.

우리가 알고 싶은건 지금 이 값이 실제 극소값인지 여부이다. 그렇다면 이 값과 실제 극소값의 차이가 0인 것을 보이면 된다.

이 값과 실제 극소 값 $C^*(G)$의 차이는... 
$$
C(G)=-\log (4)+K L\left(p_{\text {data }} \| \frac{p_{\text {data }}+p_{g}}{2}\right)+K L\left(p_{g} \| \frac{p_{\text {data }}+p_{g}}{2}\right)
$$

이렇게 됨. Jensen=Shannon divergence에 따르면...

$$
C(G)=-\log (4)+2 \cdot J S D\left(p_{\text {data }} \| p_{g}\right)
$$

 두 distribution의 차이는 non-negative이고, 두 분포가 같을 때는 JSD = 0이다.

우리는 $p_{data} = p_g$ 상황이지롱~ 그래서 JSD = 0이 되고, $C^*(G) = -\log (4) + 0$이 되어서 진짜 $-\log (4)$가 최소값이 맞았다. 따단~


### 4.2 
위에서 극대점과 극소값이 수렴한다는 것을 증명했다. 이제 학습을 하는 과정에서 실제 저 값으로 이동하는지도 증명해야 함.

---
### 사전지식: convex 

convex set: 집합 C에 속하는 domain x로 만든 line segment도 C에 속하면 convex하다. 이때 segment을 만다는 weight의 sum = 1이 되어야 함.

refer
* https://stats.stackexchange.com/questions/515274/understanding-step-in-proof-of-gan-algorithm-convergence-involving-convexity

* https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf


convex set의 구성요소는 합이 1이 되는 실수와 각 points.
convet function은 convex set에서만 존재한다( convex 함수의 정의를 생각해보면... domain이 반드시 convex set을 사용한다.)

convex function preserve property: 합이 1이 되는 실수의 가중 sum of convex functions도 convex이다.

pairwise supremum: 합이 1이 되는 실수 중 supremum이 되는 convex는 당연히 convex이다. 

이것을 뽑은 것이 $sup_{\alpha}f_{\alpha}(x)$

---
### 사전지식: pairwise supremum of convex functions
>Evidently if $f$ is a convex function and $\alpha \geq 0$, then the function $\alpha f$ is convex. If $f_{1}$ and $f_{2}$ are both convex functions, then so is their sum $f_{1}+f_{2}$. Combining nonnegative scaling and addition, we see that the set of convex functions is itself a convex cone: a nonnegative weighted sum of convex functions,
$$
f=w_{1} f_{1}+\cdots+w_{m} f_{m}
$$
>is convex. Similarly, a nonnegative weighted sum of concave functions is concave. A nonnegative, nonzero weighted sum of strictly convex (concave) functions is strictly convex (concave).

>These properties extend to infinite sums and integrals. For example if $f(x, y)$ is convex in $x$ for each $y \in \mathcal{A}$, and $w(y) \geq 0$ for each $y \in \mathcal{A}$, then the function $g$ defined as
$$
g(x)=\int_{\mathcal{A}} w(y) f(x, y) d y
$$
>is convex in $x$ (provided the integral exists)

>The pointwise maximum property extends to the pointwise supremum over an infinite set of convex functions. If for each $y \in \mathcal{A}, f(x, y)$ is convex in $x$, then the function $g$, defined as
$$
g(x)=\sup _{y \in \mathcal{A}} f(x, y)
$$
>is convex in $x .$ Here the domain of $g$ is
$$
\operatorname{dom} g=\left\{x \mid(x, y) \in \operatorname{dom} f \text { for all } y \in \mathcal{A}, \sup _{y \in \mathcal{A}} f(x, y)<\infty\right\}
$$

from https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf

위의 pointwise supremum을 GAN 수렴 증명에 사용할 것이다. 저 말을 쉽게 풀어보면, 함수 f가 domain x에서 convex 하고, 이는 각 x마다 모든 y에서 성립한다고 가정하자. 그러면 모든 x에서 f(x,y)가 상한이 되게하는 $y^*$을 뽑아서 $(x, y^*)$ pair 함수를 만들고 g(x)라고 부른다. 

---


### 사전지식: subgradient

동기: 미분 불가능한 경우에도 극대/극소 값을 구하고 싶다는 동기에서 개발 되었다고 함

아이디어: convex 함수를 선형 근사하면 항상 under estimate한다는 특징을 사용!


$$
  x \in \text{Dom}_f\\
  f(y) \ge f(x) + f'(x) \cdot (y - x)\\
  \text {for all } y \in \text{Dom}_f\\
$$

convex 함수가 kink point가 있어서 미분불가능하다고 해서 어차피 under estimate 되면 실제 함수 값보다 같거나 작다. under estimate 값의 상한이 실제 함수 값이다. min 값을 구할 때 underestimate 값의 상한 중에서 최소 값을 찾으면 극소 값이다. 

정의: 다음을 만족하는 실수 g를 함수 f의 domain 값 x에서의 subgradient라고 한다.

$$
  x \in \text{Dom}_f\\
  f(y) \ge f(x_0) + g \cdot (y - x_0)\\
  \text {for all } y \in \text{Dom}_f\\
$$

더 알기: https://en.wikipedia.org/wiki/Subderivative

---

## 수렴성 증명
 GAN의 경우가 아니라 어떤 일반화된 경우를 고민해보고, 거기에 GAN을 적용해보자. 


$f(x,\alpha)$ 모든 $\alpha$에서 x에 대해 convex일 때, pointwise supremum을 만족하는 $\alpha$들이 각 x마다 있게 만든다. 이 함수를 $f(x)$라고 하자. $f(x) = \sup_{\alpha} f(x, \alpha)$ pairwise supremum에서 적었던 대로 x에 대해서 convex 하면 pairwise supremum 함수 역시 x에 대해`서 convex 한다. 따라서 f(x) 역시 x에 대해서 convex 하다. (일단 x에서 미분해서 gradient descent을 수행하면 적어도 f(x) 값이 내려갈 가능성은 생겼다)

$f(x)$ 중 하나의 $(x,\alpha)$ pair에서 하나의 점 $x_0$에서의 $\alpha$를 $\beta$라고 하고 이 $\beta$와 전체 domain x로 구성된 함수를 $f_\beta$라고 하자. 그리고 이때 $x_0$에서 $f_\beta$의 subgradient 중 하나를 g라고 하자.

$f_{\beta}$는 특정 하나의 점 x에서 $\alpha$가 sup인 $\beta$로 구성되어 있는 함수이다. 따라서 모든 점 $x_0$에서 sup으로 만드는 $(x,\alpha)$ pair로 만들어진 함수 f(x)보다 같거나 작다. $x_0$에서 동일하고 다른 점에서는 같거나 작기 때문이다. 

이것을 표현하면, $f(x) \ge f_\beta(x) \text{ for all } x \cdots (1)$.

위에서 $f_\beta$의 x에서 subgradient을 g라고 한다고 했다. subdifferential의 정의에 따라, $f_\beta(y) \ge f_\beta(x_0) + g(y-x_0) \text{ for all y } \in \text{Dom}_f \cdots (2)$이다. 

(1)과 (2)을 합쳐서,

$$
  f(y) \ge f_\beta(y) \ge f_\beta(x) + g(y-x)\\
  f(y) \ge f_\beta(x) + g(y-x) \cdots (3)\\
$$

$x_0$에서 함수 $f_\beta(x)$에서 사용했던 subgradient g을 각 x에 대한 sup $\alpha$로 구성된 함수 f(x)에서 그대로 사용할 수 있다. 따라서 f(x) 함수를 subgradient g으로 gradient descent 하면 극소 값을 나타내는 x으로 이동할 수 있다. 

의의: $x_0$이 아닌 다른 x을 선택했다면 다른 subgradeint가 생길 수 있음. 그런 경우를 고려하지 않고 그냥 sup중 하나 뽑아서 subgradient을 구하고 미분하면, 다른 x의 경우를 고려하지 않아도 전체 f(x)가 극소값으로 이동 가능함.

핵심 정리: 
1. 모든 y에 대해서 x에서 convex한 함수 f(x, y)가 있을 때, y에 대해서 f을 상한으로 하는 함수를 f(x)라고 하자.
2. f(x)에서 아무 한점을 뽑아서 그 점에서 subgradient을 구한다.
3. 그 subgradient는 f(x)에서 최적화할 때 그대로 사용해도 극소점을 향해 이동할 수 있다!

이제 이 논의를 다시 GAN에 적용해보자. 우리는 $p_g$을 $p_{data}$가 되었을 때 optimum 값이 된다는 것을 이미 보였다. 지금 하려는 작업을 다시 복기해보면, 목적함수 V에서 주어진 G에 대해서 D가 최적화를 끝낸 상황에서 다시 G가 최적화를 수행한다. G가 최적화를 수행할 때 네트워크 함수의 파라미터 $\theta_g$가 업데이트 되고, 이 네트워크 함수의 feed forward 결과를 $p_g$으로 표현한다. 

목적함수 V에서 D가 이미 극대화를 했고, 이제 G으로 최적화를 하는 상황이다. 따라서 D에 대한 값은 상수이다. 앞의 항은 상수이므로 미분에서 제외.

$$
\begin{aligned}
V(G, D) &=\int_{\boldsymbol{x}} p_{\text {data }}(\boldsymbol{x}) \log (D(\boldsymbol{x}))+p_{g}(\boldsymbol{x}) \log (1-D(\boldsymbol{x})) d x
\end{aligned}
$$

$$
V(G, D^*) = \int_{\boldsymbol{x}} p_{g}(\boldsymbol{x}) \log (1-D^*(\boldsymbol{x})) d x
$$

$p_g$는 G의 파라미터 $\theta_g$로 표현되는 네트워크 함수이고 가짜 샘플 x을 생성한다. $\theta_g$을 미분으로 최적화하는 단계이기 때문에 $p_g$로부터 만들어진 $\boldsymbol {x}$와 $$D^{*}(\boldsymbol {x})$$는 여기에서 상수이다. 따라서 $\log (1-D^*(\boldsymbol{x}))$ 항은 상수이므로 convex이다. 즉 V는 $p_g$에 대해서 convex 하다. 

그리고 $$V(G, D^{*})$$은 $$D^{*}$$으로 D에서 최대화된 값이다. 이 부분이 위의 일반화한 경우의 $\sup_{\alpha} f(x)$이다. pairwise sup convex 함수의 한 위치에서 구한 subgradient는 전체 pairwise sup convex 함수에서 동일하게 최적화하는데 적용될 수 있다. 따라서 현재 주어진 G에서 최적화된 D을 극대화한 지점에서 subgradient을 사용해서 $p_g$을 최적화하면, 현재 $p_g$에서 뿐만 아니라 다른 $p_g$의 경우에 나올 수 있는 $D^*$에서의 최적화를 포함한다. iteration을 돌면서 $p_g$가 업데이트 되고 변할 것이다. 그리고 거기에 따라서 최적 D도 변화한다. 하지면 매 상황에서 D가 supremum의 하나이므로, $p_g$는 계속해서 점진적으로 global optimum을 향해서 수렴해갈 수 있담.

refer
* https://math.stackexchange.com/questions/2226794/convergence-of-gans


## 코드
