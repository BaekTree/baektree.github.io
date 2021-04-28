# vector-derivative
* 쉽게 쉽게 모든 단계
* notation
  * W의 경우 $W_i$은 row vector을 의미하고, 그외 다른 벡터 $v$의 경우 $v_i$은 element을 의미
  * output을 만나는 마지막 layer은 $L$, 처음 layer은 $1$, 그 중간 layer들은 $l$으로 표기.


* network

$$
    \begin{aligned}
        & \text{ layer 1} : z^{[1]}  \in R^{m_1 \times 1}, W^{[1]} \in R^{m_1 \times n}, X \in R^{n \times m}, a \in R^{m_1 \times 1}, b \in R^{m_1 \times 1}\\
        & z^{[1]} = W^{[1]} \cdot X + b^{[1]}\\
        & a^{[1]} = g(z^{[1]})\\
        
        \\

        & \text{ layer $l$ where } 1 \lt l \lt L : \\
        & z^{[l]} \in R^{m_l \times 1}, a^{[l]} \in R^{m_l \times 1},W^{[l]} \in R^{m_l \times m_{l-1}}, a^{[l-1]} \in R^{m_{l-1} \times 1}, b \in R^{m_l \times 1}\\

        & z^{[l]} = W^{[l]} \cdot a^{l-1} + b^{[l]}\\
        & a^{[l]} = g(z^{[l]}) \text{where g is element wise function. }\\


        \\
        & \text{ layer $L$ : } z \in R^{1 \times 1}, W^{[L]} \in R^{1 \times m_{L-1}}, a \in R^{1 \times 1}, b \in R^{1 \times 1}\\
        & z^{[L]} = W^{[L]} \cdot a^{[L-1]} + b^{[L]}\\
        & a^{[L]} = g(z^{[L]})\\
        & o = a^{[L]}\\
        & \mathcal{L} = \mathcal{L}(o)\\
    \end{aligned}
$$


* cost function

$$
J = \frac 1 m \sum \mathcal{L}^{(i)}
$$

* parameters

$$
    W^{[l]}, b^{[l]}, \text{for } 1 \le l \le L.
$$

* gradient

# layer $L$ 
* $z \in R^{1 \times 1}, W^{[L]} \in R^{1 \times m_{L-1}}, a \in R^{1 \times 1}, a^{[L-1]} \in R^{m_{l-1} \times 1}, b \in R^{1 \times 1}$

### $\frac{\partial \mathcal{L}}{\partial W^{[L]}} = \delta^{[L]} \cdot (a^{[L-1]})^T :  (1 \times m_{L-1})$

$$
    \begin{aligned}
        \frac{\partial \mathcal{L}}{\partial W^{[L]}_i} & : (1 \times 1)\\
        z^{[L]} & = W^{[L]} \cdot a^{[L-1]} + b^{[L]}\\

        \frac{\partial \mathcal{L}}{\partial W^{[L]}_i} & = \underbrace{\frac{\partial \mathcal{L}}{\partial z}}_{1 \times 1} \underbrace{\frac{\partial z}{\partial W^{[L]}_i}}_{1 \times 1}\\
        & = \delta^{[L]} \cdot a^{[L-1]}_i\\
        \\

        \frac{\partial \mathcal{L}}{\partial W^{[L]}} & : (1 \times m_{L-1})\\
        \frac{\partial \mathcal{L}}{\partial W^{[L]}} & = \underbrace{\frac{\partial \mathcal{L}}{\partial z}}_{1 \times 1} \underbrace{\frac{\partial z}{\partial W^{[L]}}}_{1 \times m_{l-1}}\\
        & = \begin{bmatrix}
            -\frac{\partial \mathcal{L}}{\partial W^{[L]}_i} -\\
        \end{bmatrix}\\
        & = \begin{bmatrix}
            -\delta^{[L]} \cdot a^{[L-1]}_i-\\
        \end{bmatrix}\\
        & = \delta^{[L]} \cdot (a^{[L-1]})^T\\
        \text{where } \delta^{[L]} & = \frac{\partial \mathcal{L}}{\partial z}\\
    \end{aligned}

$$

### $\frac{\partial \mathcal{L}}{\partial z} = \delta^{[L]} : (1 \times 1)$


$$
    \begin{aligned}

        \\
        \frac{\partial \mathcal{L}}{\partial z} &: (1 \times 1)\\
        \frac{\partial \mathcal{L}}{\partial z} &= \frac{\partial \mathcal{L}}{\partial o} \frac{\partial o}{\partial z} \\
    \end{aligned}

$$

### $\frac{\partial \mathcal{L}}{\partial b^{[L]}}$

$$
        \\
   \frac{\partial \mathcal{L}}{\partial b^{[L]}} = \frac{\partial \mathcal{L}}{\partial z} \frac{\partial z}{\partial b}\\
   = \delta^{[L]}\\
$$




# layer $l$
* $1 \le l \lt L : z^{[l]} \in R^{m_l \times 1}, W^{[l]} \in R^{m_l \times m_{l-1}}, a^{[l-1]} \in R^{m_{l-1} \times 1}, b \in R^{m_l \times 1}$
* $z^{[l]}  = W^{[l]} \cdot a^{[l-1]} + b^{[l]}$
* $a^{[l]}  = g(z^{[l]})$
## $\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{\partial \mathcal{L}}{\partial z^{[l]}}  \cdot (a^{[l-1]})^T$

$$
    \begin{aligned}
        \frac{\partial \mathcal{L}}{\partial W^{[l]}_{ij}} & : (1 \times 1)\\
        \frac{\partial \mathcal{L}}{\partial W^{[l]}_{ij}} & = \underbrace{\frac{\partial \mathcal{L}}{\partial z^{[l]}_i}}_{1 \times 1} \underbrace{\frac{\partial z^{[l]}_i}{\partial W^{[l]}_{ij}}}_{1 \times 1}\\
        & = \frac{\partial \mathcal{L}}{\partial z^{[l]}_i} (a_j^{[l-1]})\\
        & = \delta^l_i \cdot (a_j^{[l-1]})\\

        \\
        \frac{\partial \mathcal{L}}{\partial W^{[l]}_{i}} & : (1 \times m_{l-1})\\
         \frac{\partial \mathcal{L}}{\partial W^{[l]}_{i}} 
         & = \begin{bmatrix}
            -\frac{\partial \mathcal{L}}{\partial W^{[l]}_{ij}}-\\
        \end{bmatrix}\\
             & = \begin{bmatrix}
            -\delta^l_i \cdot (a_j^{[l-1]})-\\
        \end{bmatrix}\\
         
         & = \underbrace{\frac{\partial \mathcal{L}}{\partial z^{[l]}_i}}_{1 \times 1} \cdot \underbrace{(a^{[l-1]})^T}_{1 \times m_{l-1}}\\
        & = \delta^l_i \cdot (a^{[l-1]})^T\\

        \\
        \frac{\partial \mathcal{L}}{\partial W^{[l]}} & : (m_l \times m_{l-1})\\
        \frac{\partial \mathcal{L}}{\partial W^{[l]}} & = \begin{bmatrix}
            |\\
            \frac{\partial \mathcal{L}}{\partial W^{[l]}_{i}}\\
            |
        \end{bmatrix}\\
        & = \begin{bmatrix}
            |\\
            \frac{\partial \mathcal{L}}{\partial z^{[l]}_i}  \cdot (a^{[l-1]})^T\\
            |\\
        \end{bmatrix}\\
        & = \begin{bmatrix}
            |\\
            \delta^l_i \cdot (a^{[l-1]})^T\\
            |\\
        \end{bmatrix}\\
        & = <\delta^l , (a^{[l-1]})^T> [\because \text{ $(a^{[l-1]})^T$ is a vector.}]\\
        & = \delta^l \cdot (a^{[l-1]})^T\\
        & = \frac{\partial \mathcal{L}}{\partial z^{[l]}}  \cdot (a^{[l-1]})^T\\
    \end{aligned}

$$

---

## $\frac{\partial \mathcal{L}}{\partial z^{[l]}} = \frac{\partial \mathcal{L}}{\partial a^{[l]}} *  a'^{[l]}(z^{[l]})$
* $a^{[l]}  = g(z^{[l]})$
$$
    \begin{aligned}
        \\
        \frac{\partial \mathcal{L}}{\partial z^{[l]}_i} & = \delta^l_i\\
        & = \underbrace{\frac{\partial \mathcal{L}}{\partial a^{[l]}_i}}_{1 \times 1} \underbrace{\frac{\partial a^{[l]}_i }{\partial z^{[l]}_i}}_{1 \times 1}\\
        & =\frac{\partial \mathcal{L}}{\partial a^{[l]}_i} \cdot a'^{[l]}_i(z_i^{[l]}) \\

        \\
        \frac{\partial \mathcal{L}}{\partial z^{[l]}} & = \delta^l : (m_l \times 1)\\
        \frac{\partial \mathcal{L}}{\partial z^{[l]}} & = \begin{bmatrix}
            |\\
            \frac{\partial \mathcal{L}}{\partial z^{[l]}_i}\\
            |\\
        \end{bmatrix}\\
        & = \begin{bmatrix}
            |\\
            \frac{\partial \mathcal{L}}{\partial a^{[l]}_i} \frac{\partial a^{[l]}_i }{\partial z^{[l]}_i}\\
            |\\
        \end{bmatrix}\\
        & = \begin{bmatrix}
            |\\
            \frac{\partial \mathcal{L}}{\partial a^{[l]}_i} \cdot  a'^{[l]}_i(z_i^{[l]}) \\
            |\\
        \end{bmatrix}\\
        & = \frac{\partial \mathcal{L}}{\partial a^{[l]}} *  a'^{[l]}(z^{[l]}) \\

            
        \end{aligned}
$$

---

##  $\frac{\partial \mathcal{L}}{\partial a^{[l]}} = {W^{[l]}}^T \cdot \delta^{[l+1]}$
* $z^{[l]}  = W^{[l]} \cdot a^{[l-1]} + b^{[l]}$
* $z^{[l+1]}  = W^{[l+1]} \cdot a^{[l]} + b^{[l+1]}$

$$
        \begin{aligned}

        \\
        \frac{\partial \mathcal{L}}{\partial a^{[l]}_i} & : (1 \times 1)\\
        \frac{\partial \mathcal{L}}{\partial a^{[l]}_i} & = \underbrace{\frac{\partial z^{[l+1]}}{\partial a^{[l]}_i}}_{1 \times m_{l+1}} \cdot  \underbrace{\frac{\partial \mathcal{L}}{\partial z^{[l+1]}}}_{m_{l+1} \times 1}\\
        & \text{ denumerator formation이고 denumerator가 scalar이므로 shape 맞춰주기 위해 $z^{[l+1]}$을 transpose 가능.} \\
        & = {W^{[l]}}^T_i \cdot \delta^{[l+1]}\\
        & \text{$a_i$가 각 unit에 $W_{i1}$}와 내적.\\

        \\
        \frac{\partial \mathcal{L}}{\partial a^{[l]}} & : (m_l \times 1)\\
        \frac{\partial \mathcal{L}}{\partial a^{[l]}_i} & = \begin{bmatrix}
            |\\
             \frac{\partial \mathcal{L}}{\partial a^{[l]}_i}\\
            |\\
        \end{bmatrix}\\
        & = \begin{bmatrix}
            |\\
            \frac{\partial z^{[l+1]}}{\partial a^{[l]}_i}\cdot  \frac{\partial \mathcal{L}}{\partial z^{[l+1]}}\\
            |\\
        \end{bmatrix}\\
        & = \begin{bmatrix}
            |\\
            {W^{[l]}}^T_i \cdot \delta^{[l+1]}\\
            |\\
        \end{bmatrix}\\
        & = {W^{[l]}}^T \cdot \delta^{[l+1]}\\

        

    \end{aligned}
$$

## summing up : 
$z^{[l]} \in R^{m_l \times 1}, a^{[l]} \in R^{m_l \times 1},W^{[l]} \in R^{m_l \times m_{l-1}}, a^{[l-1]} \in R^{m_{l-1} \times 1}, b \in R^{m_l \times 1}$

$z^{[l]} = W^{[l]} \cdot a^{l-1} + b^{[l]}$
$a^{[l]} = g(z^{[l]}) \text{where g is element wise function. }$

$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{\partial \mathcal{L}}{\partial z^{[l]}}  \cdot (a^{[l-1]})^T$ (eq1)

$\frac{\partial \mathcal{L}}{\partial z^{[l]}} = \frac{\partial \mathcal{L}}{\partial a^{[l]}} *  a'^{[l]}(z^{[l]})$ (eq2)

$\frac{\partial \mathcal{L}}{\partial a^{[l]}} = {W^{[l]}}^T \cdot \delta^{[l+1]}$ (eq3)

$\therefore \frac{\partial \mathcal{L}}{\partial W^{[l]}} = {W^{[l]}}^T \cdot \delta^{[l+1]} * a'^{[l]}(z^{[l]}) \cdot (a^{[l-1]})^T$

$\frac{\partial \mathcal{L}}{\partial b^{[l]}}    = \delta^{[l]}$


# general case
$z \in R^{d}, W^{[L]} \in R^{d \times m}, a \in R^{m}, b \in R^{d}$

$$
    \begin{aligned}
      z & = W a + b\\
      a & = \sigma(z) \text{}\\
      \mathcal{L} & = \mathcal{L}(a)\\
      \\
      \frac{\partial \mathcal{L}}{\partial W} & = \frac{\partial \mathcal{L}}{\partial z} a^T \\
      \frac{\partial \mathcal{L}}{\partial b} & = \frac{\partial \mathcal{L}}{\partial z} \\
      \frac{\partial \mathcal{L}}{\partial a} & = W^T \cdot \frac{\partial \mathcal{L}}{\partial z} 
    \end{aligned}
$$

$z \in R^{d}, a \in R^{d}$

$$
    a = \sigma(z) \text{where $\sigma$ is element wise function.}\\
    \mathcal{L} = \mathcal{L}(a)\\

    \frac{\partial \mathcal{L}}{\partial z} = \frac{\partial \mathcal{L}}{\partial a} * \sigma '(z).

$$



## pseudo code
parameters : W, b, delta
```
init
    dZ(L)
    dW(L)
    db(L)

grad
    dZ(l) = W(l+1).dot.dZ(l+1) * A(l)'(z(l))
    dW(l) = dZ(l).dot.A(l-1).T
    db(l) = dZ(l)
```

# with m examples

* single example

$$
    v = Wu + b\\
    v \in R^{(n,1)}\\
    W \in R^{(n,d)}\\
    u \in R^{(d,1)}\\
    b \in R^{(n,1)}\\
$$

* m examples

$$
    
    V \in R^{(n,m)}\\
    W \in R^{(n,d)}\\
    U \in R^{(d,m)}\\
    b \in R^{(n,m)}\\
    
    V_t = WU + b\\
    U_{t+1} = g(V_t)\\
    V_{t+1} = WU_{t+1} + b\\
    \cdots
$$

V의 차원이 (..., m)으로 유지되면서 J까지 feed forward 된다. 각 examle이 독립적으로 계산된다. 서로 엮이지 않음.

## $J = \frac 1 m \sum L^i$
## $\mathcal{L} \in R^{m} = [\mathcal{L^{(1)}}, \cdots, \mathcal{L^{(i)}}, \cdots \mathcal{L^{(m)}}]$

# Back Propagation of m examples.
$z^{[l]} \in R^{m_l \times m}, a^{[l]} \in R^{m_l \times m},W^{[l]} \in R^{m_l \times m_{l-1}}, a^{[l-1]} \in R^{m_{l-1} \times m}, b \in R^{m_l \times m}$



## $\frac{\partial \mathcal{L}}{\partial Z^{[l]}} = \frac{\partial \mathcal{L}}{\partial A^{[l]}} *  A'^{[l]}(Z^{[l]})$

$$


    \begin{aligned}
    & \frac{\partial J}{\partial Z^{[l]}} = \frac 1 m \sum \frac{\partial \mathcal{L}^k}{\partial Z^{[l]}}\\
        
        & \frac{\partial \mathcal{L}^i}{\partial Z^{[l]}} = \begin{bmatrix}
            | & | & \vdots & | & \vdots & |\\
            0 & 0 & \cdots & \frac{\partial L^i}{\partial Z^{(i)[l]}} & \cdots & 0\\
            | & | & \vdots & | & \vdots & |\\
        \end{bmatrix}\\
        \\
        

        \\
        & \frac{\partial \mathcal{L}}{\partial Z^{[l]}} = \sum_k \frac{\partial \mathcal{L}^k}{\partial Z^{[l]}} (eq4)\\
        & = \begin{bmatrix}
            | &  \vdots & | & \vdots & |\\
            \frac{\partial \mathcal{L}^1}{\partial Z^{(1)[l]}} &  \cdots & \frac{\partial \mathcal{L}^i}{\partial Z^{(i)[l]}} & \cdots & \frac{\partial \mathcal{L}^m}{\partial Z^{(m)[l]}}\\
            |  & \vdots & | & \vdots & |\\
        \end{bmatrix}\\
        & = \underbrace{\begin{bmatrix}
            |&&|&&|\\
            \delta^{(1)}&\cdots&\underbrace{\delta^{(i)}}_{m_l \times 1}&\cdots&\delta^{(m)}\\
            |&&|&&|\\
        \end{bmatrix}}_{m_l \times m}\\
        & = \delta^{[l]}\\
        \\
        &\text{ since } \big [ \delta^{(i)[l]} = \frac{\partial \mathcal{L}^i}{\partial z^{(i)[l]}} = \frac{\partial \mathcal{L}^i}{\partial a^{(i)[l]}} * a'^{(i)[l]}(z^{(i)[l]})],\\
        \\
        & \delta^{[l]} \\
        &= \underbrace{\begin{bmatrix}
            |&&|&&|\\
            \frac{\partial \mathcal{L}^i}{\partial a^{(i)[l]}} * a'^{(i)[l]}(z^{(i)[l]})&\cdots&\underbrace{\frac{\partial \mathcal{L}^i}{\partial a^{(i)[l]}} * a'^{(i)[l]}(z^{(i)[l]})}_{m_l \times 1}&\cdots&\frac{\partial \mathcal{L}^i}{\partial a^{(i)[l]}} * a'^{(i)[l]}(z^{(i)[l]})\\
            |&&|&&|\\
        \end{bmatrix}}_{m_l \times m}\\
        &= \frac{\partial \mathcal{L}}{\partial A^{[l]}} *  A'^{[l]}(Z^{[l]})\\
    \end{aligned}\\

    \therefore \frac{\partial \mathcal{L}}{\partial Z^{[l]}} = \frac{\partial \mathcal{L}}{\partial A^{[l]}} *  A'^{[l]}(Z^{[l]})
$$

---

## $\frac{\partial \mathcal{L}}{\partial A^{[l]}} = {W^{[l]}}^T \cdot \delta^{[l+1]}$

$$
    \begin{aligned}
        \frac{\partial \mathcal{L}^i}{\partial a^{[l]}_i}&  = {W^{[l]}}^T \cdot \delta^{[l+1]}\\

        \frac{\partial \mathcal{L}}{\partial A^{[l]}} 
        & = \begin{bmatrix}
            - \frac{\partial \mathcal{L}^i}{\partial a^{[l]}_i} -\\
        \end{bmatrix}\\
        & = \begin{bmatrix}
            -{W^{[l]}}^T \cdot \delta^{(i)[l+1]}-\\
        \end{bmatrix}\\
        & = {W^{[l]}}^T \cdot  \delta^{[l+1]}\\
    \end{aligned}\\

    \therefore \frac{\partial \mathcal{L}}{\partial A^{[l]}} = {W^{[l]}}^T \cdot \delta^{[l+1]}\\
$$

---

## $\frac{\partial J}{\partial W^{[l]}} =\frac 1 m \frac{\partial \mathcal{L}}{\partial Z^{[l]}}  \cdot (A^{[l-1]})^T$
* w의 경우 BPTT을 하면서 sum이 자동으로 포함되어 있음...

* single example : $\frac{\partial \mathcal{L}}{\partial W^{[l]}}$

$$
\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \underbrace{\frac{\partial \mathcal{L}}{\partial z^{[l]}}}_{m_l \times 1}  \cdot \underbrace{(a^{[l-1]})^T}_{1 \times m_{l-1} }\\
\\
\frac{\partial J}{\partial W^{[l]}} = \underbrace{\frac{\partial J}{\partial z^{[l]}}}_{m_l \times m}  \cdot \underbrace{(A^{[l-1]})^T}_{m \times m_{l-1} }\\


\begin{aligned}
    
  \frac{\partial J}{\partial W^{[l]}_{ij}} & = \frac{\partial J}{\partial z_k} \frac{\partial z_i}{\partial W_{ij}}\\ 
  & = \underbrace{\delta_i}_{1 \times m} \cdot \underbrace{(a_i)^T}_{m \times 1}\\
\end{aligned}


$$
* m examples
$$
    \begin{aligned}
        
    & J = \frac 1 m \sum^m_i L^i\\
    & \frac{\partial \mathcal{L}^i}{\partial W^{[l]}} = \underbrace{\frac{\partial \mathcal{L}^i}{\partial z^{[l](i)}}}_{m_l \times 1}  \cdot \underbrace{(a^{[l-1](i)})^T}_{1 \times m_{l-1} }\\

    & = \delta^{(i)} \cdot (a^{[l-1](i)})^T \text{ from (eq1)}\\
    & \text{m 개의 example은 column vector으로 표현된다. 따라서 }\\

    \\
    \\
    & \frac{\partial \mathcal{L}}{\partial W}  =  \underbrace{\frac{\partial \mathcal{L}}{\partial Z^{[l]}}}_{m_l \times m}  \cdot \underbrace{\frac{\partial Z^{[l]}}{\partial W}}_{m \times m_{l-1} }(eq5)\\
    & =\delta^{[l]} \cdot \underbrace{(A^{[l-1]})^T}_{m \times m_{l-1} }\\
    & =  \underbrace{\begin{bmatrix}
        |&&|&&|\\
        \delta^{(1)}&-&\underbrace{\delta^{(i)}}_{m_l \times 1}&-&\delta^{(m)}\\
        |&&|&&|\\
    \end{bmatrix}}_{m_l \times m}
    \cdot
    \underbrace{\begin{bmatrix}
        -(a^{[l-1](1)})^T-\\
        |\\
        \underbrace{-(a^{[l-1](i)})^T-}_{1 \times m_{l-1}}\\
        |\\
        -(a^{[l-1](m)})^T-\\
    \end{bmatrix}}_{m \times m_{l-1}}\\
    & = \delta \cdot A^T\\


    & \text{$\frac{\partial \mathcal{L}}{\partial z^{[l]}}$의 column 마다 각 example들이 전체 m개 나열되어 있다.}\\
    & \text{$A^T$의 row 마다 각 example들이 전체 m개 나열되어 있다.}\\
    & \frac{\partial \mathcal{L}}{\partial W_{11}} = \sum^m_{k=1} \delta_{1k} \cdot (A^T)_{k1}\\
    & \text{$\frac{\partial \mathcal{L}}{\partial W_{11}}$을 구할 때 $\delta$의 첫번째 row와 $(A^T)$의 첫번째 column이 내적.}\\

    & \text{$\frac{\partial \mathcal{L}}{\partial W_{ij}}$을 구할 때 내적으로 m개의 example의 값이 모두 합해진다.}\\
    \\
    & \text{Recall that}\\
        & \frac{\partial \mathcal{L}^k}{\partial W^{[l]}_{ij}}  : (1 \times 1)\\
        & \frac{\partial \mathcal{L}^k}{\partial W^{[l]}_{ij}}  = \underbrace{\frac{\partial \mathcal{L}^k}{\partial z^{[l]}_i}}_{1 \times 1} \underbrace{\frac{\partial z^{[l]}_i}{\partial W^{[l]}_{ij}}}_{1 \times 1}\\
        & = \frac{\partial \mathcal{L}^k}{\partial z^{[l]}_i} (a_j^{[l-1]})\\
        & = \delta^l_i \cdot (a_j^{[l-1]})\\
  
\\
    
    


    \\
    & \therefore \frac{\partial J}{\partial W} =\frac 1 m\sum_k^m \frac{\partial \mathcal{L}^k}{\partial W}\\
    \\
    & \text{$\frac{\partial \mathcal{L}}{\partial W}  =  \frac{\partial \mathcal{L}}{\partial Z^{[l]}}  \cdot \frac{\partial Z^{[l]}}{\partial W}(eq5)$을 사용하여 내적으로 명시적인 summation을 없앤다. eq4 사용.}\\
    & \frac{\partial \mathcal{L}}{\partial Z^{[l]}} = \sum_k \frac{\partial \mathcal{L}^k}{\partial Z^{[l]}} (eq4)\\
    \\
    & = \frac 1 m\sum^m_k \frac{\partial \mathcal{L}^k}{\partial Z} \frac{\partial Z}{\partial W}\\
    & = \frac 1 m\frac{\partial \mathcal{L}}{\partial Z} \frac{\partial Z}{\partial W}\\
    \end{aligned}\\


    \frac{\partial J}{\partial Z^{[l]}} = \frac 1 m \sum \frac{\partial \mathcal{L}^k}{\partial Z^{(k)[l]}}\\

$$



---

## $\frac{\partial J}{\partial b^{[l]}}    = \frac 1 m\sum_i^m \frac{\partial \mathcal{L}^i}{\partial z} = \frac 1 m\sum_i^m \delta^{[l]}\\ \because \text{각 example의 layer l에서 b의 값은 동일.}$
* b의 경우 내적에 포함되지 않아서... 따로 sum을 해줘야 함...

---

## summary
$\mathcal{L} \in R^{m} = [\mathcal{L^{(1)}}, \cdots, \mathcal{L^{(i)}}, \cdots \mathcal{L^{(m)}}]$
* $\frac{\partial \mathcal{L}}{\partial A^{[l]}} = {W^{[l]}}^T \cdot \delta^{[l+1]}$
* $\frac{\partial \mathcal{L}}{\partial Z^{[l]}} = \frac{\partial \mathcal{L}}{\partial A^{[l]}} *  A'^{[l]}(Z^{[l]})$
* $\frac{\partial J}{\partial W^{[l]}} = \frac 1 m\frac{\partial \mathcal{L}}{\partial Z^{[l]}}  \cdot (A^{[l-1]})^T$
* $\frac{\partial J}{\partial b^{[l]}}    = \frac 1 m\sum_i^m \frac{\partial \mathcal{L}^i}{\partial z} = \frac 1 m\sum_i^m \delta^{[l]}$





## pseudo code
* dA = $\frac{\partial \mathcal{L}}{\partial A^{[l]}}$
* dW = $\frac{\partial J}{\partial W^{[l]}}$
* db = $\frac{\partial J}{\partial b^{[l]}}$
* dZ = $\frac{\partial \mathcal{L}}{\partial Z^{[l]}}$



```
init
    dZ(L)
    dW(L)
    db(L)
    dA(L)


grad
    dA(l) = W(l).T.dot.dZ(l+1)
    dZ(l) = dA(l).mult.A(l)'(Z(l))
    dW(l) = 1/m * dZ(l).dot.A(l-1).T
    db(l) = sum( dZ(l) )
```