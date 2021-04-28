---
title: "Heap-Proof"
date: 2020-12-29T15:34:30-04:00
categories:
  - DataStructure
tags:
  - DataStructure
  - Heap
---

* 복잡도가 $O(n)$이라는데 궁금했음...
* 알고리즘 교과서 heap 정리

# index heap

## complete Binary Tree
각 노드가 자식을 최대 2개까지만 가지는 트리(binary tree). 그리고 새로운 노드가 채워질 때 leaf의 왼쪽에서부터 채워지는 binary tree.

## 정의
### max heap
* parent 노드가 child 노드보다 항상 큰 complete binary tree

### min heap 
* parent 노드가 child 노드 보가 항상 작은 complete binary tree

## Height of Tree
* tree의 height은 root에서 leaf까지의 edge의 수이다.

## Height of node
* node의 height은 노드에서 leaf까지의 edge의 수이다.

## Full Complete Binary Tree
* complete binary tree 중에서 모든 노드가 2개의 child을 가진 tree. 꽉 채워진 binary tree이다.

## 왜 heap?
max heap이든 min heap이든 가장 위에 max 혹은 min 값을 가지고 있다. 데이터의 최소 값과 최대값을 빈번하게 호출하는데 O(1)의 시간이 걸려서 최대/최소값을 찾는데 아주 빠르다! 새로운 데이터를 삽입하는데에도 O(lg n)이 걸려서 삽입과 삭제도 괜찮은 편!

## array representation
* tree으로 도식화하지만 실제 구현은 array으로 한다(경우에 따라 연결리스트 등 다른 자료구조를 사용하기도 한다!).
* 배열이 주어졌을 때, i번째 index에 해당하는 노드의 자식의 index

```
    left child index = 2 * i, 
    right child index = 2 * i + 1.
    parent index = floor(2 / i).
```



## max heapify
* heap의 max 원소를 뽑아냈을 때, 새로운 원소를 삽입할 때, 힙의 특성을 유지시켜야 한다.
* 노드들의 위치를 변경시켜서 자식 노드가 부모 노드보다 크지 않도록 유지시키는 함수.
```
max-heapify(A,i)
    left = i.left
    right = i.right

    if A[i] < A[left]
        largest = left
    else
        largest = i

    if A[i] < A[right]
        largest = right

    if largest != i
        exchange(i, largest)
        max-heapify(A,i)
```
* 시간복잡도 O(lg n)
  * 최악의 경우 가장 작은 노드가 root 위치에 있을 때 맨 아래까지 끌어내려야 한다.
  * Complete Binary Tree이므로 전체 높이는 lg n이다.
  * 따라서 lg n번 max heapify을 실행시켜야 하므로 시간복잡도는 O(lg n)이다.
* 시간복잡도 O(lg n) 명확하게 보이기

## Lemma 1 : 전체 노드의 수가 n일때, complete binary tree의 최대 높이는  $\lfloor \lg n \rfloor.$ 
* complete binary tree을 그려보면 각 level의 맨 앞 노드가 2의 거듭제곱 번째 노드이다. 따라서 leaf의 맨 앞 노드가 아니라면 그 뒤의 노드들은 추가되어도 높이의 변화가 없다. 맨 앞의 노드가 추가될 때
*   $\log_2 n$ 의 값도 정수로 딱 나눠지면서 1이 커질 것이다. 그 다음 노드들은 다음 level의 맨 앞 노드전까지는 정수로 나눠지지 않는다. 따라서 해당 노드들의 높이는  $\lfloor \lg n \rfloor$.

## max heapify 시간복잡도

$$
T(n) = O(1) + T(?)
$$

재귀함수를 호출하는 시점을 보면 i와 자식 노드를 교환한 이후에 교환한 자식의 위치에서 호출한다. left subtree와 right subtree 중 i보다 더 큰 값을 가졌던 subtree에서 호출할 것이다. 점화식을 표현할 때 자식 subproblem의 노드 수로 표현해야 한다. 따라서 left subtree와 right subtree 중 최대한 많은 노드를 가진 subtree를 호출할 경우가 최악의 경우가 될 것이다. left와 right 중 어느 것이 더 많은 노드를 가지는가? 사실 right subtree는 left subtree보다 더 많은 노드를 가질 수 없다. heap을 complete binary tree이기 때문에 언제나 leaf의 왼쪽에서 오른쪽으로 새로운 노드를 채운다. heap이 full binary tree 상태일 때는 left와 right이 동일한 노드 수를 가질 것이다. 반면 left subtree만 full 상태일 때 left와 right의 노드 수 차이가 가장 커진다. 그리고 이때 left가 가진 노드 수가 i의 자식 subtree 중 가장 많은 노드 수를 가진 경우이다. 전체 노드가 n이라면 이 경우 left subtree는 얼마나 많은 노드를 가지는가?
## Lemma : Complete Binary Tree가 높이 h이면, 최대 노드의 수는 $2^{h+1}-1$개이다. 

$$
n(0) = 1\\
n(1) = n(0) + 2 = n(0) + 2 = 1 + 2\\
n(2) = n(1) + 2^2 = n(1) + 2^2 = 1 + 2 + 2^2\\
...\\
n(h) = n(h-1) + 2^{h} = 1 + \cdots + 2^{h}\\
= 2^{h+1}-1.
$$


## Collorally : Complete Binary Tree가 높이 h일때, leaf의 수는 최대 $2^h$개이다.
complete binary Tree가 가장 많은 노드를 가지는 Full Complete binary Tree일때

$$
\text{높이가 h-1일 때 노드의 수 } = n(h-1) = 2^{h}-1\\
\text{높이가 h일 때 노드의 수 } = n(h) = 2^{h+1}-1\\
n(h) - n(h-1) = 2^{h+1}-1 - (2^{h}-1)\\
= 2^h
$$

## Theorem : left subtree는 최대 $2n/3$의 노드를 가진다.
complete binary Tree에서 leaf가 half full 상태, 즉 왼쪽 subtree가 오른쪽 subtree보다 root에서부터 높이가 1 더 큰 full binary tree라고 하자. 왼쪽 subtree의 높이는 h이고, 오른쪽 subtree의 높이는 h-1이다. 전체 트리의 높이는 h+1이다. 
이때 왼쪽 subtree의 노드 수는 높이가 h-1 상태인 tree에서 (lemma에 의해) $2^h$개의 노드를 더 가지고 있다. 그리고 h-1 상태의 tree는 전체 노드 수가 (collorally에 의해) $2^{h}-1$개이다. $|L| = 2^h-1 + 2^h = 2^{h+1}-1$. 전체 노드 수는 leaf 자식 노드의 수 + right 자식 노드의 수 + 1(root)이다. 

$$
n = |L| + |R| + 1\\
= 2^{h+1}-1 + 2^{h} -1 + 1\\
= 3 \cdot 2^{h} -1\\
$$

전체 트리의 노드 수와 왼쪽 자식 트리의 비율을 보면

$$
    \frac{|L|}{n} = \frac{2^{h+1}-1 }{3 \cdot 2^{h} -1}\\
    \text{as } n \to \inf, h \to \inf.\\
    \lim_{n\to\inf} \frac{|L|}{n}  = \lim_{h\to\inf} \frac{2^{h+1}-1 }{3 \cdot 2^{h} -1}\\
    = \frac 2 3\\
$$

즉 heap에 n개의 노드가 있을 때 왼쪽 자식노드는 최대 $\frac {2n} {3}$개의 노드가 있다. 그리고 이때가 root의 자식 subtree가 가장 큰 노드를 가지는 경우이다.

## 그래서 시간복잡도는?

$$
T(n) = \Theta(1) + T(\frac{2n}{3})\\
$$

마스터 정리에 의해

$$
T(n) = O(\lg n)\\
$$

## insert
* Complete Binary Tree의 leaf의 맨 뒤에 새로운 노드를 삽입.
* 그리고 max-heapify을 실행시켜서 부모 노드가 자식 노드보다 항상 크도록 힙의 특성을 유지키신다.
* 시간복잡도
  * 맨뒤에 삽입 O(1)
  * max-heapify을 실행시켜서 O(lg n)
  * 총합 O(lg n)


## build heap
* 실제 heap을 구현할 때 배열을 자주 쓴다!
* 배열에서 자식 노드 부모 노드의 index는...
```
    left child index = 2 * i, 
    right child index = 2 * i + 1.
    parent index = floor(2 / i).
```

## Lemma : heap을 배열로 구현할 때, heap의 leaf nodes의 index는 $[\lfloor \frac{n}{2} \rfloor + 1, ..., n]$이다.

$\lfloor \frac{n}{2}\rfloor+1$ 노드의 left, right child을 살펴보자

$$
left = 2(\lfloor\frac{n}{2}\rfloor + 1)\\
= 2\lfloor\frac{n}{2}\rfloor + 2\\
\gt 2 \cdot (\frac{n}{2}-1) + 2\\
= 2 \cdot \frac{n}{2}-2 + 2\\
= n\\
$$

따라서 $\lfloor \frac{n}{2}\rfloor+1$노드의 left child는 n을 벗어난다! 즉 이 노드는 left 자식이 없다는 말이다. complete binary tree에서 left가 없으므로 그 뒤로 다른 노드들까지 모두 존재하지 않는다.

자식이 없는 노드는 leaf이다. 따라서 $\lfloor \frac{n}{2}\rfloor+1$부터 n까지 leaf이다.

* 코드
```
for i = floor(n/2) to 1 // O(n)
    max-heapify(A,i) // O(lgn)
```

* 시간복잡도 : $T(n) = O(n) * O(lg n) = O(nlgn)$
* 그런데 build heap에 있는 max heapify을 살펴보면 leaf 바로 위에서 max heap을 하고 위로 계속 올라간다. 그러면 다음 재귀 call에서는 항상 부모가 자식보다 큰 heap 특성을 만족한다. 그렇다면 사실상 max heapify을 실행할 때마다 걸리는 수행시간은 O(1)이다. 이것을 약 n/2번 만큼 반복하므로 전체 수행시간이 O(n)이라고 예측할 수 있다. 이것을 더 명확하게 보이자.
* 이 생각은 잘못되었다. 새로 마주한 i가 엄청 작아서 맨 아래로 내려가야 할 수도 있다. 그러면 최악의 경우 h가 걸린다.


## Lemma : Complete Binary Tree의 leaves 수는 최대 $\lceil \frac n 2 \rceil$이다.

* induction으로 증명
* base case

$$
    n = 1 = \lceil \frac 1 2 \rceil= 1 = \text{num of leaves}.
$$

* assume that the inductive hypothesis is true for h-1.
* general case
  * observation
    * (1)for hieght h, total leaves number = left subtree leaves + right subtree leaves. 
    * (2)Notice that total number of nodes is $n_l + n_r + 1$.
  * 전체 높이가 h이므로, root을 제외한 left subtree와 right subtree의 높이는 $h-1$이다.
  * left subtree leaves number = $\lceil \frac {n_l} {2} \rceil$ by inductive hypo for $h-1$.
  * right subtree leaves number = $\lceil \frac {n_r} {2} \rceil$ by inductive hypo for $h-1$.
  * (1)에 의해 total number of leaves = $\lceil \frac {n_l} {2} \rceil + \lceil \frac {n_r} {2} \rceil$이다.

$$
    \lceil \frac {n_l} {2} \rceil + \lceil \frac {n_r} {2} \rceil \le \lceil  \frac{n_l + n_r}{2} \rceil \\
    \le \lceil \frac{n_l + n_r + 1}{2} \rceil \text{((2)에 의해)}\\
    = \lceil \frac n 2 \rceil\\
$$

* 따라서 Complete Binary Tree의 최대 leaf 수는 $\lceil \frac n 2 \rceil$이고 이때는 Full Complete Binary Tree에서 모든 노드를 없애버렸을 때이다.

## Lemma : Complete Binary Tree에서 높이 h의 노드 수는 $\lceil \frac{n}{2^{h+1}}\rceil$이다.
* induction으로 증명

* base case
  * $h=0$에서 노드이 수도 1이고 전체 노드의 수도 1이다. 
  * $H=0 \to\lceil \frac{1}{2^{0+1}}\rceil =1$. 
* 높이 h-1에서 귀납가정을 참이라고 하자.
  * for $H=h-1$, num nodes = $\lceil \frac{n}{2^{h}}\rceil$.
* general case $H = h$,
  * observation
    * (1)Complete Binary Tree T에서 맨 아래 leaf을 모두 제거한 트리를 T'라고 하자. T'는 Full Binary Tree이다. 
    * (2)T와 T'는 맨 아래 줄을 빼고는 동일하다. 
    * (3)따라서 T의 임의의 h줄의 노드들은 T'의 h-1의 노드들과 동일하다(노드의 높이의 정의는 leaf에서 간선의 수이다). 
  * Lemma에 의해 T에서 없애버린 leaf의 수의 최대는 $\lceil \frac n 2 \rceil$이다.
  * 따라서 T'의 num nodes = $n - \lceil \frac n 2 \rceil = \lfloor \frac n 2 \rfloor$.
  * T의 노드 수 $N_h$와 T'의 노드 수 $N'_{h-1}$은 같다.
  * T'의 노드 수 $N'_{h-1}$에서 inductive hypo을 사용한다. 

$$
N'_{h-1}=\lceil \frac {n'} {2} \rceil =\lceil \frac {\lfloor \frac n 2 \rfloor} {2} \rceil\\
\le \lceil \frac{\frac{n}{2}}{2^h} \rceil\\
= \lceil \frac{n}{2 \cdot 2^h} \rceil\\
= \lceil \frac{n}{2^{h+1}} \rceil\\
N_h = N'_{h-1} = \lceil \frac{n}{2^{h+1}} \rceil\\
$$



* 따라서 Complete Binary Tree에서 높이 h의 노드 수는 $\lceil \frac{n}{2^{h+1}}\rceil$이다.

## Thrm : build heap의 시간복잡도는 $O(n)$이다.
* Lemma에서 확인한 정보
  * Complete Binary Tree의 leaves 수는 최대 $\lceil \frac n 2 \rceil$이고, 
  * Complete Binary Tree에서 높이 h의 노드 수는 $\lceil \frac{n}{2^{h+1}}\rceil$이다.

* build heap은 leaf 바로 전 노드부터 root까지 max heapify을 하면서 상승해간다. 
* max heapify는 $O(lgn) = O(h)$가 걸린다.
* 각 h에는 최대 $\lceil \frac{n}{2^{h+1}}\rceil$개의 노드가 있고
* h는 $\lfloor \lg n \rfloor$개가 있다.
* 따라서

$$
    T(n) = \sum_{h=0}^{\lfloor \lg n \rfloor} \lceil \frac{n}{2^{h+1}}\rceil O(h)\\
     = O(\sum_{h=0}^{\lfloor \lg n \rfloor} \lceil \frac{n}{2^{h+1}}\rceil h)\\
     = O(\sum_{h=0}^{\lfloor \lg n \rfloor}  \frac{n}{2^h} h)\\
     = O(n\sum_{h=0}^{\lfloor \lg n \rfloor}  \frac{h}{2^h} )\\
$$

* T(n)은 upper bound $n\sum_{h=0}^{\lfloor \lg n \rfloor}  \frac{h}{2^h}$이므로, $h\to \inf$일때

$$
    \sum_{h=0}^{\inf}  \frac{h}{2^h} = 2.\\
    \therefore T(n) = O(n 2)\\
    = O(n)\\

$$

* 따라서 build heap의 시간복잡도는 $O(n)$이다.