---
title: "2021-01-01-python-recap"
date: 2021-10-11T15:34:30-04:00
categories:
  - python
tags:
  - python
  - recap
---

python recap

# eashdict
dict인데 attribute으로 접근 가능!
jupyter notebook 에서
argparser 있는데
에러
ipykernel_launcher.py: error: unrecognized arguments

```
import easydict
args = easydict.EasyDict({
 
        "batchsize": 100,
        ...
})
```

# numpy
* masking
부등호 사용

np < 10
각 index에 true or false nd array return

array, matrix 모두 가능

* reshape(..., -1)
...만 남기고 모두 1차원으로 만든다.

# numpy argsort
* 이거 땜에 고생 좀 했다... 여기에 말하면서 풀어야지...
* google에 argsort wrong으로 치면 원성이 자자 하다... ㅜㅜ 
* 기본적인 사용
  * np.argsort(arr)
    * sort 순으로 index가 나온다.

```py
idx = np.argsort(arr)
arr[idx] # => numpy에 [] operator안에 ndarray가 들어가면 그 안의 index만 떼어 출력해준다.
```

* 여기까진 좋다! 더 큰 문제가 남았음. 
* 2차원 배열을 axis = 1으로 정렬하려고 하면?

```
arr = np.random.uniform(low = 0, high = 10, size=(3,2))
idx = np.argsort(arr) # axis = -1이 default이다. 2차원의 경우 가로로 정렬하고 싶을 때 사실 아무것도 안해도 됨.
arr[idx] # 이렇게 할 수 있으면 좋겠지만...
```

* 이케 하면 안됨... ndaray으로 index에 해당하는 값만 뽑아내느게 2차원은 안되는 듯. 뭔가 값이 중복되어서 나옴. 그래서

```
arr = np.random.uniform(low = 0, high = 10, size=(3,2))
idx = np.argsort(arr) # (3,2)

final = []
for i, a in enumerate(arr):
    final.append( arr[idx[i]] )# (2)
final # 이렇게 해야만 끝난다 ㅜㅜ

```

# pandas 함수를 정리해본다.

# transpose
T

# 첫번재 row을 column으로
df.columns = df.iloc[0]
df.drop(df.index[0])

row의 기준은 index으로 한꺼번에 할 수 있는 모양이다!

# colume의 데이터 내용을 split
pandas.Series.str.split

# Series 초기화
s = pd.Series([val, val, ...])

# Series append
s.append(obj)
여기서 obj는 list!
s.append([1,2,3])
s.append([4,5,6])

# 형변환
int(val)

# 정렬
sorted(list...)


# dictionary
C의 map 마찬가지로 key가 없으면 자동으로 생성한다.

```
d = {}
d["key"] = 5
d[9] = 9
```

+= 연산자는 자동으로 생성해서 업데이트 하지 않는다. 따라서 key가 있는지 확인을 해야 함.

```
d = {}

d[9] += 1 # error

if 9 not in d:
    d[9] = 8
d[9] += 1 # result : d[9] = 9
```

# dictionary

지긋지긋한 dictionary

items()
values()
keys()

이렇게만 하면 dictionary type을 반환한다. "사용"하려면 list으로 바꿔야 한다. 

```
list(dict.items())
list(dict.values())
list(dict.keys())

for k,v in dict.items()
    ...

```

# Counter
dictionary with count of frequency?
```
from collections import Counter

Counter('hello world') # Counter({'l': 3, 'o': 2, 'h': 1, 'e': 1, ' ': 1, 'w': 1, 'r': 1, 'd': 1})

```

## Counter Intersection
각 char에서 공통으로 존재하는 문자의 최소 count 반환

```

    # str1 = 'ababcc'
    dict1 = Counter(str1)
    # dict1={'a':2,'b':2,'c':2}
    
    # str2 = 'abbbcc'
    dict2 = Counter(str2)
    # dict2={'a':1,'b':3,'c':2}
  
    # take intersection of two dictionries
    # output will be result = {'a':1,'b':2,'c':2}
    result = dict1 & dict2
```

## Counter element
```
    # given Counter({'a':1,'b':2,'c':2})
    cntr = Counter({'a':1,'b':2,'c':2})
    cntr.elements()
    #[a,b,b,c,c]
    for c in cntr.elements():
        print(c)
    #a
    #b
    #c
    #...
```


## Counter most_common(K)
상위 K번째 k,v까지만 반환

```
Counter('hello world').most_common(2) 
# Counter({'l': 3, 'o': 2})
```


# None
check if object is none

```
if obj == None:
    ...
if obj is None:

if obj != None:

if obj is not None
```

# nested function
```
def outer():
    outerVal = 0
    def inner():
        print(outerVal) # undefined error
        outerVal = 1 # totally different variable from the outside one
    
    def inner2():
        nonlocal outerVal
        print(outerVal) # print 0. same reference with outside variable
        outerVal = 1 # change the outside variable.
```

inner function에서 outer variable을 변경시키지 못하는 것은... outer variable이 immutable variable이기 때문이다. String, Int, ... 등은 파이썬에서 immutable이다.
반면 list 타입은 mutable이다. 따라서 outer scope에서 list을 정의하고 별다른 새로운 정의와 scope 변경 없이 inner scope에서 list에 추가/제거하면 globally 변경 됨.

# list empty check
```
que = []
whlie que:
    ...
```

# 2d array copy
```
l = [1,2,3]
l2 = l

l2[0] = 9 # change l as well. because everything is reference in python. we need copy() to copy values.

l = [1,2,3]
l2 = l.copy()
l2[0] = 9 # l1 = [1,2,3], l2 = [9,2,3]

l = [[1,2,3], [4,5,6]]

l2 = l.copy()
l2[0][0] = 9 # change l as well. while copying inner list, it copy the inner list reference! what we need is copy DEEP.

import copy
l = [[1,2,3], [4,5,6]]

l2 = copy.deepcopy(l)


```

# list pop
default는 마지막 원소를 반환

```
pop()
```

argument으로 index을 지정가능. 이걸로 queue 처럼 쓸 수 있음.

```
que = [...]
wihle que:
    head = pop(0) # 마지막 원소를 리턴하면 다음 while iteration에서 종료
```

# leetcode local
```
from typing import List # if List is not defined...



class Solution(object):
    def func(...):
        #your implementation

foo = Solution()
foo.func(...)
print(foo.func(...))
```

# print format
```
print("Decimal : %2d, Float : %5.2f" % (1, 05.333))
print("string %d %d with mod and parenthesis and commas" %(1,2))
```


# jupyter notebook -> markdown

jupyter nbconvert --to markdown notebook.ipynb

# pass by assignment = pass by reference value
not pass by copy.
kind of pass by reference.
but copy the reference value.

case 1
```
def func(arg)
    arg.append("new element")
    print("after change: ", arg)

data = ["This", " is ", "original ", "data " ]

print(data) # This is original data
func(data) # This is original data new element
print(data) # This is original data new element
```

case 2
```
def func(arg)
    print("before change: " , arg)
    arg = ["New ", "data"]
    print("after change: ", arg)

data = ["This", " is ", "original ", "data " ]

print(data) # This is original data
func(data) # New data
print(data) # This is original data new element
```

함수에 전달되는 arg에 기존 data의 reference 값이 copy by value된다. 그래서 함수 내부에서 내용을 수정하면 바깥에서도 바뀐다.
그런데 함수 내부에서 새로 assign을 하면 reference 값이 새로운 데이터로 바뀜. 그래서 바깥에서는 안바뀜.

포인터의 주소값을 복사해서 전달한 것과 동일해보인다. 



# %run Leetcode_complete_binary_generator.ipynb
ipython import


```
%run Leetcode_complete_binary_generator.ipynb

r1 = [1,3,2,5]
r2 = [2,1,3,None,4,None,7]

r1 = createNode(r1)
print(printNode(r1))
```

# defaultdict in Hash Map Problem...

```
from collections import defaultdict


d = {}
for w in word:
    if w not in d[w]:
        d[w] = 0
    else:
        d[w] += 1

# equivalant to

dd = defaultdict(int) # int() 함수가 0을 반환. key가 없을 때 넣을 default value 값을 지정하는 것.
for w in word:
    dd[w] += 1



```

# OrderedDict

```
from collections import OrderedDict

d = {}
...

od = OrderedDict(d)
# sort by key...
```

# sort dict by value

```

d = {... : ..., ...}

d.items() # [(key, value), ...]

sorted(d.items(), key = lambda x:x[1]) # x is each pair. x[1] is the value. if sort by key, use x[0]

```

# askii

* ch.isalpha(): return True or False whether ch is in 'a' ~ 'z' or 'A' ~ 'Z'.
* ord(ch) : return number of char in askii code
* char(num) : return char following askii code


## regular expression
split(' ')으로 하면 white space가 두개 이상일 때 도 중에 빈 문자열을 포함시킨다. split()을 하면 편-안. 

```
import re

re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]','', 'Reuters  Shortsellers Wall Streets dwindling\\band of ultracynics are seeing green again')
re.sub('\\\\',' ', 'Reuters  Shortsellers Wall Streets dwindling\\band of ultracynics are seeing green again')
```

[]을 쓰면 이 안에 있는 별개의 것들을 동시에 골라서 선택. 위 코드는 저 안에 있는 것들을 삭제함.