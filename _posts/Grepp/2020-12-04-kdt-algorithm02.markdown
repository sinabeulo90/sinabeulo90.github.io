---
layout: post
title:  "Python 참고 사항들"
date:   2020-12-04
categories: "Grepp/KDT"
tags: Python
plugins: mathjax
---

## C++ swap 함수 대신 쓸수 있는 방법

{% highlight Python %}
A, B = 3, 6
print(A, B)
# 3 6

A, B = B, A
print(A, B)
# 6 3
{% endhighlight %}


-----


## dict.get() vs. dict.setdefault()

1. [dict.get(key[, default])](https://docs.python.org/3/library/stdtypes.html#dict.get)
    - 사전(Dictionary)에 등록되어 있는 키의 값을 불러온다.
    - 등록된 키가 없으면 default를 반환한다.
2. [dict.setdefault(key[, default])]((https://docs.python.org/3/library/stdtypes.html#dict.setdefault))
    - 사전(Dictionary)에 등록되어 있는 키의 값을 불러온다.
    - 등록된 키가 없으면 **키를 등록한 뒤**에 default를 반환한다.

{% highlight Python %}
# dict.get(key[, default])와 dict.setdefault(key[, default]) 비교

A = {}  # 사전(Dictionary) 정의

# dict.get(key[, default])을 사용할 경우
A.get((1, 2), [])
# []
print(A)
# {}

# dict.setdefault(key[, default])을 사용할 경우
A.setdefault((1, 3), [])
# []
print(A)
# {(1, 2): []}
{% endhighlight %}


-----


## enumerate(), list[::-1]
- [enumerate(iterable, start=0)](https://docs.python.org/3/library/functions.html#enumerate) : start 매개변수에 숫자를 지정하여 인덱스의 시작 번호를 정할 수 있다.
- list[::-1]: 리스트를 역순으로 표현할 수 있다.

{% highlight Python %}
A = [ num for num in range(5) ]
for idx, num in enumerate(A, start=1):
    print(idx, num)
# 1 0   <- 인덱스 값이 1부터 시작한다.
# 2 1
# 3 2
# 4 3
# 5 4

A[::-1]
# [4, 3, 2, 1, 0]
{% endhighlight %}


-----


## Counter.substract()
- [collections.Counter.subtract([iterable-or-mapping])](https://docs.python.org/3/library/collections.html#collections.Counter.subtract): counter에서 같은 키끼리 값을 뺄 수 있다.

{% highlight Python %}
from collections import Counter
A = Counter(["B", "C", "B", "C", "C"])
print(A)
# Counter({'C': 3, 'B': 2})

A.subtract(["C", "C", "C", "A"])
print(A)
# Counter({'B': 2, 'C': 0, 'A': -1})
{% endhighlight %}


-----


## 2차원 배열을 만들 때 주의할 점
- `[ [0] * N] ] * N`
    - 올바른 2차원 배열을 만들 수 없다.
    - `[ [0] * N ]`까지는 0이 N개인 리스트를 만들 수 있다.
    - 하지만 `[ "0이 N개인 리스트" ] * N`은 **0이 N개인 리스트 객체의 메모리 주소를 단순히 N번 복사된 리스트**가 만들어진다.
- `[ [0] * N for _ in range(N) ]`
    - 2차원 배열을 만들 수 있다.

{% highlight Python %}
N = 3
[ [0] * N ] * N
# [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
[ id([0] * N) ] * N     # id(): 객체의 메모리 주소값 반환
# [140315202592704, 140315202592704, 140315202592704]   # 주소가 동일하다.

[ [0] * N for _ in range(N) ]
# [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
[ id([0] * N) for _ in range(N) ]
# [140315202593536, 140315202591872, 140315202593536]   # 주소가 모두 다르다.
{% endhighlight %}


-----


## `S.add(5)` vs. `S |= set(5)`
- `L.append(5)` vs. `L += L + [5]`
    - 리스트는 mutable한 객체이다. mutable하다는 것은 상태를 바꿀 수 있다는 의미이다.
    - `L.append(5)`: 기존에 있는 리스트에 새로운 값을 추가하므로 $O(1)$의 시간이 걸린다.
    - `L += [5]`: `L = L + [5]`와 의미가 같으며, 새로운 리스트를 새로 생성하므로 **시간복잡도가 $O(n)$**이 된다.
- `S.add(5)` vs. `S |= set(5)`
    - 집합(Set) 또한 mutable한 객체이다.
    - `S.add(5)`: 기존에 있는 집합에 새로운 값을 추가하므로 $O(1)$의 시간이 걸린다.
    - `S |= set(5)`: `S = S | set(5)`와 의미가 같으며, 새로운 집합을 생성하므로 **시간복잡도가 $O(n)$**이 된다.
- `str += "world"`
    - 문자열은 immutable한 객체이다. immutable하다는 것은 상태를 바꿀 수 없다는 의미이다.
    - `str += "world"`: 기존에 있는 문자열에 새로운 문자열을 추가하여 생성하므로 **시간복잡도가 $O(n)$**이 된다.
    - `str = "".join([Hello”, “World”])`로 하는 방법이 `str += "world"`보다 훨씬 빠르다.
