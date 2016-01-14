---
title: "[Programming Python]Python list, dict and map."
tagline: ""
last_updated: 2016-01-11
category: big-data related
layout: post
tags : [Spark, py-spark, groupByKey]
---

# python 的列表与字典

上期博客中，对spark的 rdd， 大量使用了 map reduce 这种东西，初学者可能会摸不着头脑。

因此，这篇博客偷个懒，把以前的笔记整理出来，简明扼要的介绍下 python 的字典、列表两种常见数据结构，以及针对这两种数据结构的 ```map```, ```filter```, ```lambda``` 等较高级的用法。

高性能的 python 程序其实并不建议使用大量的 map。但这篇博客的主要目的是给 pyspark 以及 scala 这种编程语言作为铺垫的，而这些大数据编程语言在分析数据时，大量的使用了这些语句。因此使用这些语句仍然非常必要。

## 列表的基本操作：

### 定义：

```python
# 定义成由数字组成：
l_abc = [3, 2, 3, 4, 6]
```

### 列表的遍历（iterator）
遍历，把所有东西打出来。

```python
for abc in l_abc:
	print abc,
	
# 3 2 3 4 6 9
```

遍历的方法2， 使用```enumerate```得到遍历的次数：

```python
# 列表切片 l_abc[1:3]，列表的第 2 到 3 位

for i, abc in enumerate(l_abc[1:3]):
	print i, abc
```

### 列表的增加

```python
l_abc.append(33)
for abc in l_abc:
	print abc,

#3 2 3 4 6 9 33
```

### 列表的删除

```python
# 从后面删
val = l_abc.pop()  # equal to l_abc.pop(-1)
for abc in l_abc:
	print abc,

#3 2 3 4 6 9

# 从前面删
val = l_abc.pop(0)
for abc in l_abc:
	print abc,
#2 3 4 6 9
```


### 高级用法与 numpy 的使用：
生成等差数组（整数）

#### 方法1 生成 0~1 的 0.05 step 等差数组：

``` python
#方法1 先生成 0~100：
l_abc = range(0, 100, 5)
```

再各自除以100展示三种方法：

第一种， for 循环

```python
l_abc2 = []
for abc in l_abc:
	l_abc2.append(abc/float(100))

```

第二种, 列表推导式

列表推到式详见：[列表生成式](http://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/001431779637539089fd627094a43a8a7c77e6102e3a811000)

```python
l_abc2 = [ abc/float(100) for abc in l_abc ]
```

第三种, map + 普通函数 

map 用法详见：[map](http://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/0014317852443934a86aa5bb5ea47fbbd5f35282b331335000) 

```python
def div_100(val, divVal=100):
	return val/float(divVal)

l_abc2 = map(div_100, l_abc)
```

第四种, map + lambda 函数 

lambda 函数用法详见：[lambda](http://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/001431843456408652233b88b424613aa8ec2fe032fd85a000)

```python
l_abc2 = map(lambda x: x/float(100), l_abc)
```

以上四种的结果，以 tab 为分隔符打印结果, 保留两位有效数字```%1.2f```：

这个过程同样可以参考之前的四种方法。这里直接给你展示第四种：

```python
# 数字字符串转换成列表字符串
l_abc2_strOut = map(lambda x: "%1.2f" % x , l_abc2)

#打印
print "\t".join(l_abc2_strOut)
```



#### 方法2 使用 numpy. 
numpy 是 python 科学计算的专用模块。建议以后的数据运算使用这种。

```python
import numpy as np
np_abc = np.linspace(0, 1, 21)
```

打印出来。 同上。

```
l_abc2_strOut = map(lambda x: "%1.2f" % x , np_abc)
print "\t".join(l_abc2_strOut)
```


## 字典的基本操作

参考[廖雪峰博客](http://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/00143167793538255adf33371774853a0ef943280573f4d000)

### 基本用法

```python
M_abc = {
	'a' : 0,
	'b' : 1,
	'c' : 2,
	'd' : 3,
}
```

同样的生成方法, 使用 [zip](https://bradmontgomery.net/blog/pythons-zip-map-and-lambda/)：

```python
l_1 = ['a', 'b', 'c', 'd']  # key
l_2 = range(4)              # value
M_abc = dict(zip(l_1, l_2))
```

调出 ```a``` 对应的值：

```python
print M_abc['a']
# 0
```

删除与插入新值：

```python
del M_abc['a']
M_abc['a'] = 0
```

遍历：

```python
for key in M_abc:
	print key, M_abc[key]
```

所有的 key 值：

```python
print M_abc.keys()
```

结合列表的实际应用：

```python

l_abc = ['c', 'a', 'e', 'b']
for abc in l_abc:
	if abc in M_abc:
		print M_abc[abc]

"""output
2
0
1
"""
```

### 复杂用法

使用 functools(多个参数的map), [filter](http://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/001431821084171d2e0f22e7cc24305ae03aa0214d0ef29000)

```python
import functools
l_out = map(functools.partial(lambda y, x: y[x] if x in y else None, M_abc), l_abc)


# 以上等价于
def hasVal(M_abc, abc):
	out = None
	if abc in M_abc:
		out = M_abc[abc]
	return out

l_out = map(functools.partial(hasVal, M_abc), l_abc)


#扔掉空值
l_out = filter(lambda x: 0 if x is None else 1, l_out)
print "\n".join(l_out)

"""output
2
0
1
"""
```