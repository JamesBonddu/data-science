# Pandas

Dimensions|	Name|	Description
:-|:-|:-
1|	Series|	1D labeled homogeneously-typed array
2|	DataFrame|	General 2D labeled, size-mutable tabular structure with potentially heterogeneously-typed column

## DataFrame

```python
import numpy as np
import pandas as pd
dates = pd.date_range('20190701', periods=6)
print(dates)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print(df)
""" ouput:
DatetimeIndex(['2019-07-01', '2019-07-02', '2019-07-03', '2019-07-04',
               '2019-07-05', '2019-07-06'],
              dtype='datetime64[ns]', freq='D')
                   A         B         C         D
2019-07-01  0.361654  0.645069 -0.300801 -0.723086
2019-07-02  1.688343  1.607648  1.083795  0.116933
2019-07-03  1.402047  0.916365  0.217083  0.821613
2019-07-04  0.404094 -1.063325 -2.053775 -1.509199
2019-07-05 -1.022567 -0.277598 -0.161956 -0.295696
2019-07-06 -0.314412  1.181524  1.167884  1.717774
"""

# 可以通过类似Series的对象的dict来创建
df2 = pd.DataFrame({
    'A': 3.,
    'B': pd.Timestamp('20190701'),
    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
    'D': np.array([3] * 4, dtype='int32'),
    'E': pd.Categorical(['tes', 'traim', 'test', 'train']),
    'F': 'foo',
})
print(df2)
# 读取顶行和底行的方法
df2.head()
df.tail()
# 显示索引，列
"""output2
     A          B    C  D      E    F
0  3.0 2019-07-01  1.0  3    tes  foo
1  3.0 2019-07-01  1.0  3  traim  foo
2  3.0 2019-07-01  1.0  3   test  foo
3  3.0 2019-07-01  1.0  3  train  foo

Out[5]:
A           float64
B    datetime64[ns]
C           float32
D             int32
E          category
F            object
dtype: object

In [12]: df2.index
Out[12]: Int64Index([0, 1, 2, 3], dtype='int64')

In [13]: df2.columns
Out[13]: Index(['A', 'B', 'C', 'D', 'E', 'F'], dtype='object')

# 转置数据
In [16]: df2.T
Out[16]:
                     0  ...                    3
A                    3  ...                    3
B  2019-07-01 00:00:00  ...  2019-07-01 00:00:00
C                    1  ...                    1
D                    3  ...                    3
E                  tes  ...                train
F                  foo  ...                  foo
"""
# 按轴排序
df2.sort_index(axis=1, ascending=False)
# 按值排序
df2.sort_values(by='B')

# 获取值
df2['D']

df2[0:2]

# selection by label
df2.loc[dates[0]]
df2.loc[:, ['C','B']]
"""
Out[34]:
     C          B
0  1.0 2019-07-01
1  1.0 2019-07-01
2  1.0 2019-07-01
3  1.0 2019-07-01
"""

# selection by position 横着取一行
df2.iloc[0]
"""
Out[53]:
A                      3
B    2019-07-01 00:00:00
C                      1
D                      3
E                    tes
F                    foo
Name: 0, dtype: object
"""
df2.iat[1,4]
# Out[51]: 'traim'

# set a new column automatically aligns the data by indexes
s1 = pd.Series([1,3,4,5,7,8], index=list(range(6)), dtype='int64')
"""
In [67]: df2['C'] = s1

In [68]: df2
Out[68]:
     A          B  C  D      E    F
0  3.0 2019-07-01  1  3    tes  foo
1  3.0 2019-07-01  3  3  traim  foo
2  3.0 2019-07-01  4  3   test  foo
3  3.0 2019-07-01  5  3  train  foo
"""
# applying functions to data
df2.apply(np.cumsum)
```
DataFrame.to_numpy()给出基础数据的NumPy表示。请注意，当您的DataFrame列具有不同数据类型时，这可能是一项昂贵的操作，这归结为pandas和NumPy之间的根本区别：NumPy数组对整个数组有一个dtype，而pandas DataFrames每列有一个dtype.

## Series

s = pd.Series(data, index=index)

data可以是:
- python Dict
- ndarray
- 标量值

```python
import numpy as np
import pandas as pd
s = pd.Series([1, 4, 6, 2, np.nan, 9, 5])
print(s)
```

## 合并

CONCAT, pandas提供各种工具在concat/merge操作情况下将Series和DataFrame对象组合在一起

```python
import numpy as np
import pandas as pd
df = pd.DataFrame(np.random.randint(10, 3))
print(df)
pieces = [df[:2], df[8:]]
pd.concat(pieces)
```

## join 加入

```python
import numpy as np
import pandas as pd
left = pd.DataFrame({'key': ['foo', 'boom'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'boom'], 'rval': [4, 5]})
pd.merge(left, right, on='key')
"""
Out[101]:
    key  lval  rval
0   foo     1     4
1  boom     2     5
"""
```

## 分组 groupby

```python
import numpy as np
import pandas as pd
df = pd.DataFrame({
    'sex': ['man', 'woman', 'man', 'boy', 'woman', 'woman'],
    'age': [22, 23, 20, 11, 22, 23],
    'phone': np.random.randn(6)
})
df.grouby('sex').sum()
df.groupby(['sex', 'age']).sum()
"""
In [105]: df.groupby('sex').sum()
Out[105]:
       age     phone
sex
boy     11 -0.908680
man     42 -1.378370
woman   68  1.611417

In [106]: df.groupby(['sex', 'age']).sum()
Out[106]:
              phone
sex   age
boy   11  -0.908680
man   20  -0.178224
      22  -1.200146
woman 22   1.044000
      23   0.567417

"""
```

## 生成透视表

```python
import numpy as np
import pandas as pd
df = pd.DataFrame({
    'sex': ['man', 'woman', 'man', 'boy', 'woman', 'woman'],
    'age': [22, 23, 20, 11, 22, 23],
    'name': ['111', '222', '333', '444', '555', '666'],
    'revenue': [100., 233., 75.2, 45.6, 97., 123.],
})
pd.pivot_table(df, values='revenue', index=['sex', 'age'], columns=['name'])

"""
name         111    222   333   444   555    666
sex   age
boy   11     NaN    NaN   NaN  45.6   NaN    NaN
man   20     NaN    NaN  75.2   NaN   NaN    NaN
      22   100.0    NaN   NaN   NaN   NaN    NaN
woman 22     NaN    NaN   NaN   NaN  97.0    NaN
      23     NaN  233.0   NaN   NaN   NaN  123.0

"""
```

## 获取数据

```python
# csv
df.to_csv('foo.csv')
pd.read_csv('fpp.csv')

# hdf5
df.to_hdf('foo.h5', 'df')
pd.read_hdf('foo.h5', 'df')
```

## Practice

### 众数
```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    'Name':['Alisa','Bobby','Cathrine','Madonna','Rocky','Sebastian','Jaqluine',
    'Rahul','David','Andrew','Ajay','Teresa'],
    'Math':[62,47,55,74,47,77,85,63,42,32,71,57],
    'English':[89,87,67,55,47,72,76,79,44,67,99,69],
    'Chinese':[56,86,77,45,73,62,74,89,71,67,97,68]
})
df['Math'].mode()
"""
In [2]: df['Math'].mode()
Out[2]:
0    47
dtype: int64

In [3]: df['English'].mode()
Out[3]:
0    67
dtype: int64

In [4]: df['Chinese'].mode()
Out[4]:
0     45
1     56
2     62
3     67
4     68
5     71
6     73
7     74
8     77
9     86
10    89
11    97
dtype: int64
"""
```

## 中位数

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    'Name':['Alisa','Bobby','Cathrine','Madonna','Rocky','Sebastian','Jaqluine',
    'Rahul','David','Andrew','Ajay','Teresa'],
    'Math':[62,47,55,74,47,77,85,63,42,32,71,57],
    'English':[89,87,67,55,47,72,76,79,44,67,99,69],
    'Chinese':[56,86,77,45,73,62,74,89,71,67,97,68]
})
df['Math'].median()
"""
In [5]: df['Chinese'].median()
Out[5]: 72.0
"""
```

## 平均数

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    'Name':['Alisa','Bobby','Cathrine','Madonna','Rocky','Sebastian','Jaqluine',
    'Rahul','David','Andrew','Ajay','Teresa'],
    'Math':[62,47,55,74,47,77,85,63,42,32,71,57],
    'English':[89,87,67,55,47,72,76,79,44,67,99,69],
    'Chinese':[56,86,77,45,73,62,74,89,71,67,97,68]
})
df['Math'].mean()
"""
In [6]: df['Chinese'].mean()
Out[6]: 72.08333333333333

In [7]: df['Math'].mean()
Out[7]: 59.333333333333336
"""
```

## 分位数

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    'Name':['Alisa','Bobby','Cathrine','Madonna','Rocky','Sebastian','Jaqluine',
    'Rahul','David','Andrew','Ajay','Teresa'],
    'Math':[62,47,55,74,47,77,85,63,42,32,71,57],
    'English':[89,87,67,55,47,72,76,79,44,67,99,69],
    'Chinese':[56,86,77,45,73,62,74,89,71,67,97,68]
})
"""
In [12]: df['Math'].quantile(q=0.2)
Out[12]: 47.0

In [13]: df['Math'].quantile(q=0.5)
Out[13]: 59.5

In [14]: df['Math'].quantile(q=0.6)
Out[14]: 62.599999999999994

In [15]: df['Math'].quantile(q=0.7)
Out[15]: 68.6
"""
```

## 方差

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    'Name':['Alisa','Bobby','Cathrine','Madonna','Rocky','Sebastian','Jaqluine',
    'Rahul','David','Andrew','Ajay','Teresa'],
    'Math':[62,47,55,74,47,77,85,63,42,32,71,57],
    'English':[89,87,67,55,47,72,76,79,44,67,99,69],
    'Chinese':[56,86,77,45,73,62,74,89,71,67,97,68]
})
np.var(df['English'])
"""
In [16]: np.var(df['English'])
Out[16]: 254.24305555555557

In [17]: np.var(df['Math'])
Out[17]: 226.55555555555551
"""
```

## 标准差

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    'Name':['Alisa','Bobby','Cathrine','Madonna','Rocky','Sebastian','Jaqluine',
    'Rahul','David','Andrew','Ajay','Teresa'],
    'Math':[62,47,55,74,47,77,85,63,42,32,71,57],
    'English':[89,87,67,55,47,72,76,79,44,67,99,69],
    'Chinese':[56,86,77,45,73,62,74,89,71,67,97,68]
})
np.std(df['English'])
"""
In [18]: np.std(df['English'])
Out[18]: 15.94500095815474
"""
```

参考:

https://www.pypandas.cn

https://dev.pandas.io/

https://dev.pandas.io/getting_started/comparison/comparison_with_sql.html

https://www.geeksforgeeks.org/python-pandas-series-quantile/

http://www.datasciencemadesimple.com/variance-function-python-pandas-dataframe-row-column-wise-variance/


https://stackoverflow.com/questions/38884466/how-to-select-a-range-of-values-in-a-pandas-dataframe-column
