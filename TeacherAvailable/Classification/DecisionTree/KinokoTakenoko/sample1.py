import pandas as pd

data = {
    '松田の労働時間' : [160,160],
    '浅木の労働時間' : [162,175]
    }

df = pd.DataFrame(data)
print(df)

print(type(df))
print(df.shape)

df.index = ['4月','5月']
print(df)

df.columns = ['松田の労働(h)','浅木の労働(h)']
print(df)

print(df.index)
print(df.columns)

data = [
    [160,161],
    [160,175]
]

df2 = pd.DataFrame(data,index=['4月','5月'],columns=['松田の労働','浅木の労働'])
print(df2)