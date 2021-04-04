# NOTE: any questions, check user guide is ok.
import pandas as pd 

################### read excel #####################
# for the certain sheet name.
fr = pd.read_excel('data.xlsx',sheet_name='people')
# for the DataFrame structures, we can use.
# get the first 5 rows 
data = df.head()
# get all data
data = df.values
# get all column name
data = df.columns.values
data = df.index.values

################### write into the excel #####################

# need to know the dimensions are 'index' and 'column'
pd.DataFrame(np.random.random((4,4)),
                     index=['exp1','exp2','exp3','exp4'],
                     columns=['jan2015','Fab2015','Mar2015','Apr2005']) 
print(frame)
frame.to_excel("data2.xlsx") 

################### for commmon xlxs #####################
# for the certain column, compare it with others.
data['gender'][data['gender'] == 'male'] = 0
data['gender'][data['gender'] == 'female'] = 1
# we can get rows and columns with index number.
df1.iloc[:, 1:3]
# save
pd.DataFrame(data).to_excel(path, sheet_name='Sheet1', index=False, header=True)

# new column
data['列名称'] = None
# 值为None（默认值）时，只有列名，没有数据
data['profession'] = None

# get new index
# 这里的N为excel自动给行加的序号
data.loc[N] = [值1， 值2， ...]
# 比如我在第5行新增
data.loc[5] = ['James', 32, 'male']

# delete
# delete gender column
data = data.drop('gender', axis=1)
# delete the 2nd and 3rd rows
data = data.drop([2, 3], axis=0)

