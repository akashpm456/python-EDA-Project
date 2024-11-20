#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis on NYC Airbnb 2019 dataset 
# 
# 
# 
# ## Introduction 
# 
# ### The data from this analysis is from Kaggle New York City Airhub Open Data. The data describes the listing activity and metrics in NYC , Ny for 2012includes information such as the location of the listing properties , the neighborhood of the properties , room type, price, minimum rights required review and availability of the listing/
# 
# ### The Purpose of this analysis is to perform, exploratory data analysis as well as data visualization to understand how different fators influence the listing properties on Airbnb and ultimately  to make predicstions on the availability of the listing properties.
# 
# ### The following questioms will be answered on the course of this analysis.
# 
# * Where are the most of the properties listed smf where is the busiest areas?
# * what type of rooms are most popular? 
# * How different area/neighborhood affect the listing property price and demands?
# * What are the most important factors when customer choose an airbnb property 
#   - Price 
#   - Location
#   - Room Type 
#   - Customer Review
#   
#   ## Data loading and Processing 
#   
#   ### We start the analysys by importing necessary libraries and loading the data . The libraries used in this analysis are 
#   
#   #### - Pandas
#   #### - Numpy
#   #### - Matplotlib
#   #### - Seaborn 
#   #### - Sklearn 
#   #### - statsmodels
#   
#    
#   
#   
# 

# ![](https://assets.ltkcontent.com/images/92435/longitude-versus-latitude-earth_7abbbb2796.jpg) 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_csv(r"C:\Users\HP\Downloads\AB_NYC_2019.csv")


# In[4]:


df


# In[5]:


# Display a concise summary of the DataFrame, including the data types and non-null counts for each column
df.info()


# In[6]:


# Check the number of missing values in each column of the DataFrame
df.isnull().sum()


# In[7]:


# Retrieve the unique values present in the 'name' column of the DataFrame
df.name.unique()


# In[11]:


# Group the DataFrame by the 'price' column and count the number of occurrences of 'latitude' for each price value
df.groupby('price').latitude.count()


# In[12]:


# Set 'reviews_per_month' and 'last_review' to 0 for rows where 'number_of_reviews' is 0
df.loc[df.number_of_reviews==0, 'reviews_per_month'] = 0
df.loc[df.number_of_reviews==0, 'last_review'] = 0


# In[13]:


df


# In[14]:


# Filter the DataFrame to keep only rows where 'host_id' and 'host_name' are not null
df = df[pd.notnull(df['host_id'])]
df = df[pd.notnull(df['host_name'])]


# In[15]:


df


# In[16]:


# Sort the DataFrame based on the values in the 'latitude' column in ascending order
df.sort_values(by=['latitude'])


# In[17]:


# Sort the DataFrame based on the values in the 'longitude' column in ascending order
df.sort_values(by=['longitude'])


# In[18]:


# Sort the DataFrame based on the values in the 'price' column in ascending order
df.sort_values(by=['price'])


# In[19]:


# Calculate the mean value of the 'price' column in the DataFrame
np.mean(df.price)


# In[21]:


# Install the matplotlib library for data visualization
pip install matplotlib

 



# In[22]:


import matplotlib.pyplot as plt
# Create a histogram of the 'price' column with 50 bins
plt.hist(df.price,bins=50)


# In[18]:


# Count the number of rows where the 'price' column is greater than 2000
len(df[df.price > 2000])


# In[23]:


# Filter the DataFrame to include only rows where the 'price' column is less than 2000
df=df[df.price < 2000]


# In[24]:


# Create a histogram of the 'price' column with 50 bins after filtering
plt.hist(df['price'], bins=50, edgecolor='black')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Histogram of Prices (Filtered)')
plt.show()
plt.hist(df.price,bins=50)


# In[21]:


# Count the number of rows where the 'price' column is greater than 1000
len(df[df.price > 1000])


# In[25]:


# Filter the DataFrame to include only rows where the 'price' column is less than or equal to 1000
df = df[df.price <=1000]


# In[26]:


# Create a histogram of the 'price' column with 50 bins after further filtering
plt.hist(df['price'], bins=50, edgecolor='black')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Histogram of Prices (Filtered to <= 1000)')
plt.show()
plt.hist(df.price, bins=50)


# In[27]:


# Sort the DataFrame based on the values in the 'minimum_nights' column in ascending order
df.sort_values(by=['minimum_nights'])


# In[28]:


import matplotlib.pyplot as plt

# Create a histogram of the 'minimum_nights' column with 50 bins
plt.hist(df['minimum_nights'], bins=50, edgecolor='black')
plt.xlabel('Minimum Nights')
plt.ylabel('Frequency')
plt.title('Histogram of Minimum Nights')
plt.show()
plt.hist(df.minimum_nights,bins=50)


# In[29]:


# Count the number of rows where the 'minimum_nights' column is greater than 200
len(df[df.minimum_nights > 200])


# In[30]:


# Filter the DataFrame to include only rows where the 'minimum_nights' column is less than 200
df = df[df.minimum_nights < 200]


# In[31]:


import matplotlib.pyplot as plt

# Create a histogram of the 'minimum_nights' column with 50 bins after filtering
plt.hist(df['minimum_nights'], bins=50, edgecolor='black')
plt.xlabel('Minimum Nights')
plt.ylabel('Frequency')
plt.title('Histogram of Minimum Nights (Filtered)')
plt.show()
plt.hist(df.minimum_nights, bins=50)


# In[32]:


# Count the number of rows where the 'minimum_nights' column is greater than 100
len(df[df.minimum_nights > 100])


# In[33]:


# Sort the DataFrame based on the values in the 'number_of_reviews' column in ascending order
df.sort_values(by=['number_of_reviews'])


# In[34]:


import matplotlib.pyplot as plt

# Create a histogram of the 'number_of_reviews' column with 50 bins
plt.hist(df['number_of_reviews'], bins=50, edgecolor='black')
plt.xlabel('Number of Reviews')
plt.ylabel('Frequency')
plt.title('Histogram of Number of Reviews')
plt.show()
plt.hist(df.number_of_reviews, bins=50)


# In[35]:


# Count the number of listings that have received more than 300 reviews
len(df[df.number_of_reviews > 300])


# In[36]:


# Count the number of listings that have received more than 400 reviews
len(df[df.number_of_reviews > 400])


# In[37]:


# Filter the DataFrame to include only listings with 400 or fewer reviews
df=df[df.number_of_reviews <=400]


# In[38]:


# Sort the DataFrame by the number of listings each host has, in ascending order
df.sort_values(by=['calculated_host_listings_count'])


# In[39]:


# Create a histogram to visualize the distribution of the number of listings per host
plt.hist(df.calculated_host_listings_count, bins=50)


# In[40]:


# Sort the DataFrame by the availability of listings for the entire year, in ascending order
df.sort_values(by=['availability_365'])


# In[41]:


# Create a histogram to visualize the distribution of listings' availability throughout the year
plt.hist(df.availability_365, bins=50)


# In[42]:


len(df)


# 1.which neighbourhood_group is the biggest one?
# 
# 

# In[43]:


a=df.groupby(by=['neighbourhood_group']).neighbourhood_group.count()
a=a.sort_values(ascending=False)
print(a)


# 2.which neighbourhood_group is the most expensive?
# 

# In[44]:


a=df.groupby(by=['neighbourhood_group']).price.mean()
a=a.sort_values(ascending=False)
print(a)


# 3.which neighbourhood_group has the most possibility to available in year?

# In[45]:


a=df.groupby(by=['neighbourhood_group']).availability_365.sum()
a=a.sort_values(ascending=False)
print(a)


# 3.which neighbourhood_group has the most possibility to available in year?

# In[46]:


a=df.groupby(by=['neighbourhood']).availability_365.sum()
a=a.sort_values(ascending=False)
print(a)


# 4.which neughbourhood_group has the best hosts to stay for a few nights

# In[47]:


a=df.groupby(by=['neighbourhood_group']).minimum_nights.mean()
a=a.sort_values(ascending=False)
print(a)


# 5.which host_name is the most popular hosts between customers?

# In[48]:


a=df.groupby(by=['host_name']).calculated_host_listings_count.max()
a=a.sort_values(ascending=False)
print(a)


# # data = pd.read_csv(r"C:\Users\HP\Downloads\AB_NYC_2019.csv")
# data

# In[51]:


data.head(10)


# In[52]:


data.tail()


# In[53]:


data.info()


# In[54]:


data.describe()


# In[55]:


data.isna().sum()


# what are the top 10 host iDS with the highest number of bookings?

# In[56]:


df['host_id'].value_counts().iloc[:10]


# q1.what are the top 10 host iDs with the highest number of bookings?
# 

# In[61]:


df['host_id'].value_counts().iloc[:10]


# In[59]:


get_ipython().system('pip install matplotlib')



# In[60]:


# Visualizing top 10 host IDs with the highest number of bookings
top_10_host_IDs = df['host_id'].value_counts().iloc[:10]
# Plotting
plt.figure(figsize=(12, 6)) 
ax = top_10_host_IDs .plot(kind='bar', color='grey')
for bars in ax.containers:
   ax.bar_label(bars)
plt.title('Top 10 Host IDs with the Highest Number of Bookings')
plt.xlabel('Host IDs')
plt.ylabel('Number of Bookings')
plt.xticks(rotation=45, ha='right')  
plt.grid(axis='y')  
plt.tight_layout()
plt.show()





# In[62]:


# Percentage of bookings for Top 10 Host ID's 
hostidPer = (df['host_id'].value_counts().iloc[:10].sort_values(ascending=False)/len(df))*100
hostidPer


# Observation¶
# The host named Michael has 417 bookings attributed to him, accounting for 85% of the total bookings.
# The person with the Name David stands at the second position with the total bookings of 403.
# 

# In[63]:


df.head()


# Question 5: Which Neighbourhood group has the highest number of bookings?
# 

# In[64]:


# Getting value counts
df['neighbourhood_group'].value_counts()


# In[65]:


# Visualizing neighbourhood groups with the highest number of bookings
neightop = df['neighbourhood_group'].value_counts()
# Plotting
plt.figure(figsize=(12, 6)) 
ax = neightop.plot(kind='bar', color='skyblue')
for bars in ax.containers:
    ax.bar_label(bars)
plt.title('Neighbourhood Groups with the Highest Number of Bookings')
plt.xlabel('Neighbourhood Group')
plt.ylabel('Number of Bookings')
plt.xticks(rotation=45, ha='right')  
plt.grid(axis='y')  
plt.tight_layout()
plt.show()


# In[66]:


# Percentage of bookings for Neighbourhood groups
neighbourhood_grpPer = (df['neighbourhood_group'].value_counts().sort_values(ascending=False)/len(df))*100
neighbourhood_grpPer


# In[67]:


# Visualizing using pie chart
df['neighbourhood_group'].value_counts().plot(kind = 'pie', figsize = (8,8), fontsize = 15, autopct = '%1.1f%%')
plt.title("Neighbourhood Group", fontsize = 15)


# observations
# 
# * An observation reveals that among all the neighborhood groups, the Manhattan group has the highest number of bookings, totaling 21,661, which constitutes 44.3% of all bookings across all groups.
# 
# * Brooklyn ranks as the second-highest neighborhood group with a total of 20,104 bookings, covering 41% of all bookings.
# 
# * Staten Island is the neighbourhood group with the least number of bookings which constitutes only 0.76% of all the bookings

# Question 6: Which Neighbourhood Group has the maximum price range for rooms?
# 

# In[68]:


plt.figure(figsize = (15,6))
sns.boxplot(x=df['price'])
plt.show()


# In[69]:


# Generate descriptive statistics for the 'price' column, including count, mean, standard deviation, min, max, and percentiles
df['price'].describe()


# In[70]:


# Filter the DataFrame to include only listings with a price less than 334 and store the result in a new DataFrame
df_new = df[df['price'] < 334 ]
df_new.head()


# In[71]:


# Group the DataFrame by 'neighbourhood_group' and generate descriptive statistics for the 'price' column
# Transpose the result, reset the index, and store it in a new DataFrame
df.groupby(['neighbourhood_group'])['price'].describe().T.reset_index()


# Observation
# 
# The price range for Bronx Neighbourhood group is in the range 0 and 2500
# 
# The price range for Brooklyn Neighbourhood group is in the range 0 and 10000
# 
# The price range for Manhattan Neighbourhood group is in the range 0 and 10000
# 
# The price range for Queens Neighbourhood group is in the range 10 and 10000
# 
# The price range for Staten Island Neighbourhood group is in the range 13 and 5000
# 

# In[72]:


plt.figure(figsize = (15,6))
sns.violinplot(data = df_new, x = df_new['neighbourhood_group'], y = df_new['price'])
plt.title('Density and distribution of prices for each neighberhood_group', fontsize = 15)
plt.grid()


# In[73]:


plt.figure(figsize = (16,15))

plt.subplot(3,2,1)
n1 = df_new[df_new['neighbourhood_group'] == 'Brooklyn']
sns.distplot(x = n1['price'])
plt.title("Brooklyn", fontsize = 15)

plt.subplot(3,2,2)
n2 = df_new[df_new['neighbourhood_group'] == 'Manhattan']
sns.distplot(x = n2['price'])
plt.title("Manhattan", fontsize = 15)

plt.subplot(3,2,3)
n3 = df_new[df_new['neighbourhood_group'] == 'Queens']
sns.distplot(x = n3['price'])
plt.title("Queens", fontsize = 15)

plt.subplot(3,2,4)
n4 = df_new[df_new['neighbourhood_group'] == 'Staten Island']
sns.distplot(x = n4['price'])
plt.title("Staten Island", fontsize = 15)

plt.subplot(3,2,5)
n5 = df_new[df_new['neighbourhood_group'] == 'Bronx']
sns.distplot(x = n5['price'])
plt.title("Bronx", fontsize = 15)


# Observation
# 
# we can observe that Manhattan has the highest range of prices for the listings with 150 price as median observation, followed by Brooklyn with 90 per night.
# 
# Queens and Staten Island appear to have very similar distributions, Bronx is the cheapest of them all.

# Question 7: What are the Top 10 Neighbourhoods having highest number of bookings?¶

# In[74]:


df['neighbourhood'].value_counts().iloc[:10]


# In[75]:


# Visualizing the Top 10 Neighbourhoods with the highest number of bookings
plt.figure(figsize=(12, 6))
ax = sns.barplot(x=df['neighbourhood'].value_counts().iloc[:10].keys(), y=df['neighbourhood'].value_counts().iloc[:10], palette="autumn") 
for bars in ax.containers:
    ax.bar_label(bars)
plt.title("Top 10 Neighbourhoods with the highest number of bookings", fontsize=16)
plt.xlabel("Neighbourhood", fontsize=12)
plt.ylabel("Number of Bookings", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)  
plt.tight_layout()  
plt.show()


# Question 8: Which room type has highest number of bookings?¶
# 

# In[76]:


# Getting the value counts
df['room_type'].value_counts()


# In[77]:


# Visualizing using Count Plot
ax = sns.countplot(x = 'room_type',data = df, palette="Set2")

for bars in ax.containers:
    ax.bar_label(bars)


# conclusion: 
# Throught this analysis,we have a better idea on the key factors that influences the demand of an airbnb listing property.Tourists/customers prefer location close to downtown, lower price and entire room which offers them more privacy when toring the city.These can all be taken into consideration for airbnb hosts when posting their properties online.
# 

# 
