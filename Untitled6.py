#!/usr/bin/env python
# coding: utf-8

# In[1]:



# importing all the necessary libraries
import numpy as np 
import pandas as pd 
import json 
import requests 
from pandas.io.json import json_normalize 
import matplotlib.cm as cm
import matplotlib.colors as colors 
from sklearn.cluster import KMeans 
from bs4 import BeautifulSoup 
from geopy.geocoders import Nominatim  
import folium 
from pandas.io.json import json_normalize 
print("Libraries imported.")


# In[2]:


get_ipython().system('pip install geopy')
get_ipython().system('pip install folium')


# In[3]:


# fetching Wikipedia page data using requests
data = requests.get('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M').text


postalCodeList = []
boroughList = []
neighborhoodList = []

soup = BeautifulSoup(data, 'html.parser')


# In[4]:


# looking for any tables on the Wikipedia page
soup.find('table').find_all('tr')

soup.find('table').find_all('tr')

for row in soup.find('table').find_all('tr'):
    cells = row.find_all('td')
    
# looking for the necessary information we need to create our dataframe
for row in soup.find('table').find_all('tr'):
    cells = row.find_all('td')
    if(len(cells) > 0):
        postalCodeList.append(cells[0].text)
        boroughList.append(cells[1].text)
        neighborhoodList.append(cells[2].text.rstrip('\n'))
# creating the dataframe with the columns we want using pandas
toronto_df = pd.DataFrame({"PostalCode": postalCodeList,
                           "Borough": boroughList,
                           "Neighborhood": neighborhoodList})

toronto_df.head()


# In[5]:


# cleaning up the column values and taking off the unnessary "\n"
toronto_df['Neighborhood'] = toronto_df['Neighborhood'].str.replace("\n","")
toronto_df['Borough'] = toronto_df['Borough'].str.replace("\n","")
toronto_df['PostalCode'] = toronto_df['PostalCode'].str.replace("\n","")
toronto_df.head()


# In[6]:


# ignoring the cells that have Borough =! 'Not Assigned'
toronto_df_drop = toronto_df[toronto_df.Borough != "Not assigned"].reset_index(drop=True)
toronto_df_drop.head()


# In[7]:



# grouping the neighborhoods with the same Borough on the same row
toronto_df_grouped = toronto_df_drop.groupby(["PostalCode", "Borough"], as_index=False).agg(lambda x: ", ".join(x))
toronto_df_grouped.head()


# In[8]:


# making Neighborhood == 'Not Assigned' to same Borough name
for index, row in toronto_df_grouped.iterrows():
    if row["Neighborhood"] == "Not assigned":
        row["Neighborhood"] = row["Borough"]
        
toronto_df_grouped.head()


# In[9]:



# taking a look the the unique values for PostalCode in the dataset
toronto_df_grouped['PostalCode'].unique()


# In[10]:


# creating the dataset we need with the given conditions for the assignment
column_names = ["PostalCode", "Borough", "Neighborhood"]
test_df = pd.DataFrame(columns=column_names)

test_list = ["M5G", "M2H", "M4B", "M1J", "M4G", "M4M", "M1R", "M9V", "M9L", "M5V", "M1B", "M5A"]

for postcode in test_list:
    test_df = test_df.append(toronto_df_grouped[toronto_df_grouped["PostalCode"]==postcode], ignore_index=True)
    
test_df


# In[11]:



print('The shape of the dataset is: {}'.format(toronto_df_grouped.shape))


# using the Geospatial_Coordinates.csv to get respective coordinates for each postal code
df_geo_coor = pd.read_csv("./Geospatial_Coordinates.csv")
df_geo_coor.head()


# In[12]:


# merging both datasets together
df_toronto = pd.merge(toronto_df_grouped, df_geo_coor, how='left', left_on = 'PostalCode', right_on = 'Postal Code')

# remove the "Postal Code" column
df_toronto.drop("Postal Code", axis=1, inplace=True)
df_toronto.head()


# ## Question 2

# In[13]:


# dataframe with necessary columns specified in the question along with the coordinates
column_names = ["PostalCode", "Borough", "Neighborhood", "Latitude", "Longitude"]
test_df = pd.DataFrame(columns=column_names)

test_list = ["M5G", "M2H", "M4B", "M1J", "M4G", "M4M", "M1R", "M9V", "M9L", "M5V", "M1B", "M5A"]

for postcode in test_list:
    test_df = test_df.append(df_toronto[df_toronto["PostalCode"]==postcode], ignore_index=True)
    
test_df


# ## Question 3

# In[14]:


# getting the latitude and longitude of Toronto
address = "Toronto, ON"

geolocator = Nominatim(user_agent="toronto_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto city are {}, {}.'.format(latitude, longitude))


# In[15]:


# creating map of Toronto using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)
map_toronto


# In[16]:


# adding in the markers for Toronto neighborhoods
for lat, lng, borough, neighborhood in zip(
        df_toronto['Latitude'], 
        df_toronto['Longitude'], 
        df_toronto['Borough'], 
        df_toronto['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  

map_toronto


# In[17]:


# making sure to only display boroughs that contain the word Toronto
df_toronto_denc = df_toronto[df_toronto['Borough'].str.contains("Toronto")].reset_index(drop=True)
df_toronto_denc.head()


# In[18]:


# display map for only those boroughs with Toronto in name
map_toronto_denc = folium.Map(location=[latitude, longitude], zoom_start=12)
for lat, lng, borough, neighborhood in zip(
        df_toronto_denc['Latitude'], 
        df_toronto_denc['Longitude'], 
        df_toronto_denc['Borough'], 
        df_toronto_denc['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto_denc)  

map_toronto_denc


# In[19]:



# specifying credentials for Foursquare
CLIENT_ID = 'TJDVJ4QORLYPBVCMEH2CK4SYUPYP4GDUUBBNQOA0E1WKLGC1'
CLIENT_SECRET = '2GXZGPDVMQPDILIJLXLNSAFZVNJ1CH0VSVOIN2HZ2B30V1WF'
VERSION = '20180605'


# In[20]:



# assigning a value to neighborhood_name with which we'll work with
neighborhood_name = df_toronto_denc.loc[3, 'Neighborhood']
print(f"The first neighborhood's name is '{neighborhood_name}'.")


# In[21]:



#  obtaining the longitude and latitude of the neighborhood we are using
neighborhood_latitude = df_toronto_denc.loc[3, 'Latitude'] # neighborhood latitude value
neighborhood_longitude = df_toronto_denc.loc[3, 'Longitude'] # neighborhood longitude value

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))


# 
# #Getting top 100 venues that are within a radius of 500 m from Studio District:

# In[22]:



# scraping the foursquare website for the information we want and obtaining the json file as results
LIMIT = 100 # limit of number of venues returned by Foursquare API
radius = 500 # define radius
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)

# get the result to a json file
results = requests.get(url).json()
print(results)


# In[23]:


# this function extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[24]:


get_ipython().system('pip install --user pandas==1.0.3')
import pandas as pd
pd.__version__


# In[25]:


# cleaning up the json file and structuring it as a pandas dataframe
from pandas.io.json import json_normalize

venues = results['response']['groups'][0]['items']
nearby_venues = pd.json_normalize(venues)

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues = nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()


# In[26]:


# function for getting the venues nearby 
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    venues_list=[]
    
    for name, lat, lng in zip(names, latitudes, longitudes):
        # print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[27]:


# function to get venues nearby for each Neighborhood
toronto_denc_venues = getNearbyVenues(names=df_toronto_denc['Neighborhood'],
                                   latitudes=df_toronto_denc['Latitude'],
                                   longitudes=df_toronto_denc['Longitude']
                                  )


# In[28]:



# display the first 5 
toronto_denc_venues.head()

# group by neighborhood
toronto_denc_venues.groupby('Neighborhood').count()


# In[29]:


# print the number of distinct categories
print('{} uniques categories.'.format(len(toronto_denc_venues['Venue Category'].unique())))


# In[30]:


toronto_denc_onehot = pd.get_dummies(toronto_denc_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_denc_onehot['Neighborhood'] = toronto_denc_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [toronto_denc_onehot.columns[-1]] + list(toronto_denc_onehot.columns[:-1])
toronto_denc_onehot = toronto_denc_onehot[fixed_columns]

toronto_denc_onehot.head()


# In[31]:



# create dataframe where venues are grouped by frequency for each Neighborhood
toronto_denc_grouped = toronto_denc_onehot.groupby('Neighborhood').mean().reset_index()
toronto_denc_grouped.head()


# In[32]:


# function to retrieve 10 most common venues for each Neighborhood
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    return row_categories_sorted.index.values[0:num_top_venues]

num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_denc_grouped['Neighborhood']

for ind in np.arange(toronto_denc_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_denc_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# ## K Means - Clustering 

# In[33]:


# initiating the k-means algorithm to cluster the neighborhoods
# set number of clusters
kclusters = 5

toronto_denc_grouped_clustering = toronto_denc_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_denc_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# In[34]:


# clustering top 10 venues for each Neighborhood and creating a dataframe
# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_denc_merged = df_toronto_denc

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_denc_merged = toronto_denc_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

toronto_denc_merged.head() # check the last columns!


# In[35]:


# visualizing clusters using foliumb
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(
        toronto_denc_merged['Latitude'], 
        toronto_denc_merged['Longitude'], 
        toronto_denc_merged['Neighborhood'], 
        toronto_denc_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# ## Question 3

# In[36]:


toronto_denc_merged.loc[toronto_denc_merged['Cluster Labels'] == 0, toronto_denc_merged.columns[[1] + list(range(5, toronto_denc_merged.shape[1]))]]


# In[37]:


toronto_denc_merged.loc[toronto_denc_merged['Cluster Labels'] == 1, toronto_denc_merged.columns[[1] + list(range(5, toronto_denc_merged.shape[1]))]]

toronto_denc_merged.loc[toronto_denc_merged['Cluster Labels'] == 2, toronto_denc_merged.columns[[1] + list(range(5, toronto_denc_merged.shape[1]))]]

toronto_denc_merged.loc[toronto_denc_merged['Cluster Labels'] == 3, toronto_denc_merged.columns[[1] + list(range(5, toronto_denc_merged.shape[1]))]]

toronto_denc_merged.loc[toronto_denc_merged['Cluster Labels'] == 4, toronto_denc_merged.columns[[1] + list(range(5, toronto_denc_merged.shape[1]))]]


# In[ ]:




