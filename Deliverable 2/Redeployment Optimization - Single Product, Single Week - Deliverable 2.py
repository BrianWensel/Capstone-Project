
# coding: utf-8

# # Importing Packages and Datasets

# In[1]:


import pandas as pd
from pulp import *
import timeit


# In[2]:


#Timer used to calculate runtime of model
#start = timeit.default_timer()


# In[3]:


shipping_data = pd.read_csv('C:/Users/wenselbp/Desktop/Redeployment optimization/Python Files/full Shipping Lane DF.csv', index_col=['Receiving_Plant','Shipping_Plant', 'Mode'])


# In[4]:


shipping_df = shipping_data.drop(['Lane Code','Lane'], axis = 1)


# In[5]:


shipping_df


# In[6]:


material_data = pd.read_csv('C:/Users/wenselbp/Desktop/Redeployment optimization/Python Files/Material DF.csv', index_col=['Node'])


# In[7]:


material_data


# In[8]:


material_df = material_data.drop(['Material','Material Text'], axis = 1)


# In[9]:


material_df


# # Variables

# In[10]:


shipment = pulp.LpVariable.dicts("shipment",
                                     ((Receiving_Plant, Shipping_Plant, Mode) for Receiving_Plant, Shipping_Plant, Mode in shipping_df.index),
                                    lowBound=0,
                                    cat='Integer')


# In[11]:


shortage_cost = pulp.LpVariable.dicts("shortage_cost",
                                     ((Receiving_Plant, Shipping_Plant, Mode) for Receiving_Plant, Shipping_Plant, Mode in shipping_df.index),
                                    lowBound=0,
                                    cat='Integer')


# In[12]:


type(shortage_cost)


# In[13]:


#Shortage Cost Dict
for Receiving_Plant, Shipping_Plant, Mode in shipping_df.index:
    if (material_df.loc[(Receiving_Plant), 'Need_in_days'] - shipping_df.loc[(Receiving_Plant, Shipping_Plant, Mode),'Transit Time']) <= 0:
        shortage_cost[Receiving_Plant, Shipping_Plant, Mode] = abs((material_df.loc[(Receiving_Plant), 'Need_in_days'] - shipping_df.loc[(Receiving_Plant, Shipping_Plant, Mode),'Transit Time']) * material_df.loc[(Receiving_Plant), 'daily_shortage_cost'])


# In[14]:


shortage_cost[Receiving_Plant, Shipping_Plant, Mode]


# # LP Problem

# In[15]:


model = pulp.LpProblem("cost_min_problem", LpMinimize)


# In[16]:


model += pulp.lpSum(
    [shipment[Receiving_Plant, Shipping_Plant, Mode] * shipping_df.loc[(Receiving_Plant, Shipping_Plant, Mode), 'Average Rate']
for Receiving_Plant, Shipping_Plant, Mode in shipping_df.index]   
    
    + [shipment[Receiving_Plant, Shipping_Plant, Mode] * shortage_cost[(Receiving_Plant, Shipping_Plant, Mode)]        
for Receiving_Plant, Shipping_Plant, Mode in shipping_df.index]
)      

# Minmize(Cost of Transportaion + Cost of Not Delivering the Product)


# # Constraints

# In[17]:


#Shipment from any lane must be equal to demand
for node in material_df.index:
    model += shipment[(node, 1020,'IM')] + shipment[(node, 1100,'IM')] + shipment[(node, 1415,'IM')] + shipment[(node, 1645,'IM')] + shipment[(node, 1020,'TL')] + shipment[(node, 1100,'TL')] + shipment[(node, 1415,'TL')] + shipment[(node, 1645,'TL')] == material_df.loc[node, 'Projected_Shortage_Qty']


# In[18]:


#Shipment from any lane must be less or eqaul to the available supply at that node
for node in material_df.index:
    model += shipment[(1020, node,'IM')] <= material_df.loc[node, 'Available Supply']
    model += shipment[(1100, node,'IM')] <= material_df.loc[node, 'Available Supply']
    model += shipment[(1415, node,'IM')] <= material_df.loc[node, 'Available Supply']
    model += shipment[(1645, node,'IM')] <= material_df.loc[node, 'Available Supply']
    model += shipment[(1020, node,'TL')] <= material_df.loc[node, 'Available Supply']
    model += shipment[(1100, node,'TL')] <= material_df.loc[node, 'Available Supply']
    model += shipment[(1415, node,'TL')] <= material_df.loc[node, 'Available Supply']
    model += shipment[(1645, node,'TL')] <= material_df.loc[node, 'Available Supply']


# # Model Solution

# In[19]:


model.solve()
print(LpStatus[model.status])
#print(value(model.objective))


# In[20]:


def highlight_shipments(row):
    if row.shipment > 0:
        return ['background-color: lightgreen']
    else:
        return ['background-color: white']


# In[21]:


model_output = []
for Receiving_Plant, Shipping_Plant, Mode in shipment:
    columns = {
        'Receiving_Plant': Receiving_Plant,
        'Shipping_Plant': Shipping_Plant,
        'Mode': Mode,
        'shipment': shipment[(Receiving_Plant, Shipping_Plant, Mode)].varValue,
        'total_cost': shortage_cost[(Receiving_Plant, Shipping_Plant, Mode)] + shipping_df.loc[(Receiving_Plant, Shipping_Plant, Mode), 'Average Rate'],
        'shortage_cost': shortage_cost[(Receiving_Plant, Shipping_Plant, Mode)],
        'expected_days_late': abs(material_df.loc[(Receiving_Plant), 'Need_in_days'] - shipping_df.loc[(Receiving_Plant, Shipping_Plant, Mode),'Transit Time']),
        'shipping_lane_cost': shipping_df.loc[(Receiving_Plant, Shipping_Plant, Mode), 'Average Rate'],
        #'cost': (shipment[(Receiving_Plant, Shipping_Plant, Mode)].varValue / 38000) * shipping_df.loc[(Receiving_Plant, Shipping_Plant, Mode), 'Average Rate'],
    }
    model_output.append(columns)
output_df = pd.DataFrame.from_records(model_output).sort_values(['Receiving_Plant', 'Shipping_Plant', 'Mode'])
output_df.set_index(['Receiving_Plant', 'Shipping_Plant', 'Mode'], inplace=True)
output_df = output_df[output_df['shipping_lane_cost'] != 99999] #removes phantom transportation combinations
output_df= output_df[['shipment', 'expected_days_late', 'shipping_lane_cost', 'shortage_cost','total_cost']]

#Formatting DF
output_df2 = output_df.style.format({"cost": "${0:,.2f}", 
                          "shortage_cost": "${:20,.0f}", 
                          "shipping_lane_cost": "${:20,.0f}",
                          "total_cost":"${:20,.0f}"})

output_df2.apply(highlight_shipments, axis=1)


# # Optimal Shipment

# In[22]:


#Single Product, Single Week Optimal Shipment
optimal_shipment = output_df[output_df['shipment'] > 0]
optimal_shipment


# In[23]:


#Stop timer
#stop = timeit.default_timer()
#print('Run time: ', stop - start, 'seconds')


# In[24]:


#Canonical Form
print(model)

