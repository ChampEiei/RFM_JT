#%%
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel(r'raw_data\Well_know_customer full july.xlsx')
df.columns
import plotly.express as px
import pandas as pd

# Example df (replace with your dataframe)
# df = pd.read_csv("your_file.csv")

# Calculate zoom limits (exclude extreme outliers)
x_min, x_max = df['UNIT_WEIGHT'].quantile([0.01, 0.95])
y_min, y_max = df['UNIT_REVENUE'].quantile([0.01, 0.95])

# Median lines for quadrant
x_median = df['UNIT_WEIGHT'].median()
y_median = df['UNIT_REVENUE'].median()

fig = px.scatter(
    df,
    x="UNIT_WEIGHT",
    y="UNIT_REVENUE",   # bubble size (optional)
    color="TOTAL_REVENUE", # bubble color (optional)
    title="Quadrant Analysis (Zoomed on Majority)"
)

# Add quadrant lines
fig.add_vline(x=x_median, line_dash="dash", line_color="red")
fig.add_hline(y=y_median, line_dash="dash", line_color="red")

# Zoom in on majority of points
fig.update_xaxes(range=[x_min, x_max])
fig.update_yaxes(range=[y_min, y_max])

fig.update_traces(textposition="top center")
fig.show()

