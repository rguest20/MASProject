import pandas as pd
df = pd.read_csv("cultural_log.csv")

print(df.describe())
# output to file
df.to_html("cultural_log_summary.html")
df.plot(x="generation", y=["mean_alignment", "mean_fitness"], figsize=(10,5))