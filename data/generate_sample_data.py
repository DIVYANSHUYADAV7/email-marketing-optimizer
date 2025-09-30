# data/generate_sample_data.py
import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

n = 2000
np.random.seed(42)

# demographics & synthetic behavior
ages = np.random.randint(18, 70, size=n)
genders = np.random.choice(['M', 'F', 'Other'], size=n, p=[0.48, 0.48, 0.04])
locations = np.random.choice(['Delhi', 'Mumbai', 'Bengaluru', 'Kolkata','Other'], size=n)
past_opens = np.random.poisson(3, size=n)
past_clicks = np.random.binomial(past_opens, 0.2)
avg_session_minutes = np.clip(np.random.normal(5, 2, size=n), 0.5, 60)

# label: whether user opened & clicked a test email (synthetic function)
prob_open = 0.15 + 0.02*(past_opens>2) + 0.01*(avg_session_minutes>4) + 0.05*(ages<30)
prob_click = 0.05 + 0.04*(past_clicks>0)
opened = np.random.binomial(1, np.clip(prob_open, 0, 0.95))
clicked = np.where(opened==1, np.random.binomial(1, np.clip(prob_click,0,0.9)), 0)

df = pd.DataFrame({
    'email': [f'user{ix}@example.com' for ix in range(n)],
    'age': ages,
    'gender': genders,
    'location': locations,
    'past_opens': past_opens,
    'past_clicks': past_clicks,
    'avg_session_minutes': avg_session_minutes.round(2),
    'opened': opened,
    'clicked': clicked
})

df.to_csv(OUT / 'sample_contacts.csv', index=False)
print("Wrote:", OUT / 'sample_contacts.csv')
