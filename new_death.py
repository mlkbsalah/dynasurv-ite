date_cols = ['line_start_date', 'line_end_date', 'death_date']

df[date_cols] = (
    df[date_cols]
    .astype(str)                 # ensure string
    .apply(lambda s: s.str.split().str[0])  # keep only YYYY-MM-DD
    .apply(pd.to_datetime, errors='coerce')
)

df['Y_death_inline'] = (df['Y_death']==True) & ((df['death_date'] > df['line_start_date']) & (df['death_date'] < df['line_end_date']))