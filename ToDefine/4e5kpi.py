import pandas as pd

def compute_disinfo_kpis(df):
    df = df.copy()
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df = df.dropna(subset=['created_at', 'thread_id', 'user_id'])

    results = []
    for thread_id, group in df.groupby('thread_id'):
        grp = group.sort_values('created_at')
        users = grp['user_id'].nunique()
        views = len(grp)

        source_row = grp[grp['tweet_type'] == 'source']
        source_time = source_row['created_at'].min() if not source_row.empty else None
        first_re = grp[grp['tweet_type'] == 'reaction']['created_at'].min()
        last_time = grp['created_at'].max()

        time_to_spread = (first_re - source_time).total_seconds() / 60 if pd.notna(first_re) and pd.notna(source_time) else None
        lifespan = (last_time - source_time).total_seconds() / 60 if pd.notna(source_time) and pd.notna(last_time) else None

        grp['minute'] = ((grp['created_at'] - source_time).dt.total_seconds() // 60).astype(int)
        grow_rate = grp['minute'].value_counts().max() if not grp['minute'].empty else 0

        results.append({
            'thread_id': thread_id,
            'source_created_at': source_time,  # ⬅️ questa è la data del tweet originario
            'unique_reach': users,
            'views': views,
            'time_to_initial_spread_min': round(time_to_spread, 2) if time_to_spread else None,
            'content_lifespan_min': round(lifespan, 2) if lifespan else None,
            'grow_rate_tweets_per_min': int(grow_rate)
        })

    return pd.DataFrame(results)

def process_event(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    kpis = compute_disinfo_kpis(df)
    kpis.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✅ KPI salvati in: {output_csv} ({len(kpis)} thread)")

if __name__ == "__main__":
    files = [
        ("charliehebdo-all-rnr-threads_rumours.csv", "kpi_charliehebdo.csv"),
        ("putinmissing-all-rnr-threads_rumours.csv", "kpi_putinmissing.csv")
    ]

    for inp, out in files:
        process_event(inp, out)