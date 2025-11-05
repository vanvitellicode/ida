import os
import json
import pandas as pd

def extract_thread_data(thread_path, label):
    # Annotation
    ann_file = os.path.join(thread_path, "annotation.json")
    ann = json.load(open(ann_file)) if os.path.exists(ann_file) else {}

    rows = []

    # Source tweet
    src_dir = os.path.join(thread_path, "source-tweets")
    for fname in os.listdir(src_dir):
        fpath = os.path.join(src_dir, fname)
        with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
            try:
                tweet = json.load(f)
            except json.JSONDecodeError:
                print(f"[ERRORE] JSON malformato in source-tweet: {fpath} – saltato.")
                continue
            rows.append({
                'thread_id': os.path.basename(thread_path),
                'label': label,
                **ann,
                'tweet_type': 'source',
                'tweet_id': tweet.get('id_str'),
                'user_id': tweet.get('user', {}).get('id_str'),
                'screen_name': tweet.get('user', {}).get('screen_name'),
                'created_at': tweet.get('created_at'),
                'in_reply_to_status_id': tweet.get('in_reply_to_status_id'),
                'text': tweet.get('text', '').replace('\n', ' '),
                'lang': tweet.get('lang'),
                'retweet_count': tweet.get('retweet_count'),
                'favorite_count': tweet.get('favorite_count'),
                'media_url': tweet.get('entities', {}).get('media', [{}])[0].get('media_url'),
                'media_type': tweet.get('entities', {}).get('media', [{}])[0].get('type')
            })

    # Reaction tweets
    react_dir = os.path.join(thread_path, "reactions")
    for fname in os.listdir(react_dir):
        fpath = os.path.join(react_dir, fname)
        with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
            try:
                tweet = json.load(f)
            except json.JSONDecodeError:
                print(f"[ERRORE] JSON malformato in reaction: {fpath} – saltato.")
                continue
            rows.append({
                'thread_id': os.path.basename(thread_path),
                'label': label,
                **ann,
                'tweet_type': 'reaction',
                'tweet_id': tweet.get('id_str'),
                'user_id': tweet.get('user', {}).get('id_str'),
                'screen_name': tweet.get('user', {}).get('screen_name'),
                'created_at': tweet.get('created_at'),
                'in_reply_to_status_id': tweet.get('in_reply_to_status_id'),
                'text': tweet.get('text', '').replace('\n', ' '),
                'lang': tweet.get('lang'),
                'retweet_count': tweet.get('retweet_count'),
                'favorite_count': tweet.get('favorite_count'),
                'media_url': tweet.get('entities', {}).get('media', [{}])[0].get('media_url'),
                'media_type': tweet.get('entities', {}).get('media', [{}])[0].get('type')
            })

    return rows

def process_dataset(event_path, event_name):
    for label in ['rumours', 'non-rumours']:
        label_path = os.path.join(event_path, label)
        all_rows = []
        for thread_id in os.listdir(label_path):
            thread_path = os.path.join(label_path, thread_id)
            if os.path.isdir(thread_path):
                all_rows.extend(extract_thread_data(thread_path, label))
        df = pd.DataFrame(all_rows)
        out_file = f"{event_name}_{label}.csv"
        df.to_csv(out_file, index=False, encoding='utf-8-sig')
        print(f"{out_file}: {len(df)} righe")

# ESEMPIO USO
if __name__ == "__main__":
    base_path = ""  # <-- Inserisci qui il path corretto
    for event in ["putinmissing-all-rnr-threads", "charliehebdo-all-rnr-threads"]:
        process_dataset(os.path.join(base_path, event), event)