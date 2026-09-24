# Simple Bitcoin Forensics Demo Dataset

Upload these SIX files to the prediction API:
1. wallets.csv
2. transactions.csv
3. transaction_inputs.csv
4. transaction_outputs.csv
5. network_observations.csv
6. ip_metadata.csv

Size:
- 50 wallets
- 12 synthetic entities
- 15 IPs
- 196 transactions
- normalized transaction input/output rows
- network observations for each transaction

Anomaly coverage:
- W047: ransomware-like collection + rapid downstream movement
- W048/W049: short-gap layering chain
- W050: mixer-like many-in/many-out hub

Only 4/50 wallets (8%) are labeled suspicious in demo_ground_truth.csv.

demo_ground_truth.csv is evaluation-only and MUST NOT be uploaded to the prediction API.
IP addresses are synthetic/private-form placeholders and are not real attribution data.
