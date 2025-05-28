#!/usr/bin/env bash
N=$(python3 - <<EOF
import config
print(config.N_PARTITIONS)
EOF
)

for (( i=1; i<=N; i++ )); do
  echo "<require-service <daemon <worker $i>>>"
done

echo "<require-service <daemon <epoch-manager>>>"
echo "<require-service <daemon <jobq>>>"
