#!/usr/bin/env bash
# generate_services.sh

# grab how many workers you’re running
N=$(python3 - <<EOF
import config
print(config.N_PARTITIONS)
EOF
)

# emit one require-service line per worker
for (( i=1; i<=N; i++ )); do
  echo "<require-service <daemon <worker $i>>>"
done

# then the two fixed lines
echo "<require-service <daemon <epoch-manager>>>"
echo "<require-service <daemon <jobq>>>"
