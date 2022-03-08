#!/bin/bash

num_proc=${1}
shift

PRO_TMP_DIR=/tmp/pro_metrics/
rm -rf $PRO_TMP_DIR
mkdir -p $PRO_TMP_DIR
export PROMETHEUS_MULTIPROC_DIR=$PRO_TMP_DIR

python -m torch.distributed.run --nproc_per_node $num_proc $@
