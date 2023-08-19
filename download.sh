#!/bin/bash

if [[ -z $1 ]]; then
    echo "Missing training name"
    echo ""
    echo "Usage:"
    echo "  download-log.sh <training name>"
else
    cd $DR_DIR/data/logs
    rm *.log

    for name in `docker ps --format "{{.Names}}"`; do
            docker logs ${name} >& ${name}.log
    done

    #zip robomaker_log deepracer-0_robomaker.1.*.log
    #aws s3 cp robomaker_log.zip  s3://$DR_LOGS_COPY_S3_BUCKET/
    cp deepracer-0_robomaker.1.*.log /mnt/c/Users/Jacky/Downloads/deepracer-log/deepracer-0_robomaker.1.$1.log
    META_FILE=/mnt/c/Users/Jacky/Downloads/deepracer-log/deepracer-0_robomaker.1.$1.log.meta.json
    if [ -f "$META_FILE" ]; then
            rm /mnt/c/Users/Jacky/Downloads/deepracer-log/deepracer-0_robomaker.1.$1.log.meta.json
    fi
fi
