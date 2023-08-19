#!/bin/bash

cd $DR_DIR/data/logs
rm *.log *.zip

for name in `docker ps --format "{{.Names}}"`; do
        docker logs ${name} >& ${name}.log
done

#zip robomaker_log deepracer-0_robomaker.1.*.log
#aws s3 cp robomaker_log.zip  s3://$DR_LOGS_COPY_S3_BUCKET/
cp deepracer-0_robomaker.1.*.log /mnt/c/Users/Jacky/Downloads/deepracer-log