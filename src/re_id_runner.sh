
#!/bin/bash
count=4
while [ $count -le 30 ]
do
 echo "$count"
 python pose_extract.py $count
 sleep 1
 (( count++ ))
done