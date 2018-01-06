
#!/bin/bash
count=37
while [ $count -le 125 ]
do
 echo "$count"
 python pose_extract.py $count
 sleep 1
 (( count++ ))
done