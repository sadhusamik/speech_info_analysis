export train_cmd="queue.pl -l 'hostname=!b03*'"
export decode_cmd="queue.pl --mem 2G"
export cuda_cmd="queue.pl --gpu 1 -l 'hostname=b1[12345678]*|c*'"
export cuda_ccmd="queue.pl --gpu 1 -l 'hostname=c*'"