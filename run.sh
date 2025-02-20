#!/bin/bash
cd ./data
make clean && make
cd ../
make clean && make;

for ((M=128; M<=128; M=M*2))
do
	for ((B=4; B<=256; B=B*2))
	do
		for ((K=16; K<=1024; K=K*2))
		do
			cd ./data
			./gen_data $M $K
			cd - > /dev/null
			./vbat_grouped_gemm $B >> result.m
		done
	done
done
