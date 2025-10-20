cd backprop/
./backprop 65536
cd ..


cd backprop-opt1/
./backprop 65536
cd ..


cd backprop-opt2/
./backprop 65536
cd ..


cd bfs/
./bfs ../data/bfs/graph1MW_6.txt
cd ..


cd bfs-opt/
./bfs ../data/bfs/graph1MW_6.txt
cd ..


cd b+tree/
./b+tree.out file ../data/b+tree/mil.txt command ../data/b+tree/command.txt
cd ..


cd b+tree-opt/
./b+tree.out file ../data/b+tree/mil.txt command ../data/b+tree/command.txt
cd ..


cd cfd/
./euler3d ../data/cfd/fvcorr.domn.097K
./euler3d_double ../data/cfd/fvcorr.domn.097K
./pre_euler3d ../data/cfd/fvcorr.domn.097K
./pre_euler3d_double ../data/cfd/fvcorr.domn.097K

./euler3d ../data/cfd/fvcorr.domn.193K
./euler3d_double ../data/cfd/fvcorr.domn.193K
./pre_euler3d ../data/cfd/fvcorr.domn.193K
./pre_euler3d_double ../data/cfd/fvcorr.domn.193K

./euler3d ../data/cfd/missile.domn.0.2M
./euler3d_double ../data/cfd/missile.domn.0.2M
./pre_euler3d ../data/cfd/missile.domn.0.2M
./pre_euler3d_double ../data/cfd/missile.domn.0.2M
cd ..


cd cfd-opt/
./euler3d ../data/cfd/fvcorr.domn.097K
./euler3d_double ../data/cfd/fvcorr.domn.097K
./pre_euler3d ../data/cfd/fvcorr.domn.097K
./pre_euler3d_double ../data/cfd/fvcorr.domn.097K

./euler3d ../data/cfd/fvcorr.domn.193K
./euler3d_double ../data/cfd/fvcorr.domn.193K
./pre_euler3d ../data/cfd/fvcorr.domn.193K
./pre_euler3d_double ../data/cfd/fvcorr.domn.193K

./euler3d ../data/cfd/missile.domn.0.2M
./euler3d_double ../data/cfd/missile.domn.0.2M
./pre_euler3d ../data/cfd/missile.domn.0.2M
./pre_euler3d_double ../data/cfd/missile.domn.0.2M
cd ..


cd gaussian/
./gaussian -s 1024
cd ..


cd gaussian-opt/
./gaussian -f ../../data/gaussian/matrix4.txt
./gaussian -s 16
cd ..


cd heartwall/
./heartwall ../data/heartwall/test.avi 10
cd ..


cd heartwall-opt/
./heartwall ../../data/heartwall/test.avi 10
cd ..


cd hotspot/
./hotspot 512 2 2 ../data/hotspot/temp_512 ../data/hotspot/power_512 output.out
cd ..


cd hotspot-opt/
./hotspot 512 2 2 ../data/hotspot/temp_512 ../data/hotspot/power_512 output.out
cd ..


cd huffman/
./pavle ../data/huffman/test1024_H2.206587175259.in 
cd ..


cd huffman-opt/
./pavle ../data/huffman/test1024_H2.206587175259.in 
cd ..


cd kmeans/
./kmeans -o -i ../data/kmeans/kdd_cup 
cd ..


cd kmeans-opt/
./kmeans -o -i ../data/kmeans/kdd_cup 
cd ..


cd lavaMD/
./lavaMD -boxes1d 10
cd ..


cd lavaMD-opt/
./lavaMD -boxes1d 10
cd ..


cd lud/
# cuda/lud_cuda -i ../data/lud/256.dat 
cuda/lud_cuda -s 256 -v
cd ..


cd lud-opt/
# cuda/lud_cuda -i ../data/lud/256.dat 
cuda/lud_cuda -s 256 -v
cd ..


cd myocyte/
./myocyte.out 100 100 1
cd ..


cd myocyte-opt1/
./myocyte.out 100 100 1
cd ..


cd myocyte-opt2/
./myocyte.out 100 100 1
cd ..


cd myocyte-opt3/
./myocyte.out 100 100 1
cd ..


cd nw/
./needle 2048 10
cd ..


cd nw-opt/
./needle 2048 10
cd ..


cd particlefilter/
./particlefilter_float -x 128 -y 128 -z 10 -np 1000
cd ..


cd particlefilter-opt/
./particlefilter_naive -x 128 -y 128 -z 10 -np 1000
./particlefilter_float -x 128 -y 128 -z 10 -np 1000
cd ..


cd pathfinder/
./pathfinder 100000 100 20 > result.txt
cd ..


cd pathfinder-opt/
./pathfinder 100000 100 20 > result.txt
cd ..


cd srad/srad_v1/
./srad 100 0.5 502 458
cd ../..


cd srad/srad_v1-opt/
./srad 100 0.5 502 458
cd ../..


cd srad/srad_v2/
./srad 2048 2048 0 127 0 127 0.5 2
cd ../..


cd streamcluster/
./sc_gpu 10 20 256 1024 1024 1000 none output.txt 1
cd ..


cd streamcluster-opt/
./sc_gpu 10 20 256 1024 1024 1000 none output.txt 1
cd ..


