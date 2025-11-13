cd b+tree/
ncu -f -o rodinia_b+tree_b+tree.out_file__data_b+tree_mil_txt_command__data_b+tree_command_txt --set full --import-source=yes ./b+tree.out file ../data/b+tree/mil.txt command ../data/b+tree/command.txt
cd ..


cd b+tree-opt/
ncu -f -o rodinia_b+tree-opt_b+tree.out_file__data_b+tree_mil_txt_command__data_b+tree_command_txt --set full --import-source=yes ./b+tree.out file ../data/b+tree/mil.txt command ../data/b+tree/command.txt
cd ..


cd backprop/
ncu -f -o rodinia_backprop_backprop_65536 --set full --import-source=yes ./backprop 65536
cd ..


cd backprop-opt1/
ncu -f -o rodinia_backprop-opt1_backprop_65536 --set full --import-source=yes ./backprop 65536
cd ..


cd backprop-opt2/
ncu -f -o rodinia_backprop-opt2_backprop_65536 --set full --import-source=yes ./backprop 65536
cd ..


cd bfs/
ncu -f -o rodinia_bfs_bfs__data_bfs_graph1MW_6_txt --set full --import-source=yes ./bfs ../data/bfs/graph1MW_6.txt
cd ..


cd bfs-opt/
ncu -f -o rodinia_bfs-opt_bfs__data_bfs_graph1MW_6_txt --set full --import-source=yes ./bfs ../data/bfs/graph1MW_6.txt
cd ..

# skipped, not in paper
#cd cfd/
#ncu -f -o rodinia_cfd_euler3d__data_cfd_fvcorr_domn_097K --set full --import-source=yes ./euler3d ../data/cfd/fvcorr.domn.097K
#ncu -f -o rodinia_cfd_euler3d_double__data_cfd_fvcorr_domn_097K --set full --import-source=yes ./euler3d_double ../data/cfd/fvcorr.domn.097K
#ncu -f -o rodinia_cfd_pre_euler3d__data_cfd_fvcorr_domn_097K --set full --import-source=yes ./pre_euler3d ../data/cfd/fvcorr.domn.097K
#ncu -f -o rodinia_cfd_pre_euler3d_double__data_cfd_fvcorr_domn_097K --set full --import-source=yes ./pre_euler3d_double ../data/cfd/fvcorr.domn.097K
#
#ncu -f -o rodinia_cfd_euler3d__data_cfd_fvcorr_domn_193K --set full --import-source=yes ./euler3d ../data/cfd/fvcorr.domn.193K
#ncu -f -o rodinia_cfd_euler3d_double__data_cfd_fvcorr_domn_193K --set full --import-source=yes ./euler3d_double ../data/cfd/fvcorr.domn.193K
#ncu -f -o rodinia_cfd_pre_euler3d__data_cfd_fvcorr_domn_193K --set full --import-source=yes ./pre_euler3d ../data/cfd/fvcorr.domn.193K
#ncu -f -o rodinia_cfd_pre_euler3d_double__data_cfd_fvcorr_domn_193K --set full --import-source=yes ./pre_euler3d_double ../data/cfd/fvcorr.domn.193K
#
#ncu -f -o rodinia_cfd_euler3d__data_cfd_missile_domn_0_2M --set full --import-source=yes ./euler3d ../data/cfd/missile.domn.0.2M
#ncu -f -o rodinia_cfd_euler3d_double__data_cfd_missile_domn_0_2M --set full --import-source=yes ./euler3d_double ../data/cfd/missile.domn.0.2M
#ncu -f -o rodinia_cfd_pre_euler3d__data_cfd_missile_domn_0_2M --set full --import-source=yes ./pre_euler3d ../data/cfd/missile.domn.0.2M
#ncu -f -o rodinia_cfd_pre_euler3d_double__data_cfd_missile_domn_0_2M --set full --import-source=yes ./pre_euler3d_double ../data/cfd/missile.domn.0.2M
#
#cd ..
#
# skipped, not in paper
#cd cfd-opt/
#ncu -f -o rodinia_cfd-opt_euler3d__data_cfd_fvcorr_domn_097K --set full --import-source=yes ./euler3d ../data/cfd/fvcorr.domn.097K
#ncu -f -o rodinia_cfd-opt_euler3d_double__data_cfd_fvcorr_domn_097K --set full --import-source=yes ./euler3d_double ../data/cfd/fvcorr.domn.097K
#ncu -f -o rodinia_cfd-opt_pre_euler3d__data_cfd_fvcorr_domn_097K --set full --import-source=yes ./pre_euler3d ../data/cfd/fvcorr.domn.097K
#ncu -f -o rodinia_cfd-opt_pre_euler3d_double__data_cfd_fvcorr_domn_097K --set full --import-source=yes ./pre_euler3d_double ../data/cfd/fvcorr.domn.097K
#
#ncu -f -o rodinia_cfd-opt_euler3d__data_cfd_fvcorr_domn_193K --set full --import-source=yes ./euler3d ../data/cfd/fvcorr.domn.193K
#ncu -f -o rodinia_cfd-opt_euler3d_double__data_cfd_fvcorr_domn_193K --set full --import-source=yes ./euler3d_double ../data/cfd/fvcorr.domn.193K
#ncu -f -o rodinia_cfd-opt_pre_euler3d__data_cfd_fvcorr_domn_193K --set full --import-source=yes ./pre_euler3d ../data/cfd/fvcorr.domn.193K
#ncu -f -o rodinia_cfd-opt_pre_euler3d_double__data_cfd_fvcorr_domn_193K --set full --import-source=yes ./pre_euler3d_double ../data/cfd/fvcorr.domn.193K
#
#ncu -f -o rodinia_cfd-opt_euler3d__data_cfd_missile_domn_0_2M --set full --import-source=yes ./euler3d ../data/cfd/missile.domn.0.2M
#ncu -f -o rodinia_cfd-opt_euler3d_double__data_cfd_missile_domn_0_2M --set full --import-source=yes ./euler3d_double ../data/cfd/missile.domn.0.2M
#ncu -f -o rodinia_cfd-opt_pre_euler3d__data_cfd_missile_domn_0_2M --set full --import-source=yes ./pre_euler3d ../data/cfd/missile.domn.0.2M
#ncu -f -o rodinia_cfd-opt_pre_euler3d_double__data_cfd_missile_domn_0_2M --set full --import-source=yes ./pre_euler3d_double ../data/cfd/missile.domn.0.2M
#
#cd ..


cd gaussian/
ncu -f -o rodinia_gaussian_gaussian_-s_1024 --set full --import-source=yes -k "Fan2" --launch-skip 250 --launch-count 50 ./gaussian -s 1024
cd ..


# skipped, error code 11
#cd gaussian-opt/
#ncu -f -o rodinia_gaussian-opt_gaussian_-f___data_gaussian_matrix4_txt --set full --import-source=yes -k "Fan2" --launch-skip 250 --launch-count 50 ./gaussian -f ../../data/gaussian/matrix4.txt
#cd ..


cd heartwall/
ncu -f -o rodinia_heartwall_heartwall__data_heartwall_test_avi_10 --set full --import-source=yes ./heartwall ../data/heartwall/test.avi 10
cd ..


# skipped, error opening avi file, code 255
#cd heartwall-opt/
#ncu -f -o rodinia_heartwall-opt_heartwall___data_heartwall_test_avi_10 --set full --import-source=yes ./heartwall ../../data/heartwall/test.avi 10
#cd ..


cd hotspot/
ncu -f -o rodinia_hotspot_hotspot_512_2_2__data_hotspot_temp_512__data_hotspot_power_512_output_out --set full --import-source=yes ./hotspot 512 2 2 ../data/hotspot/temp_512 ../data/hotspot/power_512 output.out
cd ..


cd hotspot-opt/
ncu -f -o rodinia_hotspot-opt_hotspot_512_2_2__data_hotspot_temp_512__data_hotspot_power_512_output_out --set full --import-source=yes ./hotspot 512 2 2 ../data/hotspot/temp_512 ../data/hotspot/power_512 output.out
cd ..


cd huffman/
ncu -f -o rodinia_huffman_pavle__data_huffman_test1024_H2_206587175259_in --set full --import-source=yes ./pavle ../data/huffman/test1024_H2.206587175259.in 
cd ..


cd huffman-opt/
ncu -f -o rodinia_huffman-opt_pavle__data_huffman_test1024_H2_206587175259_in --set full --import-source=yes ./pavle ../data/huffman/test1024_H2.206587175259.in 
cd ..


# skipped, uses legacy texture references that no longer compile
#cd kmeans/
#ncu -f -o rodinia_kmeans_kmeans_-o_-i__data_kmeans_kdd_cup --set full --import-source=yes ./kmeans -o -i ../data/kmeans/kdd_cup 
#cd ..
#
#
#cd kmeans-opt/
#ncu -f -o rodinia_kmeans-opt_kmeans_-o_-i__data_kmeans_kdd_cup --set full --import-source=yes ./kmeans -o -i ../data/kmeans/kdd_cup 
#cd ..


cd lavaMD/
ncu -f -o rodinia_lavaMD_lavaMD_-boxes1d_10 --set full --import-source=yes ./lavaMD -boxes1d 10
cd ..


cd lavaMD-opt/
ncu -f -o rodinia_lavaMD-opt_lavaMD_-boxes1d_10 --set full --import-source=yes ./lavaMD -boxes1d 10
cd ..


cd lud/
ncu -f -o rodinia_lud_lud_cuda_-s_256_-v --set full --import-source=yes ./cuda/lud_cuda -s 256 -v
cd ..


cd lud-opt/
ncu -f -o rodinia_lud-opt_lud_cuda_-s_256_-v --set full --import-source=yes ./cuda/lud_cuda -s 256 -v
cd ..


cd myocyte/
ncu -f -o rodinia_myocyte_myocyte.out_100_100_1 --set full --import-source=yes ./myocyte.out 100 100 1
cd ..


cd myocyte-opt1/
ncu -f -o rodinia_myocyte-opt1_myocyte.out_100_100_1 --set full --import-source=yes ./myocyte.out 100 100 1
cd ..


cd myocyte-opt2/
ncu -f -o rodinia_myocyte-opt2_myocyte.out_100_100_1 --set full --import-source=yes ./myocyte.out 100 100 1
cd ..


cd myocyte-opt3/
ncu -f -o rodinia_myocyte-opt3_myocyte.out_100_100_1 --set full --import-source=yes ./myocyte.out 100 100 1
cd ..


cd nw/
ncu -f -o rodinia_nw_needle_2048_10 --set full --import-source=yes ./needle 2048 10
cd ..


cd nw-opt/
ncu -f -o rodinia_nw-opt_needle_2048_10 --set full --import-source=yes ./needle 2048 10
cd ..


cd particlefilter/
ncu -f -o rodinia_particlefilter_particlefilter_float_-x_128_-y_128_-z_10_-np_1000 --set full --import-source=yes ./particlefilter_float -x 128 -y 128 -z 10 -np 1000
cd ..


cd particlefilter-opt/
ncu -f -o rodinia_particlefilter-opt_particlefilter_float_-x_128_-y_128_-z_10_-np_1000 --set full --import-source=yes ./particlefilter_float -x 128 -y 128 -z 10 -np 1000
cd ..


cd pathfinder/
ncu -f -o rodinia_pathfinder_pathfinder_100000_100_20_>_result_txt --set full --import-source=yes ./pathfinder 100000 100 20 > result.txt
cd ..


cd pathfinder-opt/
ncu -f -o rodinia_pathfinder-opt_pathfinder_100000_100_20_>_result_txt --set full --import-source=yes ./pathfinder 100000 100 20 > result.txt
cd ..


cd srad/srad_v1/
ncu -f -o rodinia_srad_srad_100_0_5_502_458 --set full --import-source=yes -k "reduce" ./srad 100 0.5 502 458
cd ../..


cd srad/srad_v1-opt/
ncu -f -o rodinia_srad_srad_100_0_5_502_458 --set full --import-source=yes -k "reduce" ./srad 100 0.5 502 458
cd ../..


# skipped, not in paper
#cd srad/srad_v2/
#ncu -f -o rodinia_srad_srad_2048_2048_0_127_0_127_0_5_2 --set full --import-source=yes -k "reduce" ./srad 2048 2048 0 127 0 127 0.5 2
#cd ../..


cd streamcluster/
ncu -f -o rodinia_streamcluster_sc_gpu_10_20_256_1024_1024_1000_none_output_txt_1 --set full --import-source=yes --launch-skip 200 --launch-count 200 ./sc_gpu 10 20 256 1024 1024 1000 none output.txt 1
cd ..


cd streamcluster-opt/
ncu -f -o rodinia_streamcluster-opt_sc_gpu_10_20_256_1024_1024_1000_none_output_txt_1 --set full --import-source=yes --launch-skip 200 --launch-count 200 ./sc_gpu 10 20 256 1024 1024 1000 none output.txt 1
cd ..
