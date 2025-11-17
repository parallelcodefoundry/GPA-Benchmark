#!/bin/bash
METRICS="--metrics regex:sm__inst_executed_pipe_[^.]*.avg.pct_of_peak_sustained_active$,regex:sm__sass_thread_inst_executed_op.*sum$,regex:l1tex__t_set_.*_pipe_lsu_mem_global_op_ld.sum$,regex:l1tex__t_set_accesses.sum$,regex:l1tex__t_requests.sum$,regex:l1tex__m_xbar2l1tex_read_sectors.sum$,sm__average_thread_inst_executed_pred_on_per_inst_executed_realtime,regex:sm__sass_inst_executed.*sum$,regex:sm__inst_issued.avg.per_cycle_active$,regex:.*throughput.avg.pct_of_peak_sustained_active$,regex:.*throughput.avg.pct_of_peak_sustained_elapsed$"

cd b+tree/
ncu -f -o rodinia_b+tree_b+tree.out_file__data_b+tree_mil_txt_command__data_b+tree_command_txt --set full --import-source=yes $METRICS --kernel-name findRangeK ./b+tree.out file ../data/b+tree/mil.txt command ../data/b+tree/command.txt
cd ..


cd b+tree-opt/
ncu -f -o rodinia_b+tree-opt_b+tree.out_file__data_b+tree_mil_txt_command__data_b+tree_command_txt --set full --import-source=yes $METRICS --kernel-name findRangeK ./b+tree.out file ../data/b+tree/mil.txt command ../data/b+tree/command.txt
cd ..


cd backprop/
ncu -f -o rodinia_backprop_backprop_65536 --set full --import-source=yes $METRICS --kernel-name bpnn_layerforward_CUDA ./backprop 65536
cd ..


cd backprop-opt1/
ncu -f -o rodinia_backprop-opt1_backprop_65536 --set full --import-source=yes $METRICS --kernel-name bpnn_layerforward_CUDA ./backprop 65536
cd ..


cd backprop-opt2/
ncu -f -o rodinia_backprop-opt2_backprop_65536 --set full --import-source=yes $METRICS --kernel-name bpnn_layerforward_CUDA ./backprop 65536
cd ..


cd bfs/
ncu -f -o rodinia_bfs_bfs__data_bfs_graph1MW_6_txt --set full --import-source=yes $METRICS --kernel-name Kernel --launch-skip 8 --launch-count 1 ./bfs ../data/bfs/graph1MW_6.txt
cd ..


cd bfs-opt/
ncu -f -o rodinia_bfs-opt_bfs__data_bfs_graph1MW_6_txt --set full --import-source=yes $METRICS --kernel-name Kernel --launch-skip 8 --launch-count 1 ./bfs ../data/bfs/graph1MW_6.txt
cd ..

# skipped, not in paper
#cd cfd/
#ncu -f -o rodinia_cfd_euler3d__data_cfd_fvcorr_domn_097K --set full --import-source=yes $METRICS ./euler3d ../data/cfd/fvcorr.domn.097K
#ncu -f -o rodinia_cfd_euler3d_double__data_cfd_fvcorr_domn_097K --set full --import-source=yes $METRICS ./euler3d_double ../data/cfd/fvcorr.domn.097K
#ncu -f -o rodinia_cfd_pre_euler3d__data_cfd_fvcorr_domn_097K --set full --import-source=yes $METRICS ./pre_euler3d ../data/cfd/fvcorr.domn.097K
#ncu -f -o rodinia_cfd_pre_euler3d_double__data_cfd_fvcorr_domn_097K --set full --import-source=yes $METRICS ./pre_euler3d_double ../data/cfd/fvcorr.domn.097K
#
#ncu -f -o rodinia_cfd_euler3d__data_cfd_fvcorr_domn_193K --set full --import-source=yes $METRICS ./euler3d ../data/cfd/fvcorr.domn.193K
#ncu -f -o rodinia_cfd_euler3d_double__data_cfd_fvcorr_domn_193K --set full --import-source=yes $METRICS ./euler3d_double ../data/cfd/fvcorr.domn.193K
#ncu -f -o rodinia_cfd_pre_euler3d__data_cfd_fvcorr_domn_193K --set full --import-source=yes $METRICS ./pre_euler3d ../data/cfd/fvcorr.domn.193K
#ncu -f -o rodinia_cfd_pre_euler3d_double__data_cfd_fvcorr_domn_193K --set full --import-source=yes $METRICS ./pre_euler3d_double ../data/cfd/fvcorr.domn.193K
#
#ncu -f -o rodinia_cfd_euler3d__data_cfd_missile_domn_0_2M --set full --import-source=yes $METRICS ./euler3d ../data/cfd/missile.domn.0.2M
#ncu -f -o rodinia_cfd_euler3d_double__data_cfd_missile_domn_0_2M --set full --import-source=yes $METRICS ./euler3d_double ../data/cfd/missile.domn.0.2M
#ncu -f -o rodinia_cfd_pre_euler3d__data_cfd_missile_domn_0_2M --set full --import-source=yes $METRICS ./pre_euler3d ../data/cfd/missile.domn.0.2M
#ncu -f -o rodinia_cfd_pre_euler3d_double__data_cfd_missile_domn_0_2M --set full --import-source=yes $METRICS ./pre_euler3d_double ../data/cfd/missile.domn.0.2M
#
#cd ..
#
# skipped, not in paper
#cd cfd-opt/
#ncu -f -o rodinia_cfd-opt_euler3d__data_cfd_fvcorr_domn_097K --set full --import-source=yes $METRICS ./euler3d ../data/cfd/fvcorr.domn.097K
#ncu -f -o rodinia_cfd-opt_euler3d_double__data_cfd_fvcorr_domn_097K --set full --import-source=yes $METRICS ./euler3d_double ../data/cfd/fvcorr.domn.097K
#ncu -f -o rodinia_cfd-opt_pre_euler3d__data_cfd_fvcorr_domn_097K --set full --import-source=yes $METRICS ./pre_euler3d ../data/cfd/fvcorr.domn.097K
#ncu -f -o rodinia_cfd-opt_pre_euler3d_double__data_cfd_fvcorr_domn_097K --set full --import-source=yes $METRICS ./pre_euler3d_double ../data/cfd/fvcorr.domn.097K
#
#ncu -f -o rodinia_cfd-opt_euler3d__data_cfd_fvcorr_domn_193K --set full --import-source=yes $METRICS ./euler3d ../data/cfd/fvcorr.domn.193K
#ncu -f -o rodinia_cfd-opt_euler3d_double__data_cfd_fvcorr_domn_193K --set full --import-source=yes $METRICS ./euler3d_double ../data/cfd/fvcorr.domn.193K
#ncu -f -o rodinia_cfd-opt_pre_euler3d__data_cfd_fvcorr_domn_193K --set full --import-source=yes $METRICS ./pre_euler3d ../data/cfd/fvcorr.domn.193K
#ncu -f -o rodinia_cfd-opt_pre_euler3d_double__data_cfd_fvcorr_domn_193K --set full --import-source=yes $METRICS ./pre_euler3d_double ../data/cfd/fvcorr.domn.193K
#
#ncu -f -o rodinia_cfd-opt_euler3d__data_cfd_missile_domn_0_2M --set full --import-source=yes $METRICS ./euler3d ../data/cfd/missile.domn.0.2M
#ncu -f -o rodinia_cfd-opt_euler3d_double__data_cfd_missile_domn_0_2M --set full --import-source=yes $METRICS ./euler3d_double ../data/cfd/missile.domn.0.2M
#ncu -f -o rodinia_cfd-opt_pre_euler3d__data_cfd_missile_domn_0_2M --set full --import-source=yes $METRICS ./pre_euler3d ../data/cfd/missile.domn.0.2M
#ncu -f -o rodinia_cfd-opt_pre_euler3d_double__data_cfd_missile_domn_0_2M --set full --import-source=yes $METRICS ./pre_euler3d_double ../data/cfd/missile.domn.0.2M
#
#cd ..


cd gaussian/
ncu -f -o rodinia_gaussian_gaussian_-s_1024 --set full --import-source=yes $METRICS -k "Fan2" --launch-skip 250 --launch-count 1 ./gaussian -s 1024
cd ..


# skipped, error code 11
#cd gaussian-opt/
#ncu -f -o rodinia_gaussian-opt_gaussian_-f___data_gaussian_matrix4_txt --set full --import-source=yes $METRICS -k "Fan2" --launch-skip 250 --launch-count 50 ./gaussian -f ../../data/gaussian/matrix4.txt
#cd ..


cd heartwall/
ncu -f -o rodinia_heartwall_heartwall__data_heartwall_test_avi_10 --set full --import-source=yes $METRICS --launch-skip 4 --launch-count 1 ./heartwall ../data/heartwall/test.avi 10
cd ..


# skipped, error opening avi file, code 255
#cd heartwall-opt/
#ncu -f -o rodinia_heartwall-opt_heartwall___data_heartwall_test_avi_10 --set full --import-source=yes $METRICS ./heartwall ../../data/heartwall/test.avi 10
#cd ..


cd hotspot/
ncu -f -o rodinia_hotspot_hotspot_512_2_2__data_hotspot_temp_512__data_hotspot_power_512_output_out --set full --import-source=yes $METRICS ./hotspot 512 2 2 ../data/hotspot/temp_512 ../data/hotspot/power_512 output.out
cd ..


cd hotspot-opt/
ncu -f -o rodinia_hotspot-opt_hotspot_512_2_2__data_hotspot_temp_512__data_hotspot_power_512_output_out --set full --import-source=yes $METRICS ./hotspot 512 2 2 ../data/hotspot/temp_512 ../data/hotspot/power_512 output.out
cd ..


cd huffman/
ncu -f -o rodinia_huffman_pavle__data_huffman_test1024_H2_206587175259_in --set full --import-source=yes $METRICS --kernel-name vlc_encode_kernel_sm64huff --launch-skip 3 --launch-count 1 ./pavle ../data/huffman/test1024_H2.206587175259.in
cd ..


cd huffman-opt/
ncu -f -o rodinia_huffman-opt_pavle__data_huffman_test1024_H2_206587175259_in --set full --import-source=yes $METRICS --kernel-name vlc_encode_kernel_sm64huff --launch-skip 3 --launch-count 1 ./pavle ../data/huffman/test1024_H2.206587175259.in
cd ..


# skipped, uses legacy texture references that no longer compile
#cd kmeans/
#ncu -f -o rodinia_kmeans_kmeans_-o_-i__data_kmeans_kdd_cup --set full --import-source=yes $METRICS ./kmeans -o -i ../data/kmeans/kdd_cup
#cd ..
#
#
#cd kmeans-opt/
#ncu -f -o rodinia_kmeans-opt_kmeans_-o_-i__data_kmeans_kdd_cup --set full --import-source=yes $METRICS ./kmeans -o -i ../data/kmeans/kdd_cup
#cd ..


cd lavaMD/
ncu -f -o rodinia_lavaMD_lavaMD_-boxes1d_10 --set full --import-source=yes $METRICS ./lavaMD -boxes1d 10
cd ..


cd lavaMD-opt/
ncu -f -o rodinia_lavaMD-opt_lavaMD_-boxes1d_10 --set full --import-source=yes $METRICS ./lavaMD -boxes1d 10
cd ..


cd lud/
ncu -f -o rodinia_lud_lud_cuda_-s_256_-v --set full --import-source=yes $METRICS --kernel-name lud_diagonal --launch-skip 7 --launch-count 1 ./cuda/lud_cuda -s 256 -v
cd ..


cd lud-opt/
ncu -f -o rodinia_lud-opt_lud_cuda_-s_256_-v --set full --import-source=yes $METRICS --kernel-name lud_diagonal --launch-skip 7 --launch-count 1 ./cuda/lud_cuda -s 256 -v
cd ..


cd myocyte/
ncu -f -o rodinia_myocyte_myocyte.out_100_100_1 --set full --import-source=yes $METRICS ./myocyte.out 100 100 1
cd ..


cd myocyte-opt1/
ncu -f -o rodinia_myocyte-opt1_myocyte.out_100_100_1 --set full --import-source=yes $METRICS ./myocyte.out 100 100 1
cd ..


cd myocyte-opt2/
ncu -f -o rodinia_myocyte-opt2_myocyte.out_100_100_1 --set full --import-source=yes $METRICS ./myocyte.out 100 100 1
cd ..


cd myocyte-opt3/
ncu -f -o rodinia_myocyte-opt3_myocyte.out_100_100_1 --set full --import-source=yes $METRICS ./myocyte.out 100 100 1
cd ..


cd nw/
ncu -f -o rodinia_nw_needle_2048_10 --set full --import-source=yes $METRICS --kernel-name needle_cuda_shared_1 --launch-skip 127 --launch-count 1 ./needle 2048 10
cd ..


cd nw-opt/
ncu -f -o rodinia_nw-opt_needle_2048_10 --set full --import-source=yes $METRICS --kernel-name needle_cuda_shared_1 --launch-skip 127 --launch-count 1 ./needle 2048 10
cd ..


# skipped, error code 6
#cd particlefilter/
#ncu -f -o rodinia_particlefilter_particlefilter_float_-x_128_-y_128_-z_10_-np_1000 --set full --import-source=yes $METRICS --kernel-name likelihood_kernel --launch-skip 2 --launch-count 1 ./particlefilter_float -x 128 -y 128 -z 10 -np 1000
#cd ..
#
#
#cd particlefilter-opt/
#ncu -f -o rodinia_particlefilter-opt_particlefilter_float_-x_128_-y_128_-z_10_-np_1000 --set full --import-source=yes $METRICS --kernel-name likelihood_kernel --launch-skip 2 --launch-count 1 ./particlefilter_float -x 128 -y 128 -z 10 -np 1000
#cd ..


cd pathfinder/
ncu -f -o rodinia_pathfinder_pathfinder_100000_100_20_>_result_txt --set full --import-source=yes $METRICS --launch-skip 2 --launch-count 1 ./pathfinder 100000 100 20 > result.txt
cd ..


cd pathfinder-opt/
ncu -f -o rodinia_pathfinder-opt_pathfinder_100000_100_20_>_result_txt --set full --import-source=yes $METRICS --launch-skip 2 --launch-count 1 ./pathfinder 100000 100 20 > result.txt
cd ..


cd srad/srad_v1/
ncu -f -o rodinia_srad_srad_100_0_5_502_458 --set full --import-source=yes $METRICS -k "reduce" --launch-skip 99 --launch-count 1 ./srad 100 0.5 502 458
cd ../..


cd srad/srad_v1-opt/
ncu -f -o rodinia_srad_srad_100_0_5_502_458 --set full --import-source=yes $METRICS -k "reduce" --launch-skip 99 --launch-count 1 ./srad 100 0.5 502 458
cd ../..


# skipped, not in paper
#cd srad/srad_v2/
#ncu -f -o rodinia_srad_srad_2048_2048_0_127_0_127_0_5_2 --set full --import-source=yes $METRICS -k "reduce" ./srad 2048 2048 0 127 0 127 0.5 2
#cd ../..


cd streamcluster/
ncu -f -o rodinia_streamcluster_sc_gpu_10_20_256_1024_1024_1000_none_output_txt_1 --set full --import-source=yes $METRICS --launch-skip 300 --launch-count 1 ./sc_gpu 10 20 256 1024 1024 1000 none output.txt 1
cd ..


cd streamcluster-opt/
ncu -f -o rodinia_streamcluster-opt_sc_gpu_10_20_256_1024_1024_1000_none_output_txt_1 --set full --import-source=yes $METRICS --launch-skip 200 --launch-count 1 ./sc_gpu 10 20 256 1024 1024 1000 none output.txt 1
cd ..
