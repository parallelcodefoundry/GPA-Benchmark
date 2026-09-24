#include "XSbench_header.cuh"

int main(int argc, char *argv[]) {
        // =====================================================================
        // Initialization & Command Line Read-In
        // =====================================================================
        int version = 20;
        int mype = 0;
        double omp_start, omp_end;
        int nprocs = 1;
        unsigned long long verification;

        // Process CLI Fields -- store in "Inputs" structure
        Inputs in = read_CLI(argc, argv);

        // Print-out of Input Summary
        if (mype == 0)
                print_inputs(in, nprocs, version);

        // =====================================================================
        // Prepare Nuclide Energy Grids, Unionized Energy Grid, & Material Data
        // This is not reflective of a real Monte Carlo simulation workload,
        // therefore, do not profile this region!
        // =====================================================================

        SimulationData SD;

        // If read from file mode is selected, skip initialization and load
        // all simulation data structures from file instead
        if (in.binary_mode == READ)
                SD = binary_read(in);
        else
                SD = grid_init_do_not_profile(in, mype);

        // If writing from file mode is selected, write all simulation data
        // structures to file
        if (in.binary_mode == WRITE && mype == 0)
                binary_write(in, SD);

	Profile profile;

        // =====================================================================
        // Cross Section (XS) Parallel Lookup Simulation
        // This is the section that should be profiled, as it reflects a
        // realistic continuous energy Monte Carlo macroscopic cross section
        // lookup kernel.
        // =====================================================================
        if (mype == 0) {
                printf("\n");
                border_print();
                center_print("SIMULATION", 79);
                border_print();
        }

        // Start Simulation Timer
        omp_start = get_time();

        // Run simulation
        if (in.simulation_method == EVENT_BASED) {
                verification = run_event_based_simulation(in, SD, mype, &profile, in.thread_block_size);
        } else {
                printf(
                        "History-based simulation not implemented in CUDA code. Instead,\nuse "
                        "the event-based method with \"-m event\" argument.\n");
                exit(1);
        }

        if (mype == 0) {
                printf("\n");
                printf("Simulation complete.\n");
        }


        // End Simulation Timer
        omp_end = get_time();

        // Release device memory
        release_memory(SD);

        // Final Hash Step
        verification = verification % 999983;

        // Print / Save Results and Exit
        int is_invalid_result =
                print_results(in, mype, omp_end - omp_start, nprocs, verification);

	print_profile(profile, in);

	// APPEB/Frontier port: exit 0 regardless of XSBench's built-in hash check. That check only
	// knows the hash of the DEFAULT lookup count, so it reports "INAVALID" (and used to exit 1)
	// for every other -l, including the scored -l 100000000. Correctness is the harness's job:
	// driver_apps.frontier.yaml compares the printed checksum with the value stored for the
	// exact run arguments (expected_checksum). Crashes and HIP errors still exit non-zero.
	(void) is_invalid_result;
	return 0;
}
