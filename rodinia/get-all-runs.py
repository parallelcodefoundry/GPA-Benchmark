#!/usr/bin/env python3

import os
import sys
from pathlib import Path

def get_run_prof_command(mode, dir_name, run_file):
    # Check if file contains newlines
    if '\n' in run_file:
        # Get the line that starts with ./
        exe_line = next(line for line in run_file.split('\n') if line.startswith('./'))
    else:
        exe_line = run_file

    exe_name = exe_line.split('./')[1].split()[0]
    exe_args = '_'.join(exe_line.split()[1:]).replace('..', '').replace('/', '_').replace('.', '_')
    prof_command = ""
    if mode == 'profile':
        profile_name = f"rodinia_{dir_name}_{exe_name}_{exe_args}"
        prof_command = f"ncu profile -f -o {profile_name} --set full --import-source=yes"
    return prof_command + " " + exe_line


def main():
    if len(sys.argv) != 2:
        print("Usage: ./get-all-runs.py <MODE>")
        sys.exit(1)

    mode = sys.argv[1]
    output_file = f"{mode}-all.sh"
    if mode != 'run' and mode != 'profile':
        print("Usage: ./get-all-runs.py <profile|run>")
        sys.exit(1)

    # Check if the output file already exists, early exit if it does
    if os.path.exists(output_file):
        print(f"{output_file} file already exists")
        sys.exit(1)

    # Get all subdirectories in the current directory
    subdirs = sorted([d for d in Path('.').iterdir() if d.is_dir()])

    with open(output_file, 'w') as f:
        # Loop over all subdirectories
        for dir_path in subdirs:
            dir_name = dir_path.name
            dir_str = f"{dir_name}/"

            # Skip specific directories
            if dir_name in ['data', 'common']:
                continue

            # Handle regular directories (not srad, cfd, or cfd-opt)
            if dir_name not in ['srad', 'cfd', 'cfd-opt']:
                run_file = dir_path / 'run'
                if run_file.exists():
                    f.write(f"cd {dir_str}\n")
                    with open(run_file, 'r') as run_f:
                        prof_command = get_run_prof_command(mode, dir_name, run_f.read())
                        f.write(f"{prof_command}\n")
                    f.write("cd ..\n")
                    f.write("\n\n")

            # Handle srad directory specially
            elif dir_name == 'srad':
                srad_subdirs = sorted([d for d in dir_path.iterdir() if d.is_dir()])
                for subdir in srad_subdirs:
                    run_file = subdir / 'run'
                    if run_file.exists():
                        # Get relative path from current directory
                        rel_path = subdir.relative_to('.')
                        f.write(f"cd {rel_path}/\n")
                        with open(run_file, 'r') as run_f:
                            prof_command = get_run_prof_command(mode, dir_name, run_f.read())
                            f.write(f"{prof_command}\n")
                        f.write("cd ../..\n")
                        f.write("\n\n")

            # Handle cfd and cfd-opt directories
            elif dir_name in ['cfd', 'cfd-opt']:
                f.write(f"cd {dir_str}\n")

                # Define executables and datasets
                executables = ['euler3d', 'euler3d_double', 'pre_euler3d', 'pre_euler3d_double']
                datasets = ['fvcorr.domn.097K', 'fvcorr.domn.193K', 'missile.domn.0.2M']

                # Generate commands for each dataset with all executables
                for dataset in datasets:
                    for executable in executables:
                        prof_command = get_run_prof_command(mode, dir_name, f"./{executable} ../data/cfd/{dataset}")
                        f.write(f"{prof_command}\n")
                    f.write("\n")

                f.write("cd ..\n")
                f.write("\n\n")

    # Make the output file executable
    os.chmod(output_file, 0o755)
    print(f"Successfully created {output_file}")

if __name__ == '__main__':
    main()

