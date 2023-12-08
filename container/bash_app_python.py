# this is a helper function to perform md simulations. Executed by parsl on HPC infrastructure, but

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import cloudpickle
from parsl import File, bash_app, python_app
from parsl.dataflow.dflow import AppFuture

def get_hostname():
    import socket
    return socket.gethostname()


# @typeguard.typechecked
def bash_app_python(
    function=None,
    executors="all",
    precommand="",  # command to run before the python command
):
    def decorator(func):
        def wrapper(
            *args,
            execution_folder=None,
            stdout=None,
            stderr=None,
            inputs=[],
            outputs=[],
            **kwargs,
        ):
            # merge in and outputs
            inputs = [*inputs, *kwargs.pop("inputs", [])]
            outputs = [*outputs, *kwargs.pop("outputs", [])]

            @bash_app(executors=executors)
            def fun(*args, stdout, stderr, inputs, outputs, **kwargs):
                if len(inputs) > 1:
                    kwargs["inputs"] = inputs[:-1]
                if len(outputs) > 1:
                    kwargs["outputs"] = outputs[:-1]

                filename_in = inputs[-1].filepath
                filename_out = outputs[-1].filepath
                fold = os.path.dirname(filename_in)
                if not os.path.exists(fold):
                    os.mkdir(fold)

                with open(filename_in, "rb+") as f:
                    cloudpickle.dump((func, args, kwargs), f)

                #This part is just to bind free cores to the ranks of the job, a file per node is created which keeps track of which (group of) cores are used
                #This would not be necessary if parsl/mpirun/slurm does what it should do, which is assign the ranks to the free cores, but it doesn't
                precommand_ranknr = ""
                endcommand = ""
                if precommand != "":

                    #To get number of replicas
                    lsp = precommand.split()
                    flag = False
                    for el in lsp:
                        if el == "-n":
                            flag = True
                        elif flag:
                            num_replicas = int(el)
                            num_parts = int(128/num_replicas)   #128 is the number of cores per node
                            break

                    from filelock import Timeout, FileLock
                    import numpy as np
                    hostname_txt = "configs/rankfiles/" + get_hostname() + ".txt"
                    lock = FileLock(hostname_txt+".lock")
                    with lock:
                        if not os.path.exists(hostname_txt):
                            with open(hostname_txt, "w") as file:
                                file.write("This is the file for node "+get_hostname()+".\n")
                        fr = open(hostname_txt, "r")
                        start_lst = []
                        for line in fr.readlines():
                            lsp = line.split()
                            if lsp[0] == "start":
                                if int(lsp[1]) < num_parts:
                                    if int(lsp[1]) in start_lst:
                                        start_lst.remove(int(lsp[1]))
                                    start_lst.append(int(lsp[1]))
                            if lsp[0] == "end":
                                if int(lsp[1]) < num_parts:
                                    start_lst.remove(int(lsp[1]))
                        fr.close()
                        if len(start_lst) == num_parts: #If a job is terminated due to walltime for example the "end" line is not printed, therefore it could seem like all cores are used
                            rank_nr = start_lst[0] #Use the cores that got used the latest, because one of them must be free and the oldest used one is the safest option
                        else:
                            for i in range(num_parts):
                                if i not in start_lst:
                                    rank_nr = i
                                    break
                        fa = open(hostname_txt, "a")
                        fa.write("start " + str(rank_nr) + "\n")
                        fa.close()
                        precommand_ranknr = precommand + "_" + str(num_replicas) + "_" + str(rank_nr)
                        endcommand = 'echo "end '+str(rank_nr)+'" >> '+ hostname_txt
                
                return f"{precommand_ranknr} python  -u { os.path.realpath( __file__ ) } --folder {execution_folder} --file_in {  os.path.relpath(filename_in,execution_folder)  }  --file_out  {  os.path.relpath(filename_out,execution_folder)  }; {endcommand}"


            fun.__name__ = func.__name__

            def rename(name):
                path, name = os.path.split(name.filepath)
                return os.path.join(path, f"bash_app_{name}")

            if execution_folder is None:
                p = Path.cwd() / func.__name__

                i = 0
                while p.exists():
                    p = Path.cwd() / (f"{func.__name__}_{i:0>3}")
                    i += 1

                execution_folder = p

            if isinstance(execution_folder, str):
                execution_folder = Path(execution_folder)

            execution_folder.mkdir(exist_ok=True)

            def rename_num(name, i):
                stem = name.name.split(".")
                stem[0] = f"{stem[0]}_{i:0>3}"
                return name.parent / ".".join(stem)

            def find_num(name):
                i = 0
                while rename_num(name, i).exists():
                    i += 1
                return i, rename_num(name, i)

            _, lockfile = find_num(execution_folder / f"{func.__name__}.lock")

            with open(lockfile, "w+"):
                pass

            i, file_in = find_num(execution_folder / f"{func.__name__}.inp.cloudpickle")
            with open(file_in, "w+"):
                pass

            file_out = rename_num(
                execution_folder / f"{func.__name__}.outp.cloudpickle", i
            )

            i, stdout = find_num(
                execution_folder
                / (f"{ func.__name__}.stdout" if stdout is None else Path(stdout))
            )

            stderr = rename_num(
                execution_folder
                / (f"{ func.__name__}.stderr" if stderr is None else Path(stderr)),
                i,
            )

            future: AppFuture = fun(
                inputs=[*inputs, File(str(file_in))],
                outputs=[*[File(rename(o)) for o in outputs], File(str(file_out))],
                stdout=str(stdout),
                stderr=str(stderr),
                *args,
                **kwargs,
            )

            @python_app(executors=["default_threads"])
            def load(inputs=[], outputs=[]):
                with open(inputs[-1].filepath, "rb") as f:
                    result = cloudpickle.load(f)
                import os
                import shutil

                os.remove(inputs[-1].filepath)
                for i, o in zip(inputs[:-1], outputs):
                    shutil.move(i.filepath, o.filepath)

                # transfer complete,remove lock file
                os.remove(str(lockfile))

                return result

            return load(inputs=future.outputs, outputs=outputs)

        return wrapper

    if function is not None:
        return decorator(function)
    return decorator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute python function in a bash app"
    )
    parser.add_argument(
        "--file_in",
        type=str,
        help="path to file f containing cloudpickle.dump((func, args, kwargs), f)",
    )

    parser.add_argument(
        "--file_out",
        type=str,
        help="path to file f containing cloudpickle.dump((func, args, kwargs), f)",
    )
    parser.add_argument("--folder", type=str, help="working directory")
    args = parser.parse_args()
    os.chdir(args.folder)

    rank = 0
    use_mpi = False

    try:
        from mpi4py import MPI

        MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        num_ranks = comm.Get_size()

        if num_ranks > 1:
            use_mpi = True
    except:
        pass

    if rank == 0:
        print("#" * 20)
        print(f"got input {sys.argv}")
        print(f"task started at {datetime.now():%d/%m/%Y %H:%M:%S }")
        print(f"working in folder {os.getcwd()}")
        if use_mpi:
            print(f"using mpi with {num_ranks} ranks")

        with open(args.file_in, "rb") as f:
            func, fargs, fkwargs = cloudpickle.load(f)
        print("#" * 20)
    else:
        func = None
        fargs = None
        fkwargs = None

    if use_mpi:
        func = comm.bcast(func, root=0)
        fargs = comm.bcast(fargs, root=0)
        fkwargs = comm.bcast(fkwargs, root=0)

    a = func(*fargs, **fkwargs)

    if use_mpi:
        a = comm.gather(a, root=0)

    if rank == 0:
        with open(args.file_out, "wb+") as f:
            cloudpickle.dump(a, f)

        os.remove(args.file_in)

        print("#" * 20)
        print(f"task finished at {datetime.now():%d/%m/%Y %H:%M:%S}")
        print("#" * 20)
