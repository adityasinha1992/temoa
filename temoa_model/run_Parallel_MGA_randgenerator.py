from mpi4py import MPI
import time
from Parallel_MGA_randgenerator_6 import parse_args, pipeline

number_of_MGA_iterations = 200
input_file = 'data_files/US_9R_TS_NZ_trunc_4periods_EFfix_updated.dat'
output_file = 'data_files/MGA_DBs_9RTS_EFfix_slack5/MGA_randgenerator.db'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

njobs = number_of_MGA_iterations
# could also be something like len(arglist) if arglist is specfied prior to making connections to the nodes. 
# See Lucas' example run file 'run_tclr_model.py' for this.

my_jobs = list(range(rank, njobs, nprocs))

print("my rank is", rank)
print("number of processors is", nprocs)
print("my_jobs looks like", my_jobs)

arglist_to_pass = ['-i', input_file, '-o', output_file, '-n', str(len(my_jobs)), '-c', str(my_jobs)]
# my_jobs_str = [str(_my_job) for _my_job in my_jobs]
# arglist_to_pass = arglist_to_pass + my_jobs_str

print(arglist_to_pass)

start_time = time.time()
args = parse_args(arglist_to_pass)
pipeline(args)

print('the time for the job is', time.time() - start_time) # update time function with perf counter

# for job in my_jobs:
#     print('job:', job)
#     print('rank', rank)
#     args = parse_args(arglist[job])
#     pipeline(args)


# # NFS SQLITE WORKAROUND
# tmp_db_file = "<master database file here>"
# # sqlite and NFS dont play well.
# # so when running in an environment with an NFS file system,
# # it is best to make a local copy of the database for the model run
# # and then copy it back to the main storage system afterwards.
# # It is best to tag it with the user name in case others are doing this too.
# user_name = os.getlogin()
# # make dir if it does not exist
# if not os.path.isdir(f"/tmp/{user_name}_tempdb"):
#     os.mkdir(f"/tmp/{user_name}_tempdb")

# # when using a temp file for the nfs issues, use the line directly below this one
# # i usually use a scenario name for <temp_database_name>
# db_file = f"/tmp/{user_name}_tempdb/<temp_database_name>.sqlite"

# # if using a tmp db, this command copies the current database to the temp location
# shutil.copy(
#     tmp_db_file,
#     db_file
# )

# ##### RUNNING THE MODEL HERE

# # after you finished your model run, you should move the file back to the main storage system
# shutil.move(
# 	db_file,
# 	final_db_location,
# )

