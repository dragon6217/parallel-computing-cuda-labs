# HT Test
run: HTtest
    # Run 'base' (coarse-lock) with 16 threads and reduced workload
	./HTtest 5000 16 200000 10 0
    # Run 'better' (fine-lock) with 16 threads and reduced workload
	./HTtest 5000 16 200000 10 1

HTtest: main.cc 
	g++ -std=c++11 -g -pthread  $< -o $@ -lboost_system  -lboost_thread

clean:
	rm -f HTtest *.err *.log *.out

# Condor
remote_base: HTtest
	condor_submit HW2_base.cmd

remote_better: HTtest
	condor_submit HW2_better.cmd

report:
	cat base.out
	cat better.out

queue:
	condor_q

status:
	condor_status

remove:
	condor_rm

# Development
format:
	clang-format -i -style=Google *.cc *.h
