#########################
# Environment variable
#########################
build=build
src=src
GPP=g++ -O3 -std=c++11 -pthread -lpthread -fopenmp -Wl,--no-as-needed

#########################
# Run
#########################
jointable: jointable_o $(build)/driver.o
	$(GPP) -o jointable \
		$(build)/driver.o \
		$(build)/jointable.o

jointable_o:
	mkdir -p $(build)
	$(GPP) -c $(src)/jointable.cpp -o $(build)/jointable.o

# You don't need to run this
# $(build)/driver.o is provided
driver_o: jointable_o
	$(GPP) -c $(src)/driver.cpp -o $(build)/driver.o

#########################
# Generate
#########################
generate: gene_o
	$(GPP) -o generate \
		$(build)/generate.o

gene_o:
	$(GPP) -c $(src)/generate.cpp -o $(build)/generate.o

#########################
# Utils
#########################
clean:
	rm -f $(build)/jointable.o $(build)/generate.o jointable generate

format:
	clang-format -i -style=Google $(src)/*.cpp $(src)/*.h

#########################
# Submit
#########################
submit_same:
	mkdir -p result
	condor_submit jointable_same.cmd

submit_diff:
	mkdir -p result
	condor_submit jointable_diff.cmd