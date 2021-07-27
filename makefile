CC := nvcc
CFLAGS := -O3 -lcublas -lblas -llapack --gpu-architecture=sm_50

objects    = cuda_subs.o CEED_phon_subs.o CEED_phon_main.o
executable = program.out

$(executable): $(objects)
			$(CC) -o $@ $^ ${CFLAGS}

cuda_subs.o: cuda_subs.cu cuda_subs.h
			$(CC) -o $@ -c $< ${CFLAGS}

CEED_phon_subs.o: CEED_phon_subs.cpp CEED_phon_subs.h
			$(CC) -o $@ -c $< ${CFLAGS}

CEED_phon_main.o: CEED_phon_main.cpp
			$(CC) -o $@ -c $< ${CFLAGS}

.PHONY: clean
clean:
	      rm $(objects) $(executable)
