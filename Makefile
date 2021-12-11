NVCC=nvcc
CUDAFLAGS=


heat: heat.o
	$(NVCC) -o $@ $^

heat.o: main.cu
	$(NVCC) $(CUDAFLAGS) -c -o $@ $<

clean:
	rm -rf archivos/*.txt
	rm -rf heat.*
	rm -rf heat