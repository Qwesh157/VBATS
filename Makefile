CC=nvcc
TARGET := vbat

CXXFLAGS += 
INCLUDES  += -I./include
LDFLAGS = -lcublas

SRCS := ./main.cu

CUR_OBJS=${SRCS:.cu=.o}

EXECUTABLE=vbat_grouped_gemm

all:$(EXECUTABLE)

$(EXECUTABLE): $(CUR_OBJS)
	$(CC) $(CUR_OBJS) $(LDFLAGS) -o $(EXECUTABLE)
      
	
%.o:%.cu
	$(CC) -c -w $< $(CXXFLAGS) $(INCLUDES) -o $@
	
	
clean:
	rm -f $(EXECUTABLE)
	rm -f ./src/*.o
	rm -f ./*.o
	rm -f result.m