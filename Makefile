CC = gcc
CFLAGS = -O3 -fPIC -fopenmp -Wall
LDFLAGS = -shared -fopenmp

TARGET = libsolver.so
SRCS = src/c/solver.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ -lm

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)
