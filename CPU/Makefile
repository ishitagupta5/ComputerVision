CLAGS = -02 -g -Wall
LDFLAGS = -lm

programs = sobel

all: $(programs)


sobel: sobel_filter.c 
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	-rm -f $(programs)