PYX_FILES = $(shell find cief -type f -name '*.pyx')
C_FILES = $(subst .pyx,.c,$(PYX_FILES))

%.c: %.pyx
	cythonize -X language_level=3 -a -i $<

clean:
	rm $(C_FILES)

all: $(C_FILES)

