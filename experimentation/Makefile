IN = $(wildcard data/inputs/*.json)
IN2 = $(addsuffix .pdf,$(basename $(IN)))
OUT = $(subst data/inputs/,data/outputs/,$(IN2))
CLI = python experimentation/cli.py

create_specs:
	$(CLI) create-specs

data/outputs/%.pdf: data/inputs/%.json
	$(CLI) run $<

collect: 
	$(CLI) collect-parquets


run_all: 
	$(CLI) run-thread all
	$(CLI) collect-parquets

R:
	Rscript R_scripts/run.R 

all:  create_specs run_all 
	Rscript R_script/run.R 

run_all_fast: 
	$(CLI) run-thread all_fast
	$(CLI) collect-parquets

run_test: 
	$(CLI) run-thread test
	$(CLI) collect-parquets

clean:
	rm data/inputs/*.json
