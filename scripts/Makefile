CWD:=$(strip $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST)))))
MODEL_DIR:=model_files

models:=tabular

stems:= \
	$(models) \
	$(addsuffix _data, $(models)) \
	$(addsuffix _oodd, $(models)) \
	$(addsuffix _data_oodd, $(models))

targets:= \
	$(stems) \
	$(addsuffix _default, $(stems)) \
	$(addsuffix _plotly, $(stems))

all: $(targets)

clean:
	rm -rf $(MODEL_DIR)
	find .. -maxdepth 3 -name "__pycache__" -exec rm -rf {} \;

$(MODEL_DIR):
	mkdir -p $(MODEL_DIR)

# this is so that make sees when the relevant model files are up to date
$(targets): %: $(MODEL_DIR)/%

# the | is important - without it the all target doesn't correctly ignore
# targets which were run already
$(MODEL_DIR)/%: | $(MODEL_DIR)
	PYTHONPATH=.. python main.py $@

	# from previous directory, to check that everything works when
	# local module directories are not visible
	cd .. && python $(CWD)/test.py $(CWD)/$@
