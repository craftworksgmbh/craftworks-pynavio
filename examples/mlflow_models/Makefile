CWD:=$(strip $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST)))))
MODEL_DIR:=model_files

misc:=images background explanations

models:=minimal tabular image timeseries

stems:= \
	$(models) \
	$(addsuffix _data, $(models)) \
	$(addsuffix _oodd, $(models)) \
	$(addsuffix _data_oodd, $(models))

targets:= \
	$(stems) \
	$(addsuffix _default, $(stems)) \
	$(addsuffix _plotly, $(stems)) \
	trainer \
	error_model \
	additional_fields

all: $(targets)

clean:
	rm -rf $(MODEL_DIR) $(misc)
	find . -maxdepth 2 -name "__pycache__" -exec rm -rf {} \;

$(MODEL_DIR):
	mkdir -p $(MODEL_DIR)

# this is so that make sees when the relevant model files are up to date
$(targets): %: $(MODEL_DIR)/%

# the | is important - without it the all target doesn't correctly ignore
# targets which were run already
$(MODEL_DIR)/%: | $(MODEL_DIR)
	python main.py $@

	# from previous directory, to check that everything works when
	# local module directories are not visible
	cd .. && python $(CWD)/../../utils/test.py $(CWD)/$@

code-style:
	python -m isort main.py models/ utils/
	python -m yapf -rip main.py models/ utils/

