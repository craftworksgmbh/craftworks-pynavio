CWD:=$(strip $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST)))))
MODEL_DIR:=model_files

# add model here if it can support have any combination of:
# data, oodd, plotly or default explanations
models:=image minimal tabular timeseries

# add model here if it can support have any combination of:
# data, oodd, default explanations
models_no_plotly:=car_price_model pump_leakage_model

stems_no_plotly:= \
	$(models_no_plotly) \
	$(addsuffix _data, $(models_no_plotly)) \
	$(addsuffix _oodd, $(models_no_plotly)) \
	$(addsuffix _data_oodd, $(models_no_plotly))

stems:= \
	$(models) \
	$(addsuffix _data, $(models)) \
	$(addsuffix _oodd, $(models)) \
	$(addsuffix _data_oodd, $(models))

# add model here if you want specific features enabled only
targets:= \
	$(stems) \
	$(addsuffix _default, $(stems)) \
	$(addsuffix _plotly, $(stems)) \
	$(stems_no_plotly) \
	$(addsuffix _default, $(stems_no_plotly)) \
	additional_fields \
	error_model \
	image_pytorch \
	image_tensorflow \
	image_onnx \
	minimal_boolean \
	minimal_trainer \
	timeseries_trainer \
	type_demo_data_oodd \
	visual_inspection_model

all: $(targets)

clean:
	rm -rf $(MODEL_DIR)
	find . -maxdepth 3 -name "__pycache__" -exec rm -rf {} \;

$(MODEL_DIR):
	mkdir -p $(MODEL_DIR)

# this is so that make sees when the relevant model files are up to date
$(targets): %: $(MODEL_DIR)/%

# the | is important - without it the all target doesn't correctly ignore
# targets which were run already
$(MODEL_DIR)/%: | $(MODEL_DIR)
	PYTHONPATH=. python scripts/main.py $@

	# from previous directory, to check that everything works when
	# local module directories are not visible
	cd .. && python $(CWD)/scripts/test.py $(CWD)/$@
