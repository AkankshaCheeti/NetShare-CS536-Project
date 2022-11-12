export PROJECT_DIR 			= $(PWD)
export SOURCE_DIR			= $(PROJECT_DIR)/src
export PREPROCESSING_DIR	= $(PROJECT_DIR)/
export RESULTS_DIR 			= $(PROJECT_DIR)/results

export ROOT_USER 			= annusmanarchitect
export PYTHON 				= python3

.DEFAULT_GOAL				:= train-no-dp

preprocess-no-dp:
	cd $(PREPROCESSING_DIR) && bash run_no_privacy.sh

preprocess-with-dp:
	cd $(PREPROCESSING_DIR) && bash run_privacy.sh

train-no-dp:
	cd $(SOURCE_DIR) && $(PYTHON) main.py --root_user $(ROOT_USER) --config_file config_test1_pcap_no_dp --measurer_file measurers_localhost.ini --measurement

generate-no-dp:
	cd $(SOURCE_DIR) && $(PYTHON) main.py --root_user $(ROOT_USER) --config_file config_test1_pcap_no_dp --measurer_file measurers_localhost.ini --generation

clean-results:
	rm -rf $(RESULTS_DIR)

clean: clean-results