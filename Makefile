export SUDO 						= echo $(USER_PASSWORD) | sudo -S

export PROJECT_DIR 					= $(PWD)
export SCRIPTS_DIR					= $(PROJECT_DIR)/scripts
export SOURCE_DIR					= $(PROJECT_DIR)/src
export PREPROCESSING_DIR			= $(PROJECT_DIR)/preprocess
export RESULTS_DIR 					= $(PROJECT_DIR)/results

export CONTAINER_WORK_DIR		 	= /workdir
export CONTAINER_SCRIPTS_DIR		= $(CONTAINER_WORK_DIR)/scripts
export CONTAINER_SOURCE_DIR			= $(CONTAINER_WORK_DIR)/src
export CONTAINER_PREPROCESSING_DIR	= $(CONTAINER_WORK_DIR)/preprocess
export CONTAINER_RESULTS_DIR 		= $(CONTAINER_WORK_DIR)/results

export ROOT_USER 					= annusmanarchitect
export PYTHON 						= python3
export PYTHON_DOCKER				= $(SCRIPTS_DIR)/python3.6

.DEFAULT_GOAL						:= train-no-dp

install-docker:
	cd $(SCRIPTS_DIR) && $(SUDO) bash install-docker.sh

preprocess-no-dp:
	cd $(PREPROCESSING_DIR) && bash run_no_privacy.sh

preprocess-with-dp:
	cd $(PREPROCESSING_DIR) && bash run_privacy.sh

train-no-dp:
	cd $(SOURCE_DIR) && $(PYTHON) main.py --root_user $(ROOT_USER) \
		--config_file config_test1_pcap_no_dp \
		--measurer_file measurers_localhost.ini --measurement

generate-no-dp:
	cd $(SOURCE_DIR) && $(PYTHON) main.py --root_user $(ROOT_USER) \
		--config_file config_test1_pcap_no_dp \
		--measurer_file measurers_localhost.ini --generation

# docker-train-no-dp:
# 	$(PYTHON_DOCKER) "cd $(SOURCE_DIR) && $(PYTHON) main.py --root_user $(ROOT_USER) \
# 		--config_file config_test1_pcap_no_dp \
# 		--measurer_file measurers_localhost.ini --measurement"

# docker-generate-no-dp:
# 	$(PYTHON_DOCKER) "cd $(SOURCE_DIR) && $(PYTHON) main.py --root_user $(ROOT_USER) \
# 		--config_file config_test1_pcap_no_dp \
# 		--measurer_file measurers_localhost.ini --generation"

clean-results:
	rm -rf $(RESULTS_DIR)

clean: clean-results