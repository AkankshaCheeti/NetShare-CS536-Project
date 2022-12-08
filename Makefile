export SUDO 						= echo $(USER_PASSWORD) | sudo -S

export PROJECT_DIR 					= $(PWD)
export SCRIPTS_DIR					= $(PROJECT_DIR)/scripts
export SOURCE_DIR					= $(PROJECT_DIR)/src
export EVAL_SOURCE_DIR				= $(PROJECT_DIR)/eval
export PREPROCESSING_DIR			= $(PROJECT_DIR)/preprocess
export RESULTS_DIR 					= $(PROJECT_DIR)/results
export BACKUP_RESULTS_DIR 			= $(PROJECT_DIR)/backup_results

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

########################################################################################
################################### Preprocessing ######################################
########################################################################################

preprocess-no-dp:
	cd $(PREPROCESSING_DIR) && bash run_no_privacy.sh

preprocess-no-dp-botnet-dataset:
	cd $(PREPROCESSING_DIR) && bash run_no_privacy_new_dataset.sh

preprocess-with-dp:
	cd $(PREPROCESSING_DIR) && bash run_privacy.sh

########################################################################################
################################### Train Models #######################################
########################################################################################

train-no-dp:
	cd $(SOURCE_DIR) && $(PYTHON) main.py --root_user $(ROOT_USER) \
		--config_file config_test1_pcap_no_dp \
		--measurer_file measurers_localhost.ini --measurement

train-botnet-benign-no-dp:
	cd $(SOURCE_DIR) && $(PYTHON) main.py --root_user $(ROOT_USER) \
		--config_file config_botnet_benign_test_pcap_no_dp \
		--measurer_file measurers_localhost.ini --measurement

train-botnet-malicious-no-dp:
	cd $(SOURCE_DIR) && $(PYTHON) main.py --root_user $(ROOT_USER) \
		--config_file config_botnet_malicious_test_pcap_no_dp \
		--measurer_file measurers_localhost.ini --measurement

# docker-train-no-dp:
# 	$(PYTHON_DOCKER) "cd $(SOURCE_DIR) && $(PYTHON) main.py --root_user $(ROOT_USER) \
# 		--config_file config_test1_pcap_no_dp \
# 		--measurer_file measurers_localhost.ini --measurement"

########################################################################################
################################## Generate Flows ######################################
########################################################################################

generate-botnet-benign-no-dp:
	cd $(SOURCE_DIR) && $(PYTHON) main.py --root_user $(ROOT_USER) \
		--config_file config_botnet_benign_test_pcap_no_dp \
		--measurer_file measurers_localhost.ini --generation

generate-botnet-malicious-no-dp:
	cd $(SOURCE_DIR) && $(PYTHON) main.py --root_user $(ROOT_USER) \
		--config_file config_botnet_malicious_test_pcap_no_dp \
		--measurer_file measurers_localhost.ini --generation

generate-no-dp:
	cd $(SOURCE_DIR) && $(PYTHON) main.py --root_user $(ROOT_USER) \
		--config_file config_test1_pcap_no_dp \
		--measurer_file measurers_localhost.ini --generation

# docker-generate-no-dp:
# 	$(PYTHON_DOCKER) "cd $(SOURCE_DIR) && $(PYTHON) main.py --root_user $(ROOT_USER) \
# 		--config_file config_test1_pcap_no_dp \
# 		--measurer_file measurers_localhost.ini --generation"

########################################################################################
################################# Generate Results #####################################
########################################################################################

######################################## CDF ###########################################

generate-caida-cdf:
	cd $(EVAL_SOURCE_DIR)/fidelity && $(PYTHON) plot_cdf.py \
		--type PCAP --dataset $(BACKUP_RESULTS_DIR)/caida/ \
		--results $(BACKUP_RESULTS_DIR)/plots/caida

generate-ugr16-cdf:
	cd $(EVAL_SOURCE_DIR)/fidelity && $(PYTHON) plot_cdf.py \
		--type NETFLOW --dataset $(BACKUP_RESULTS_DIR)/ugr16/ \
		--results $(BACKUP_RESULTS_DIR)/plots/ugr16

generate-botnet-malicious-cdf:
	cd $(EVAL_SOURCE_DIR)/fidelity && $(PYTHON) plot_cdf.py \
		--type PCAP --dataset $(BACKUP_RESULTS_DIR)/botnet-malicious/ \
		--results $(BACKUP_RESULTS_DIR)/plots/botnet-malicious

cdf: generate-caida-cdf generate-ugr16-cdf generate-botnet-malicious-cdf

#################################### Bar Plots ########################################

generate-ugr16-barplot:
	cd $(EVAL_SOURCE_DIR)/fidelity && $(PYTHON) plot_bar_plot.py \
		--method run_netflow_qualitative_plots \
		--dataset $(BACKUP_RESULTS_DIR)/ugr16/ \
		--results $(BACKUP_RESULTS_DIR)/plots/ugr16

barplots: generate-ugr16-barplot

##################################### Fidelity ########################################

generate-caida-fidelity:
	cd $(EVAL_SOURCE_DIR)/fidelity && $(PYTHON) plot_bar_plot.py \
		--method run_pcap_dist_metrics \
		--dataset $(BACKUP_RESULTS_DIR)/caida/ \
		--results $(BACKUP_RESULTS_DIR)/plots/caida

generate-botnet-malicious-fidelity:
	cd $(EVAL_SOURCE_DIR)/fidelity && $(PYTHON) plot_bar_plot.py \
		--method run_pcap_dist_metrics \
		--dataset $(BACKUP_RESULTS_DIR)/botnet-malicious/ \
		--results $(BACKUP_RESULTS_DIR)/plots/botnet-malicious

generate-ugr16-fidelity:
	cd $(EVAL_SOURCE_DIR)/fidelity && $(PYTHON) plot_bar_plot.py \
		--method run_netflow_dist_metrics \
		--dataset $(BACKUP_RESULTS_DIR)/ugr16/ \
		--results $(BACKUP_RESULTS_DIR)/plots/ugr16

fidelity: generate-caida-fidelity generate-botnet-malicious-fidelity generate-ugr16-fidelity

################################# Count-Min Sketch ####################################

HASH						?= csiphash # mmh3, horner
WIDTH_SCALE					?= 0.1
DEPTH						?= 5
PERCENTILE					?= 0.1

generate-caida-cms:
	cd $(EVAL_SOURCE_DIR)/countminsketch && $(PYTHON) countminsketch.py \
		--dataset $(BACKUP_RESULTS_DIR)/caida/ \
		--keys srcip --hash $(HASH) \
		--width_scale $(WIDTH_SCALE) --depth $(DEPTH) --percentile $(PERCENTILE)

# generate-botnet-malicious-cms:
# 	cd $(EVAL_SOURCE_DIR)/countminsketch && $(PYTHON) countminsketch.py \
# 		--dataset $(BACKUP_RESULTS_DIR)/botnet-malicious/ \
# 		--keys srcip dstip srcport dstport proto --hash $(HASH) \
# 		--width 10000 --depth 5 --percentile $(PERCENTILE)

################################## ML Evaluations #####################################

RUNS						?= 

anomaly-ugr16:
	cd $(EVAL_SOURCE_DIR)/anomalydetection && $(PYTHON) ugr16_anomaly.py \
		--dataset $(BACKUP_RESULTS_DIR)/ugr16/ --runs $(RUNS)

anomaly-botnet:
	cd $(EVAL_SOURCE_DIR)/anomalydetection && $(PYTHON) botnet_anomaly.py \
		--dataset $(BACKUP_RESULTS_DIR)/botnet/ --runs $(RUNS)

#################################### All Plots ########################################

plots: cdf barplots fidelity

########################################################################################
############################## Clean Synthetic Flows ###################################
########################################################################################

clean-results:
	-rm -rf $(RESULTS_DIR)

clean-plots:
	-rm -rf $(BACKUP_RESULTS_DIR)/plots/*

clean: clean-plots # clean-results
