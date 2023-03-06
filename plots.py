import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from random import random


def read_data_file():
	df = pd.read_json('datapoints_bald.json')
	plot_data_list = []
	dataset_names = df["dataset"].unique()
	active_learners = df["active_learner"].unique()

	for dataset in dataset_names:
		dataset_df = df.loc[df["dataset"] == dataset]
		for active_learner in active_learners:
			active_learner_df = dataset_df.loc[dataset_df["active_learner"] == active_learner]
			for al_cycles in active_learner_df["al_cycles"]:
				active_learner_df_cycle = active_learner_df.loc[active_learner_df["al_cycles"] == al_cycles]
				num_experiments = active_learner_df_cycle["experiment_num"].unique()
				try:
					num_experiments_max = np.amax(num_experiments)
				except ValueError:
					num_experiments_max = 0
				num_examples = active_learner_df_cycle["num_examples"].iloc[0]
				mean_accuracy = np.mean(np.array(active_learner_df_cycle["acc"]), axis=0)
				datapoint = {
					'dataset': dataset,
					'active_learner': active_learner,
					'experiment_num': num_experiments_max,
					'al_cycles': al_cycles,
					'mean_accuracy': mean_accuracy,
					'num_examples': num_examples
				}
				plot_data_list.append(datapoint)
	return plot_data_list


def plot_data(plot_data_list):
	plot_data_df = pd.DataFrame(plot_data_list)
	dataset_names = plot_data_df["dataset"].unique()
	active_learners = plot_data_df["active_learner"].unique()

	for dataset in dataset_names:
		dataset_df = plot_data_df.loc[plot_data_df["dataset"] == dataset]
		for al_cycle in dataset_df["al_cycles"]:
			plt.clf()
			active_learner_df_cycle = dataset_df.loc[dataset_df["al_cycles"] == al_cycle]
			for active_learner in active_learners:
				active_learner_df = active_learner_df_cycle.loc[active_learner_df_cycle["active_learner"] == active_learner]
				num_examples = active_learner_df["num_examples"].iloc[0]
				accuracy = active_learner_df["mean_accuracy"].iloc[0]
				plt.plot(num_examples, accuracy, label=(dataset + " for " + active_learner))
			plt.ylabel('Accuracy')
			plt.xlabel(r'Number of examples')
			plt.legend(loc='lower right', borderaxespad=0.)
			# plt.show()
			if not os.path.exists("plots"):
				os.makedirs("plots")
			plt.savefig("./plots/" + dataset + str(al_cycle) + ".png")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
	plot_data_list = read_data_file()
	plot_data(plot_data_list)
