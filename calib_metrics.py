import math
import numpy as np

def ece_equal_interval(allprobs, allpreds, alllabels):

	probs_of_bins = {}
	preds_of_bins = {}
	labels_of_bins = {}


	for bin in range(1, 11):
		probs_of_bins[bin] = []
		preds_of_bins[bin] = []
		labels_of_bins[bin] = []
			
	for prob, pred, label in zip(allprobs, allpreds, alllabels):
		bin = math.ceil(prob * 10)
		probs_of_bins[bin].append(prob)
		preds_of_bins[bin].append(pred)
		labels_of_bins[bin].append(label)
 
	ECE = 0
	for bin in range(1,11):
		probs = probs_of_bins[bin]
		preds = preds_of_bins[bin]
		labels = labels_of_bins[bin]
		avg_probs = sum([prob for prob in probs]) /len(probs) if len(probs) != 0 else 0
		bin_acc = sum([int(i==j) for i,j in zip(preds, labels)]) / len(probs) if len(probs) != 0 else 0
		ECE += abs(bin_acc-avg_probs) * len(probs)
	
	return ECE / len(allprobs) # [0,1] 


def ece_equal_mass(allprobs, allpreds, alllabels):
	probs_of_bins = {} 
	preds_of_bins = {}
	labels_of_bins = {}

	for bin in range(100):
		probs_of_bins[bin] = []
		preds_of_bins[bin] = []
		labels_of_bins[bin] = []

	# Sort by prob
	data = zip(allprobs, allpreds, alllabels)
	data = sorted(data)
	allprobs, allpreds, alllabels = zip(*data)
	bin_num = 100
	num_samples_per_bin = math.ceil(len(alllabels) / bin_num)

	for i, (prob, pred, label) in enumerate(zip(allprobs, allpreds, alllabels)):
		bin = int(i / num_samples_per_bin)
		probs_of_bins[bin].append(prob)
		preds_of_bins[bin].append(pred)
		labels_of_bins[bin].append(label)

	ECE = 0

	for bin in range(100):
		probs = probs_of_bins[bin]
		preds = preds_of_bins[bin]
		labels = labels_of_bins[bin]

		# Calculate the average predicted probability for this bin
		avg_probs = sum(probs) / len(probs) if len(probs) != 0 else 0
		# if len(probs) != 0:
		# 	total_prob = sum([prob for prob in probs])
		# 	avg_probs = total_prob / len(probs)
		# else:
		# 	avg_probs = 0
			
		# Calculate the accuracy for this bin considering entire predicted tokens
		bin_correct = [int(pred == label) for pred, label in zip(preds, labels)]
		bin_acc = sum(bin_correct) / len(bin_correct) if len(bin_correct) != 0 else 0

		ECE += abs(bin_acc - avg_probs) * len(probs)

	return ECE / len(allprobs)


def compute_ece(allprobs_list, allpreds_list, alllabels_list):

	if not isinstance(allprobs_list[0], list):
		allprobs_list = [allprobs_list]
		allpreds_list = [allpreds_list]
		alllabels_list = [alllabels_list]
	
	acc_list = []
	avg_prob_list = []
	
	ECE_equal_mass_list = []
	ECE_equal_mass_subset_list = {0:[], 1:[]}

	ECE_equal_interval_list = []
	ECE_equal_interval_subset_list = {0:[], 1:[]}


	for allprobs, allpreds, alllabels in zip(allprobs_list, allpreds_list, alllabels_list):

		avg_prob = np.mean(allprobs)
		avg_prob_list.append(avg_prob)
		acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)]) / len(alllabels)
		acc_list.append(acc)

		ECE_equal_mass_subset_list[0].append(np.mean([prob for (prob, pred, label) in zip(allprobs, allpreds, alllabels) if pred != label]))
		ECE_equal_mass_subset_list[1].append(1-np.mean([prob for (prob, pred, label) in zip(allprobs, allpreds, alllabels) if pred == label]))
		ECE_equal_mass = ece_equal_mass(allprobs, allpreds, alllabels)
		ECE_equal_mass_list.append(ECE_equal_mass)

		ECE_equal_interval_subset_list[0].append(np.mean([prob for (prob, pred, label) in zip(allprobs, allpreds, alllabels) if pred != label]))
		ECE_equal_interval_subset_list[1].append(1-np.mean([prob for (prob, pred, label) in zip(allprobs, allpreds, alllabels) if pred == label]))
		# ECE_equal_interval = ece_equal_interval(allprobs, allpreds, alllabels)
		# ECE_equal_interval_list.append(ECE_equal_interval)


	avg_ECE_equal_mass_subset = {0: np.mean(ECE_equal_mass_subset_list[0]),1: np.mean(ECE_equal_mass_subset_list[1])}
	std_ECE_equal_mass_subset = {0: np.std(ECE_equal_mass_subset_list[0]),1: np.std(ECE_equal_mass_subset_list[1])}
	avg_ECE_equal_mass = np.mean(ECE_equal_mass_list)
	std_ECE_equal_mass = np.std(ECE_equal_mass_list)

	avg_ECE_equal_interval_subset = {0: np.mean(ECE_equal_interval_subset_list[0]),1: np.mean(ECE_equal_interval_subset_list[1])}
	std_ECE_equal_interval_subset = {0: np.std(ECE_equal_interval_subset_list[0]),1: np.std(ECE_equal_interval_subset_list[1])}
	# avg_ECE_equal_interval = np.mean(ECE_equal_interval_list)
	# std_ECE_equal_interval = np.std(ECE_equal_interval_list)

	avg_acc = np.mean(acc_list)
	std_acc = np.std(acc_list)
	avg_probs = np.mean(avg_prob_list)
	std_probs = np.std(avg_prob_list)

	print("acc:", avg_acc, std_acc)
	print("avg_probs:", avg_probs, std_probs)
	print("|avg_acc-avg_prob| =", abs(avg_acc - avg_probs))

	# ece_equal_mass
	# for key in range(2):
	# 	print(f"ECE_equal_mass on subsets [{key}]:", avg_ECE_equal_mass_subset[key], std_ECE_equal_mass_subset[key])
	print("ECE_equal_mass: ", avg_ECE_equal_mass, std_ECE_equal_mass)

	output_string = f"avg_probs: {avg_probs} std_probs: {std_probs}\n |avg_acc-avg_prob| {abs(avg_acc - avg_probs)} \n ECE_equal_mass: {avg_ECE_equal_mass} std {std_ECE_equal_mass}"
	return output_string 
