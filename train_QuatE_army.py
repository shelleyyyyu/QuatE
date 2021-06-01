import config
from  models import *
import json
import os
import argparse

def main(args):
    con = config.Config()
    con.set_in_path(args.dataset)
    con.set_work_threads(8)
    con.set_train_times(40000)
    con.set_nbatches(10)
    con.set_alpha(0.1)
    con.set_bern(1)
    con.set_dimension(100)
    con.set_lmbda(0.1)
    con.set_lmbda_two(0.01)
    con.set_margin(1.0)
    con.set_ent_neg_rate(1)
    con.set_rel_neg_rate(0)
    con.set_opt_method("adagrad")
    con.set_save_steps(5000)
    con.set_valid_steps(5000)
    con.set_early_stopping_patience(10)
    CHECKPOINT_DIR = os.path.basename(os.path.dirname(args.dataset))+'_checkpoint'
    con.set_checkpoint_dir(CHECKPOINT_DIR)
    RESULT_DIR = os.path.basename(os.path.dirname(args.dataset))+'_result'
    con.set_result_dir(RESULT_DIR)
    con.set_test_link(True)
    con.set_test_triple(True)
    con.init()
    con.set_train_model(QuatE)
    con.train()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='transE_argument')
	parser.add_argument('--dataset', type=str, default="./benchmarks/army_clean_all_shuffle_small/", help='dataset directory')
	args = parser.parse_args()
	main(args)