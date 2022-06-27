from __future__ import print_function
import os
import os.path as osp
import argparse
import sys
import h5py
import time
import datetime
import numpy as np
import glob
import pandas as pd
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli

from utils import Logger, read_json, write_json, save_checkpoint, init_expr_path
from models import *
from rewards import compute_reward
import vsum_tools

parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")
# Dataset options
parser.add_argument('--expr', type=str, required=True, choices = ['exp1', 'exp2', 'exp3'], help="experiments type : ['exp1', 'exp2', 'exp3']")
# Model options
parser.add_argument('--input-dim', type=int, default=1024, help="input dimension (default: 1024)")
parser.add_argument('--hidden-dim', type=int, default=256, help="hidden unit dimension of DSN (default: 256)")
parser.add_argument('--num-layers', type=int, default=1, help="number of RNN layers (default: 1)")
parser.add_argument('--rnn-cell', type=str, default='lstm', help="RNN cell type (default: lstm)")
# Optimization options
parser.add_argument('--lr', type=float, default=1e-05, help="learning rate (default: 1e-05)")
parser.add_argument('--weight-decay', type=float, default=1e-05, help="weight decay rate (default: 1e-05)")
parser.add_argument('--max-epoch', type=int, default=60, help="maximum epoch for training (default: 60)")
parser.add_argument('--stepsize', type=int, default=30, help="how many steps to decay learning rate (default: 30)")
parser.add_argument('--gamma', type=float, default=0.1, help="learning rate decay (default: 0.1)")
parser.add_argument('--num-episode', type=int, default=5, help="number of episodes (default: 5)")
parser.add_argument('--beta', type=float, default=0.01, help="weight for summary length penalty term (default: 0.01)")
# Misc
parser.add_argument('--seed', type=int, default=12345, help="random seed (default: 1)")
parser.add_argument('--gpu', type=str, default='0', help="which gpu devices to use")
parser.add_argument('--use-cpu', action='store_true', help="use cpu device")
parser.add_argument('--evaluate', action='store_true', help="whether to do evaluation only")
parser.add_argument('--resume', type=str, default='', help="path to resume file")
parser.add_argument('--verbose', action='store_true', help="whether to show detailed test results")
# parser.add_argument('--save-results', action='store_true', help="whether to save output results")

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_gpu = torch.cuda.is_available()
if args.use_cpu: use_gpu = False

def main():
    """
    log : /{expr_name}/{data_file}/log_*.txt
    """

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")
    
    base_dir, datasets = init_expr_path(args.expr)
    if args.verbose: table = [["No.", "Video", "F-score"]]

    for i, file_name in enumerate(datasets):
        h5_path = os.path.join(base_dir, f'{file_name}.h5')
        save_dir = f'{args.expr}/{file_name}'
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'model'), exist_ok=True)
        
        if not args.evaluate:
            # sys.stdout = Logger(osp.join(save_dir, 'log_train.txt'))
            sys.stdout = Logger(osp.join(save_dir, 'log_train.txt'))
        else:
            sys.stdout = Logger(osp.join(save_dir, 'log_test.txt'))
        print("==========\nArgs:{}\n==========".format(args))
        

        print("Initialize dataset : {}".format(file_name))
        dataset = h5py.File(h5_path, 'r')
        assert dataset is not None, "dataset is None!"

        num_videos = len(dataset.keys())
        if 'tvsum' in file_name:
            dataset_type = 'tvsum'
            dataset_json = '/data/project/rw/video_summarization/dataset/tvsum_splits.json'
        elif 'summe' in file_name:
            dataset_type = 'summe'
            dataset_json = '/data/project/rw/video_summarization/dataset/summe_splits.json'
        else:
            raise NotImplementedError()

        json_data = read_json(dataset_json)
        train_keys = json_data['train_keys']
        test_keys = json_data['test_keys']
        print("# total videos {}. # train videos {}. # test videos {}".format(num_videos, len(train_keys), len(test_keys)))


        print("Initialize model")
        model = DSN(in_dim=args.input_dim, hid_dim=args.hidden_dim, num_layers=args.num_layers, cell=args.rnn_cell)
        print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.stepsize > 0:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

        if args.resume:
            print("Loading checkpoint from '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint)
        else:
            start_epoch = 0

        if use_gpu:
            # model = nn.DataParallel(model).cuda()
            model = model.cuda()

        if args.evaluate:
            print("Evaluate only")
            weights_filepath = f'{save_dir}/best_model*.tar.pth'
            weights_filename = glob.glob(weights_filepath)
            assert len(weights_filename) != 0
            weights_filename = weights_filename[0]
            print("Loading model:", weights_filename)
            model.load_state_dict(torch.load(weights_filename, map_location=lambda storage, loc: storage))
            val_fscore, df = evaluate(model, dataset, test_keys, use_gpu, dataset_type)

            if args.verbose:
                table.append([i+1, file_name, "{:.1%}".format(val_fscore)])
                # print(tabulate(table))
            df.to_csv(f'{save_dir}/best_epoch_results.csv', index=False)
            dataset.close()
            continue

        print("==> Start training")
        start_time = time.time()
        baselines = {key: 0. for key in train_keys} # baseline rewards for videos
        reward_writers = {key: [] for key in train_keys} # record reward changes for each video
        max_val_fscore = 0
        max_val_fscore_epoch = 0

        for epoch in range(start_epoch, args.max_epoch):
            model.train()
            idxs = np.arange(len(train_keys))
            np.random.shuffle(idxs) # shuffle indices

            for idx in idxs:
                key = train_keys[idx]
                seq = dataset[key]['features'][...] # sequence of features, (seq_len, dim)
                seq = torch.from_numpy(seq).unsqueeze(0) # input shape (1, seq_len, dim)
                if use_gpu: seq = seq.cuda()
                probs = model(seq) # output shape (1, seq_len, 1)

                cost = args.beta * (probs.mean() - 0.5)**2 # minimize summary length penalty term [Eq.11]
                m = Bernoulli(probs)
                epis_rewards = []
                for _ in range(args.num_episode):
                    actions = m.sample()
                    log_probs = m.log_prob(actions)
                    reward = compute_reward(seq, actions, use_gpu=use_gpu)
                    expected_reward = log_probs.mean() * (reward - baselines[key])
                    cost -= expected_reward # minimize negative expected reward
                    epis_rewards.append(reward.item())

                optimizer.zero_grad()
                cost.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                baselines[key] = 0.9 * baselines[key] + 0.1 * np.mean(epis_rewards) # update baseline reward via moving average
                reward_writers[key].append(np.mean(epis_rewards))

            epoch_reward = np.mean([reward_writers[key][epoch] for key in train_keys])
            print("epoch {}/{}\t reward {}\t".format(epoch+1, args.max_epoch, epoch_reward))

            val_fscore = evaluate(model, dataset, test_keys, use_gpu, dataset_type)

            if max_val_fscore < val_fscore:
                max_val_fscore = val_fscore
                max_val_fscore_epoch = epoch
            print('   Test F-score avg/max: {0:0.5}/{1:0.5}'.format(val_fscore, max_val_fscore))

            model_state_dict = model.state_dict()
            model_save_path = osp.join(save_dir, 'model', f'epoch{epoch}_{val_fscore}.pth.tar')
            save_checkpoint(model_state_dict, model_save_path)
            print("Model saved to {}".format(model_save_path))
            
        write_json(reward_writers, osp.join(save_dir, f'rewards.json'))

        elapsed = round(time.time() - start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

        print(f"The best F-score model\t F-score : {max_val_fscore} @{max_val_fscore_epoch}")
        os.system(f'cp {save_dir}/model/epoch{max_val_fscore_epoch}_*.pth.tar {save_dir}/best_model_{max_val_fscore_epoch}_{round(max_val_fscore*100,3)}.tar.pth')

        dataset.close()
    
    if args.verbose:
        print(tabulate(table))


def evaluate(model, dataset, test_keys, use_gpu, dataset_type):
    print("==> Test")
    with torch.no_grad():
        model.eval()
        fms = []
        eval_metric = 'avg' if dataset_type == 'tvsum' else 'max'

        # if args.save_results:
            # h5_res = h5py.File(osp.join(save_dir, 'result.h5'), 'w')

        if args.evaluate:
            df = pd.DataFrame()
        
        for key_idx, key in enumerate(test_keys):
            f_scores = []
            seq = dataset[key]['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)
            if use_gpu: 
                seq = seq.cuda()
            probs = model(seq)
            probs = probs.data.cpu().squeeze().numpy()

            cps = dataset[key]['change_points'][...]
            num_frames = dataset[key]['n_frames'][()]
            nfps = dataset[key]['n_frame_per_seg'][...].tolist()
            positions = dataset[key]['picks'][...]
            user_summary = dataset[key]['user_summary'][...]
            sum_ratio = dataset[key]['sum_ratio'][...]
            video_boundary = dataset[key]['video_boundary'][...]

            for user_id in range(user_summary.shape[0]):
                machine_summary = vsum_tools.generate_summary(probs, cps, num_frames, nfps, positions, proportion=sum_ratio[user_id])
                fm, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary[user_id], eval_metric)
                f_scores.append(fm)
                if args.evaluate:
                    coverage = vsum_tools.coverage_count(key, user_id, machine_summary, user_summary[user_id], video_boundary, sum_ratio[user_id])
                    df = df.append(coverage, ignore_index=True)
            
            if eval_metric == 'avg':
                final_f_score = np.mean(f_scores)
            elif eval_metric == 'max':
                final_f_score = np.max(f_scores)
            fms.append(final_f_score)

            # if args.verbose:
            #     table.append([key_idx+1, key, "{:.1%}".format(final_f_score)])

            # if args.save_results:
            #     h5_res.create_dataset(key + '/score', data=probs)
            #     h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
            #     h5_res.create_dataset(key + '/gtscore', data=dataset[key]['gtscore'][...])
            #     h5_res.create_dataset(key + '/fm', data=fm)

    # if args.save_results: h5_res.close()
    mean_fm = np.mean(fms)
    print("Average F-score {:.1%}".format(mean_fm))

    if not args.evaluate:
        return mean_fm
    else:
        df = df[['video_id', 'user_id', 'sum_ratio', 'v1_frames', 'v1_pred_frames', 'v1_gt_frames', 'v1_n_overlap', 'v1_overlap_ratio', 'v1_pred_sum_ratio', 'v1_gt_sum_ratio',\
                'v2_frames', 'v2_pred_frames', 'v2_gt_frames', 'v2_n_overlap', 'v2_overlap_ratio', 'v2_pred_sum_ratio', 'v2_gt_sum_ratio',\
                'v3_frames', 'v3_pred_frames', 'v3_gt_frames', 'v3_n_overlap', 'v3_overlap_ratio', 'v3_pred_sum_ratio', 'v3_gt_sum_ratio',\
                'v4_frames', 'v4_pred_frames', 'v4_gt_frames', 'v4_n_overlap', 'v4_overlap_ratio', 'v4_pred_sum_ratio', 'v4_gt_sum_ratio']]
        return mean_fm, df

if __name__ == '__main__':
    main()
