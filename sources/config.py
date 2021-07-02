from os.path import join

RESOURCE_PATH = './resources'
CHECKPOINT_FOLDER = join(RESOURCE_PATH, 'checkpoint')
CHECKPOINT_PATTERN = join(RESOURCE_PATH, 'checkpoint/{}_{}_{}_{}.pt')  # nll, auc, acc_score
