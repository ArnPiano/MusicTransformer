import torch

load_model = False                      # has the model to be loaded from a previous checkpoint?

epochs = 1000                           # number of epochs
rate = .1                               # dropout rate
label_smoothing = .1                    # label smoothing parameter in the cross entropy loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

B = 2                                   # Batch Size
L = 1024                                # Maximum Length of the model
D = 512                                 # Dimension of the embedding and the model
H = 8                                   # Number of heads in each relative attention
N = 6                                   # Number of Transformer models stacked
d = D//2                                 # Dimension of fully connected layer in single Embedding Layers

pad_token = 388                         # pad token
sos_token = 389                         # SOS token
eos_token = 390                         # EOS token
vocab_size = 391                        # possible outputs

eval_frequency = 100                    # How frequent is the evaluation of the model (in batches)?
ckp_frequency = 50                      # After how many epochs is a stable checkpoint created?


DATASET_DIR = 'midi_in'                 # DIRECTORY OF THE DATASET'S MIDI FILES
RUN_DIR = 'test_run\\'
pickle_dir = 'pickle_dir'                 # Where the pickle data is stored
log_dir = RUN_DIR+'training_summary'    # where the log fo the training is stored
midi_out_dir = RUN_DIR+'midi_out'       # Where the generated midi is stored
model_dir= RUN_DIR+'models\\'           # Where the model's checkpoint are stored

stop_and_go = 'stop_and_go.pth.tar'     # Name of the checkpoint that's needed to stop the training and restart
best_model = 'best.pth.tar'             # name of the checkpoint of the best model so far
final = 'final.pth.tar'                 # name of the checkpoint of the final model

def train_ckp(e: int)-> str:
    """
    Returns the name of the training checkpoint with name 'ckp-xxxx.pth.tar', with a four digit epoch number as x.

    :param e: int, epoch
    :return: str, name
    """

    return 'ckp-{:04}.pth.tar'.format(e)


composition_length = 1024               # Length of the composition to be generated
number_of_trials_before_giving_up = 3  # Number of tests before giving up composing


if __name__ == '__main__':

    print('Testing checkpoints')
    from model import MusicTransformer
    from utils import NoamOptimizer
    import torch.optim as optim

    model = MusicTransformer(D, L, N, H)
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(.9, .98), eps=1e-9)
    scheduler = NoamOptimizer(D, optimizer=optimizer)
    test_checkpoint = {
        'epoch': 0,
        'state_dict': model.state_dict(),
        'optimizer': scheduler.optimizer.state_dict(),
        'best_val_loss': 4
    }
    torch.save(test_checkpoint, model_dir+stop_and_go)
    test_checkpoint = torch.load(model_dir+stop_and_go)

    model.load_state_dict(test_checkpoint['state_dict'])
    scheduler.optimizer.load_state_dict(test_checkpoint['optimizer'])
    e=test_checkpoint['epoch']


    for k in test_checkpoint:
        print(k, test_checkpoint[k])
