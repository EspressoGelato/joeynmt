import random
from collections import namedtuple
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.nn.utils.rnn import pad_sequence

from joeynmt.training import *
from joeynmt.model import build_model
from joeynmt.builders import build_optimizer, build_scheduler, \
    build_gradient_clipper
from joeynmt.ql_search import transformer_greedy, push_sample_to_memory

from joeynmt.batch import Batch
from joeynmt.data import load_data

from joeynmt.vocabulary import Vocabulary
from joeynmt.metrics import bleu
from joeynmt.prediction import validate_on_data

from joeynmt.helpers import log_data_info, load_config, log_cfg, \
    store_attention_plots, load_checkpoint, make_model_dir, \
    make_logger, set_seed, symlink_update, ConfigurationError

from joeynmt.loss import XentLoss


Transition = namedtuple('Transition',
                        ('source_sentence', 'prefix','next_word', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        # save a transition
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1 ) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class TrainStatistics:
    def __init__(self, steps: int = 0, stop: bool = False,
                 total_tokens: int = 0, best_ckpt_iter: int = 0,
                 best_ckpt_score: float = np.inf,
                 minimize_metric: bool = True) -> None:
        # global update step counter
        self.steps = steps
        # stop training if this flag is True
        # by reaching learning rate minimum
        self.stop = stop
        # number of total tokens seen so far
        self.total_tokens = total_tokens
        # store iteration point of best ckpt
        self.best_ckpt_iter = best_ckpt_iter
        # initial values for best scores
        self.best_ckpt_score = best_ckpt_score
        # minimize or maximize score
        self.minimize_metric = minimize_metric

    def is_best(self, score):
        if self.minimize_metric:
            is_best = score < self.best_ckpt_score
        else:
            is_best = score > self.best_ckpt_score
        return is_best



def Q_learning(cfg_file: str) -> None:
    """
    Main training function. After training, also test on test data if given.
    :param cfg_file: path to configuration yaml file
    """
    cfg = load_config(cfg_file) # config is a dict
    # make logger
    model_dir = make_model_dir(cfg["training"]["model_dir"],
                   overwrite=cfg["training"].get("overwrite", False))
    _ = make_logger(model_dir, mode="train")    # version string returned
    # TODO: save version number in model checkpoints

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    # load the data
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(
        data_cfg=cfg["data"])
    # The training data is filtered to include sentences up to `max_sent_length`
    #     on source and target side.

    train_config = cfg["training"]

    # data & batch handling # (modified from TrainManager)
    shuffle = train_config.get("shuffle", True)
    batch_size = train_config["batch_size"]
    batch_type = train_config.get("batch_type", "sentence")
    use_cuda = train_config["use_cuda"] and torch.cuda.is_available()

    # validation part config
    # validation
    validation_freq = train_config.get("validation_freq", 1000)
    ckpt_queue = queue.Queue(
        maxsize=train_config.get("keep_last_ckpts", 5))
    eval_batch_size = train_config.get("eval_batch_size", batch_size)
    level = cfg["data"]["level"]
    eval_metric = train_config.get("eval_metric", "bleu")
    n_gpu = torch.cuda.device_count() if use_cuda else 0
    eval_batch_type = train_config.get("eval_batch_type", batch_type)
    # eval options
    test_config = cfg["testing"]
    bpe_type = test_config.get("bpe_type", "subword-nmt")
    sacrebleu = {"remove_whitespace": True, "tokenize": "13a"}
    max_output_length = train_config.get("max_output_length", None)
    minimize_metric = True
    # initialize training statistics
    stats = TrainStatistics(
        steps=0,
        stop=False,
        total_tokens=0,
        best_ckpt_iter=0,
        best_ckpt_score=np.inf if minimize_metric else -np.inf,
        minimize_metric=minimize_metric
    )

    early_stopping_metric = train_config.get("early_stopping_metric",
                                             "eval_metric")
    if early_stopping_metric in ["ppl", "loss"]:
        minimize_metric = True
    elif early_stopping_metric == "eval_metric":
        if eval_metric in ["bleu", "chrf",
                                "token_accuracy", "sequence_accuracy"]:
            minimize_metric = False
        # eval metric that has to get minimized (not yet implemented)
        else:
            minimize_metric = True

    # data loader(modified from train_and_validate function
    # Returns a torchtext iterator for a torchtext dataset.
    # param dataset: torchtext dataset containing src and optionally trg
    train_iter = make_data_iter(train_data,
                                batch_size= batch_size,
                                batch_type= batch_type,
                                train=True, shuffle = shuffle)

    # initialize the Replay Memory D with capacity N
    memory = ReplayMemory(10000)

    # initialize two DQN networks
    policy_net = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab) # Q_network
    target_net = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab) # Q_hat_network

    if use_cuda:
        policy_net.cuda()
        target_net.cuda()

    target_net.load_state_dict(policy_net.state_dict())
    # Initialize target net Q_hat with weights equal to policy_net

    target_net.eval() # target_net not update the parameters, test mode

    # Optimizer
    optimizer = build_optimizer(config=cfg["training"],
                                parameters=policy_net.parameters())
    # Loss function
    loss_function = torch.nn.MSELoss()

    pad_index = policy_net.pad_index
    # print('!!!'*10, pad_index)

    cross_entropy_loss = XentLoss(pad_index=pad_index)
    policy_net.loss_function = cross_entropy_loss



    num_episodes = 100
    TARGET_UPDATE = 10

    for i_episode in range(num_episodes):
        # Outer loop

        # get batch
        for i, batch in enumerate(iter(train_iter)): # joeynmt training.py 377

            # create a Batch object from torchtext batch
            # ( use class Batch from batch.py)
            # return the sentences same length (with padding) in one batch
            batch = Batch(batch, policy_net.pad_index,
                                     use_cuda= use_cuda)
            # we want to get batch.src and batch.trg
            # the shape of batch.src: (batch_size * length of the sentence)

            # source here is represented by the word index not word embedding.
            # Use Model._encode: self.src_embed(src) to turn word index into word embedding.

            encoder_output_batch, _, _, _ = policy_net(
                return_type="encode",
                src = batch.src,
                src_length = batch.src_length,
                src_mask = batch.src_mask,
            )
            #print('@@@@@@@@@@@', batch.src_length)
            #print(len(batch.src_length))
            #print(batch.src)
            print('batch.src.shape is: ', batch.src.shape)


            # get the translated output of a batch
            trans_output_batch, _ = transformer_greedy(
                src_mask = batch.src_mask, max_output_length = 100, model = policy_net,
                encoder_output = encoder_output_batch, eps_threshold = 0.1)

            print('trans_output_batch.shape is:', trans_output_batch.shape)
            # batch_size * max_translation_sentence_length

            # decode back to symbols
            # Convert multiple arrays containing sequences of token IDs to their
            # sentences, optionally cutting them off at the end-of-sequence token.
            # :param arrays: 2D array containing indices
            # :param cut_at_eos: cut the decoded sentences at the first <eos>
            # :param skip_pad: skip generated <pad> tokens
            # :return: list of list of strings (tokens)

            #print('***************************************batch.trg', batch.trg)
            print('batch.trg.shape is:', batch.trg.shape)
            #print(trans_output_batch.shape)
            #print('&&&&&&&&&&&&&7', trans_output_batch[0])
            #print(trans_output_batch[1])

            reward_batch = []# Get the reward_batch (Get the bleu score of the sentences in a batch)

            for i in range(int(batch.src.shape[0])):
                all_outputs = [trans_output_batch[i]]
                hypotheses = policy_net.trg_vocab.arrays_to_sentences(arrays = all_outputs,
                                                                  cut_at_eos=True)

                all_ref = [batch.trg[i]]
                references = policy_net.trg_vocab.arrays_to_sentences(arrays= all_ref,
                                                                  cut_at_eos=True)

                #print('!!!!hypo', hypotheses)
                #print('!!!ref', references)
                # evaluate with metric on full dataset
                join_char = " " if level in ["word", "bpe"] else ""
                # valid_sources = [join_char.join(s) for s in data.src]
                valid_references = [join_char.join(t) for t in references]
                valid_hypotheses = [join_char.join(t) for t in hypotheses]
                current_valid_score = bleu(
                    valid_hypotheses, valid_references,
                    tokenize=sacrebleu["tokenize"])

                reward_batch.append(current_valid_score)
            print('reward batch is', reward_batch)

            #reward_batch = bleu(hypotheses, references, tokenize="13a")
            #print(reward_batch)
            # shape?

            # make prefix and push tuples into memory
            push_sample_to_memory(eos_index = policy_net.eos_index, memory = memory, src_batch = batch.src,
                                  trans_output_batch = trans_output_batch,
                                  reward_batch= reward_batch, max_output_length=100)
            #if use_cuda:
             #   torch.cuda.empty_cache()


            T =10
            # inner loop
            mini_BATCH_SIZE = 32

            for t in range(T):
                # Sample mini-batch from the memory
                transitions = memory.sample(mini_BATCH_SIZE)
                # transition = [Transition(source=array([]), prefix=array([]), next_word= int, reward= int),
                #               Transition(source=array([]), prefix=array([]), next_word= int, reward= int,...]
                # Each Transition is what we push into memory for one sentence: memory.push(source, prefix, next_word, reward_batch[i])
                mini_batch = Transition(*zip(*transitions))
                # merge the same class in transition together
                # mini_batch = Transition(source=(array([]), array([]),...), prefix=(array([],...),
                #               next_word=array([...]), reward=array([...]))
                # mini_batch.reward is tuple: length is mini_BATCH_SIZE.


                # concatenate together into a tensor.
                words = []
                for word in mini_batch.next_word:
                    new_word = word.unsqueeze(0)
                    words.append(new_word)
                mini_next_word = torch.cat(words)  # shape (mini_BATCH_SIZE,)
                mini_reward = torch.tensor(mini_batch.reward)  # shape (mini_BATCH_SIZE,)

                #print(mini_next_word.shape)
                #print(mini_reward.shape)



                mini_src_length = [len(item) for item in mini_batch.source_sentence]
                mini_src_length = torch.Tensor(mini_src_length)

                mini_src = pad_sequence(mini_batch.source_sentence, batch_first=True, padding_value = float(pad_index))
                # shape (mini_BATCH_SIZE, max_length_src)

                length_prefix = [len(item) for item in mini_batch.prefix]
                mini_prefix_length = torch.Tensor(length_prefix)

                #print('###########mininbatch_prefix',mini_batch.prefix)

                prefix_list = []
                for prefix_ in mini_batch.prefix:
                    prefix_ =  torch.from_numpy(prefix_)
                    prefix_list.append(prefix_)

                mini_prefix = pad_sequence(prefix_list, batch_first=True, padding_value = pad_index)
                # shape (mini_BATCH_SIZE, max_length_prefix)

                mini_src_mask = (mini_src != pad_index).unsqueeze(1)
                mini_trg_mask = (mini_prefix != pad_index).unsqueeze(1)

                # max_length_src = torch.max(mini_src_length)#max([len(item) for item in mini_batch.source_sentence])

                if use_cuda:
                    mini_src = mini_src.cuda()
                    mini_prefix = mini_prefix.cuda()
                    mini_src_mask = mini_src_mask.cuda()
                    mini_src_length = mini_src_length.cuda()
                    mini_trg_mask = mini_trg_mask.cuda()
                    mini_next_word = mini_next_word.cuda()


                #print(next(policy_net.parameters()).is_cuda)
                #print(mini_trg_mask.get_device())
                # calculate the Q_value
                logits_Q, _, _, _ = policy_net._encode_decode(
                    src = mini_src,
                    trg_input= mini_prefix,
                    src_mask= mini_src_mask,
                    src_length = mini_src_length,
                    trg_mask = mini_trg_mask  # trg_mask = (self.trg_input != pad_index).unsqueeze(1)
                )

                #print('%%%%%%%%%%%%%%%%%%%%%%%' * 10, logits_Q.shape) # torch.Size([64, 99, 31716])
                #print(mini_BATCH_SIZE) #64
                #print(logits_Q)
                #print('??????', mini_prefix_length.shape)
                #print(mini_prefix_length)

                # length_prefix = max([len(item) for item in mini_batch.prefix])
                # logits_Q shape: batch_size * length of the sentence * total number of words in corpus.
                logits_Q = logits_Q[range(mini_BATCH_SIZE), mini_prefix_length.long()-1,:]

                # logits shape: mini_batch_size * total number of words in corpus
                Q_value = logits_Q[range(mini_BATCH_SIZE), mini_next_word]

                mini_prefix_add = torch.cat([mini_prefix, mini_next_word.unsqueeze(1)],dim=1)
                mini_trg_mask_add = (mini_prefix_add != pad_index).unsqueeze(1)

                if use_cuda:
                    mini_prefix_add = mini_prefix_add.cuda()
                    mini_trg_mask_add = mini_trg_mask_add.cuda()

                logits_Q_hat,_,_,_ = target_net._encode_decode(
                    src = mini_src,
                    trg_input= mini_prefix_add,
                    src_mask= mini_src_mask,
                    src_length = mini_src_length,
                    trg_mask = mini_trg_mask_add
                )
                logits_Q_hat = logits_Q_hat[range(mini_BATCH_SIZE), mini_prefix_length.long(),:]
                Q_hat_value, indices = torch.max(logits_Q_hat, dim = 1)

                #print('%%%%%%%%%%%%%%%', Q_hat_value.shape)
                #print(Q_hat_value)


                Gamma = 0.1
                yj = mini_reward.float()
                #print('@@@@@@@@@@@@@@@@@@@@@@@@ mini reward', mini_reward)
                index = (mini_reward == 0)

                #print('index', index)
                #print(yj.shape, index.shape, Q_hat_value.shape)
                if use_cuda:
                    yj = yj.cuda()
                    Q_hat_value = Q_hat_value.cuda()

                yj[index] = Gamma * Q_hat_value[index]
            


                yj.detach()
                policy_net.zero_grad()

                # Compute loss
                loss = loss_function(yj, Q_value)
                print('loss', loss)

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()

                for param in policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()
                # increment step counter
                stats.steps += 1
                print('step', stats.steps)

                if t % TARGET_UPDATE == 0:
                    print('update the parameters in target_net.')
                    target_net.load_state_dict(policy_net.state_dict())
                    


                if stats.steps % validation_freq == 0: #Validation
                    print('Start validation')

                    valid_score, valid_loss, valid_ppl, valid_sources, \
                    valid_sources_raw, valid_references, valid_hypotheses, \
                    valid_hypotheses_raw, valid_attention_scores = \
                        validate_on_data(
                            model = policy_net,
                            data = dev_data,
                            batch_size = eval_batch_size,
                            use_cuda = use_cuda,
                            level = level,
                            eval_metric = eval_metric,
                            n_gpu = n_gpu,
                            compute_loss = True,
                            beam_size = 1,
                            beam_alpha = -1,
                            batch_type = eval_batch_type,
                            postprocess = True,
                            bpe_type = bpe_type,
                            sacrebleu = sacrebleu,
                            max_output_length = max_output_length
                        )
                    print('validation_loss: {}, validation_score: {}'.format(valid_loss, valid_score))
                    new_best = False
                    if early_stopping_metric == "loss":
                        ckpt_score = valid_loss
                    elif early_stopping_metric in ["ppl", "perplexity"]:
                        ckpt_score = valid_ppl
                    else:
                        ckpt_score = valid_score
                    if stats.is_best(ckpt_score):
                        stats.best_ckpt_score = ckpt_score
                        stats.best_ckpt_iter = stats.steps
                        logger.info('Hooray! New best validation result [%s]!',
                                    early_stopping_metric)
                        if ckpt_queue.maxsize > 0:
                            logger.info("Saving new checkpoint.")
                            new_best = True

                            # def _save_checkpoint(self) -> None:
                            """
                            Save the model's current parameters and the training state to a
                            checkpoint.
                            The training state contains the total number of training steps,
                            the total number of training tokens,
                            the best checkpoint score and iteration so far,
                            and optimizer and scheduler states.
                            """
                            model_path = "{}/{}.ckpt".format(model_dir, stats.steps)
                            model_state_dict = policy_net.module.state_dict() \
                                if isinstance(policy_net, torch.nn.DataParallel) \
                                else policy_net.state_dict()
                            state = {
                                "steps": stats.steps,
                                "total_tokens": stats.total_tokens,
                                "best_ckpt_score": stats.best_ckpt_score,
                                "best_ckpt_iteration": stats.best_ckpt_iter,
                                "model_state": model_state_dict,
                                "optimizer_state": optimizer.state_dict(),
                                #"scheduler_state": scheduler.state_dict() if
                                #self.scheduler is not None else None,
                                #'amp_state': amp.state_dict() if self.fp16 else None
                            }
                            torch.save(state, model_path)
                            if ckpt_queue.full():
                                to_delete = ckpt_queue.get()  # delete oldest ckpt
                                try:
                                    os.remove(to_delete)
                                except FileNotFoundError:
                                    logger.warning("Wanted to delete old checkpoint %s but "
                                                   "file does not exist.", to_delete)

                            ckpt_queue.put(model_path)

                            best_path = "{}/best.ckpt".format(model_dir)
                            try:
                                # create/modify symbolic link for best checkpoint
                                symlink_update("{}.ckpt".format(stats.steps), best_path)
                            except OSError:
                                # overwrite best.ckpt
                                torch.save(state, best_path)



Q_learning(cfg_file = '../configs/transformer_iwslt14_deen_bpe.yaml')

