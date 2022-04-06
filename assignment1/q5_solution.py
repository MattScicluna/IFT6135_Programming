# **Question 5 (Interpreting the Model)**
"""
(30 points) In real-world applications of deep learning, it is *crucial* that we verify that our models are learning what we expect them to learn. In this exercise, we will replicate a part of figure 3b from [Basset](https://pubmed.ncbi.nlm.nih.gov/27197224/).

In genetics, there exists well known DNA *motifs*: short sequences which appear throughtout our DNA, and whose function are well documented. We expect that the filters of the first convolution layer should learn to identify some of these motifs in order to solve this task.

**Please submit the answers to this exercise on a single paged PDF!**
"""

import csv
import math

import scipy
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair

import solution 

batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set RNG
seed = 42
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if device.type=='cuda':
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

model = torch.load('model_params.pt').to(device)
basset_dataset_test = solution.BassetDataset(path='/content/A1', f5name='er.h5', split='test')
basset_dataloader_test = DataLoader(basset_dataset_test,
                                    batch_size=batch_size,
                                    drop_last=True,
                                    shuffle=False,
                                    num_workers=1)



"""
1. First, we need to ensure that our model has learned something. Plot the ROC curve and compute the AUC of your model after training. Compare the ROC curves and the AUC before and after training with your simulated models. What do you notice?
"""

predictions_tensor = torch.tensor([])
trgs_tensor = torch.tensor([])

model.eval()

for i, out in enumerate(basset_dataloader_test):
    seqs = out['sequence'].to(device)
    trgs = out['target'].to(device)

    predictions = model(seqs)
    # Use sklearn implementation to ensure numerical accuracy
    #fpr, tpr, _ = metrics.roc_curve(trgs.flatten().cpu().detach().numpy(),
    #                                predictions.flatten().cpu().detach().numpy())
    #output['total_score'] += metrics.auc(fpr, tpr)
    predictions = 1/ (1 + np.exp(-predictions.flatten().cpu().detach().numpy()))
    predictions_tensor = torch.cat([predictions_tensor, predictions.detach().cpu()])
    trgs_tensor = torch.cat([trgs_tensor, trgs.detach().cpu()])

score, fpr, tpr = solution.compute_auc(trgs_tensor,
                                       predictions_tensor)

# taken from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
plt.figure()
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % score,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.show()

"""
2. We represent motifs as position weight matrices (PWMs). This is a matrix of size $4$ $\times$ the motif length, where the $(i,j)$th entry is a count of how often base-pair $i$ occurs at position $j$. Open the PWM for the CTCF motif, which can be found in `MA0139.1.jaspar`. Normalize this matrix so that each column sums to $1$.
"""

to_match = []

with open("MA0139.1.jaspar", "r") as tsv:
    tsv.readline()
    for line in csv.reader(tsv):
        to_match.append(line[0][1:].strip()[1:-1].strip().split())

to_match = np.array(to_match, dtype=int)
to_match = to_match / to_match.sum(0)  # ACGT ordering

"""
3. In the methods section of the [paper](https://pubmed.ncbi.nlm.nih.gov/27197224/) (page 998), the authors describe how they converted each of the $300$ filters into normalized PWMs. First, for each filter, they determined the maximum activated value across the *dataset* (you may use a subset of the test set here). Compute these values.
"""

max_activation_vals = torch.zeros(300).to(device)
for i, out in enumerate(basset_dataloader_test):
    max_activation_vals = torch.max(
        max_activation_vals,
        model.conv1(out['sequence'].to(device)).sum(3).max(0).values.max(1).values
    )
    if i == 100: # just do subset!
        break

"""
4. Next, they counted the base-pair occurrences in the set of sequences that activate the filter to a value that is more than half of its maximum value.

  Note: You should use `torch.functional.unfold`.
"""

activation_vals = max_activation_vals / 2 # want half of this value


PWMs = torch.zeros(19, 4, 300).to(device)

#  Now build PWMs (See https://en.wikipedia.org/wiki/Position_weight_matrix for definition)
#  technically this is a position prob matrix but whatever
for i, out in enumerate(basset_dataloader_test):

    # Unroll sequence
    input_ = out['sequence'].to(device)
    kernel_size = _pair((19, 4))
    out_channels = 300
    dilation = _pair(1)
    padding = _pair((9,0))
    stride = _pair((1,1))
    n_channels = 1

    #hout = ((input_.shape[2] + 2 * padding[0] - dilation[0] * (kernel_size[0]-1)-1)//stride[0])+1
    #wout = ((input_.shape[3] + 2 * padding[1] - dilation[1] * (kernel_size[1]-1)-1)//stride[1])+1

    inputUnfolded = F.unfold(input_, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride)
    inputUnfolded = inputUnfolded.reshape(-1, 19, 4, 600)

    is_bigger = (model.conv1(input_).squeeze() > activation_vals.unsqueeze(0).unsqueeze(2))

    is_bigger = is_bigger.unsqueeze(1).unsqueeze(1).permute(0,1,2,4,3)
    inputUnfolded = inputUnfolded.unsqueeze(4)

    PWMs += (inputUnfolded * is_bigger).sum(0).sum(2).detach()

    if i == 100:  # just do subset!
        break

PWMs_final = (PWMs / PWMs.sum(1).unsqueeze(1)).permute(1,0,2).cpu().numpy()

"""
5. Given your 300 PWMs derived from your convolution filters, check to see if any of them are similar to the PWM for CTCF. You could quantify the similarity using *Pearson Correlation Coefficient*. Make a visualization of the PWM of the CTCF motif along with the most similar ones learned from the network. 
"""

#PWMs_final = PWMs_final.detach().cpu().numpy()
max_index = 0
max_val = 0

for i in range(300):
    try:
        val, p = scipy.stats.pearsonr(to_match.flatten(), PWMs_final[:,:,i].flatten())
        if val > max_val:
            max_val = val
            max_index = i
            print(val, p, max_index)
    except:
        pass

plt.imshow(PWMs_final[:,:,max_index], cmap='Reds', vmin=0, vmax=1)

plt.imshow(to_match, cmap='Reds', vmin=0, vmax=1)
