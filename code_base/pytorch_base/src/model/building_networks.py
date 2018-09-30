# import networks
import src.model.proposal as proposal
import src.model.densecap as densecap
import src.model.e2e_densecap as e2e_densecap
import src.model.sequence_generator as sg
import src.model.move_forward_tell as mft
from src.model.captioning import CaptioningModel
from src.model.hierarchical_captioning import HierarchicalCaptioningModel

# networks mapping
SST = proposal.SST
BiSST = proposal.BiSST
DenseCap = densecap.DenseVideoCaptioning
E2E_DenseCap = e2e_densecap.E2EDenseVideoCaptioning
SequenceGenerator = sg.SequenceGenerator
MFT = mft.MoveForwardTell


def get_proposal_network(config, proposal_network_type):
    if proposal_network_type == "SST":
        M = SST
    elif proposal_network_type == "BiSST":
        M = BiSST
    else:
        raise NotImplementedError(
            "Not supported proposal network ({})".format(
                proposal_network_type))
    return M(config)

def get_captioning_network(config, captioning_network_type, prefix=""):
    if captioning_network_type == "rnn":
        if len(prefix) == 0:
            prefix = "wordRNN"
        return CaptioningModel(config, name=prefix)
    elif captioning_network_type == "hierarchical":
        return HierarchicalCaptioningModel(config, prefix=prefix)
    else:
        raise NotImplementedError(
            "Not supported captioning network ({})".format(
                captioning_network_type))

def get_sequence_generator(config, seq_gen_type):
    if seq_gen_type == "pointer_network":
        return sg.PointerNetwork(config, name="seq_gen")
    elif seq_gen_type == "move_forward_network":
        return mft.MoveForwardTell(config, name="seq_gen")
    else:
        raise NotImplementedError(
            "Not supported sequence generator network ({})".format(
                seq_gen_type))
