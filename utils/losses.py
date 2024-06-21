from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity


def get_loss(loss_name: str) -> losses.BaseMetricLossFunction:
    """
    A function that returns a specific loss function based on the input loss_name parameter.

    Parameters:
    loss_name (str): The name of the loss function to be returned.

    Returns:
    Specific loss function based on the input loss_name.
    """
    if loss_name == "SupConLoss":
        return losses.SupConLoss(temperature=0.07)
    if loss_name == "CircleLoss":
        # these are params for image retrieval
        return losses.CircleLoss(m=0.4, gamma=80)
    if loss_name == "MultiSimilarityLoss":
        return losses.MultiSimilarityLoss(
            alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity()
        )
    if loss_name == "ContrastiveLoss":
        return losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    if loss_name == "Lifted":
        return losses.GeneralizedLiftedStructureLoss(
            neg_margin=0, pos_margin=1, distance=DotProductSimilarity()
        )
    if loss_name == "FastAPLoss":
        return losses.FastAPLoss(num_bins=30)
    if loss_name == "NTXentLoss":
        # The MoCo paper uses 0.07, while SimCLR uses 0.5.
        return losses.NTXentLoss(temperature=0.07)
    if loss_name == "TripletMarginLoss":
        # or an int, for example 100
        return losses.TripletMarginLoss(
            margin=0.1, swap=False, smooth_loss=False, triplets_per_anchor="all"
        )
    if loss_name == "CentroidTripletLoss":
        return losses.CentroidTripletLoss(
            margin=0.05,
            swap=False,
            smooth_loss=False,
            triplets_per_anchor="all",
        )
    if loss_name == "ArcFaceLoss":
        return losses.ArcFaceLoss(
            num_classes=10, embedding_size=512, margin=0.5, scale=64
        )
    if loss_name == "AngularLoss":
        return losses.AngularLoss(alpha=40)
    raise NotImplementedError(f"Sorry, <{loss_name}> loss function is not implemented!")


def get_miner(miner_name: str, margin: float = 0.1) -> miners.BaseMiner:
    """
    Generate a miner based on the provided miner_name and margin.

    Parameters:
    miner_name (str): The name of the miner to generate.
    margin (float): The margin value for the miner (default is 0.1).

    Returns:
    miners.TripletMarginMiner or miners.MultiSimilarityMiner or miners.PairMarginMiner or None:
    The generated miner based on the miner_name. Returns None if the miner_name is not recognized.
    """
    if miner_name == "TripletMarginMiner":
        # all, hard, semihard, easy
        return miners.TripletMarginMiner(margin=margin, type_of_triplets="semihard")
    if miner_name == "MultiSimilarityMiner":
        return miners.MultiSimilarityMiner(epsilon=margin, distance=CosineSimilarity())
    if miner_name == "PairMarginMiner":
        return miners.PairMarginMiner(
            pos_margin=0.7, neg_margin=0.3, distance=DotProductSimilarity()
        )
    if miner_name == "UniformHistogramMiner":
        return miners.UniformHistogramMiner(
            num_bins=100, pos_per_bin=10, neg_per_bin=10
        )
    if miner_name == "BatchHardMiner":
        return miners.BatchHardMiner()
    if miner_name == "DistanceWeightedMiner":
        return miners.DistanceWeightedMiner(cutoff=0.5, nonzero_loss_cutoff=1.4)
    if miner_name == "BatchEasyHardMiner":
        return miners.BatchEasyHardMiner()
    if miner_name == "AngularMiner":
        return miners.AngularMiner()
    
    return None


if __name__ == "__main__":
    # Example usage
    loss = get_loss("ArcFaceLoss")
    print("PARAMTERS:", loss.parameters())
    miner = get_miner("TripletMarginMiner")
