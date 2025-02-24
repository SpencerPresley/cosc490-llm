from easydict import EasyDict
from basics import run_all_basics_demo
from model import load_data, run, visualize_epochs, visualize_configs
from typing import List, Tuple, Dict, Union

EMBEDDING_TYPES = [
    "glove-twitter-50",
    "glove-twitter-100",
    "glove-twitter-200",
    "word2vec-google-news-300",
]


def single_run(
    dev_d: Dict[str, List[Union[str, int]]],
    train_d: Dict[str, List[Union[str, int]]],
    test_d: Dict[str, List[Union[str, int]]],
):
    train_config = EasyDict(
        {
            "batch_size": 64,
            "lr": 0.025,
            "num_epochs": 20,
            "save_path": "model.pth",
            "embeddings": EMBEDDING_TYPES[0],
            "num_classes": 2,
        }
    )

    epoch_train_losses, _, epoch_dev_loss, epoch_dev_accs, _, _ = run(
        train_config, dev_d, train_d, test_d
    )
    visualize_epochs(epoch_train_losses, epoch_dev_loss, "single_run_loss.png")


def explore_embeddings(
    dev_d: Dict[str, List[Union[str, int]]],
    train_d: Dict[str, List[Union[str, int]]],
    test_d: Dict[str, List[Union[str, int]]],
):
    all_emb_epoch_dev_accs, all_emb_epoch_dev_losses = [], []

    for embedding_type in EMBEDDING_TYPES:
        train_config = EasyDict(
            {
                "batch_size": 64,
                "lr": 0.025,
                "num_epochs": 20,
                "save_path": "model.pth",
                "embeddings": embedding_type,
                "num_classes": 2,
            }
        )

        _, _, epoch_dev_loss, epoch_dev_accs, _, _ = run(
            train_config, dev_d, train_d, test_d
        )
        all_emb_epoch_dev_accs.append(epoch_dev_accs)
        all_emb_epoch_dev_losses.append(epoch_dev_loss)
    visualize_configs(
        all_emb_epoch_dev_accs, EMBEDDING_TYPES, "Accuracy", "./embedding_acc.png"
    )
    visualize_configs(
        all_emb_epoch_dev_losses, EMBEDDING_TYPES, "Loss", "./embedding_loss.png"
    )


if __name__ == "__main__":
    # Run every demo function in basics.py
    # uncomment the following line to run
    # run_all_basics_demo()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_single_run", action="store_true")
    args = parser.parse_args()

    # load raw data
    dev_data, train_data, test_data = load_data()

    # Run a single training run
    if args.run_single_run:
        single_run(dev_data, train_data, test_data)

    # Explore different embeddings
    else:
        explore_embeddings(dev_data, train_data, test_data)
