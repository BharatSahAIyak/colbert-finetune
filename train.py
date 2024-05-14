import pandas as pd
from ragatouille import RAGTrainer
import os
import torch
from colbert.infra import ColBERTConfig
from colbert.training.rerank_batcher import RerankBatcher
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from colbert.infra import ColBERTConfig
from colbert.training.rerank_batcher import RerankBatcher
from colbert.utils.amp import MixedPrecisionManager
from colbert.training.lazy_batcher import LazyBatcher
from colbert.modeling.colbert import ColBERT
from colbert.modeling.reranker.electra import ElectraReranker
from colbert.infra.config import ColBERTConfig
from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints
import torch.distributed as dist


def prepare_data(
    content,
    train_qna,
    output_data_path="./data/",
    model_name="MyFineTunedColBERT",
    checkpoint="colbert-ir/colbertv2.0",
):
    train_df = pd.merge(
        train_qna,
        content[["id", "content"]],
        how="left",
        left_on="content_row",
        right_on="id",
    )

    # filter train data based on length
    train_df["content_length"] = train_df["content"].str.split().apply(len)
    train_df = train_df.loc[train_df["content_length"] > 100, :]
    train_df = train_df[["question", "answer", "id"]]
    train_pairs = [
        (r["question"], r["answer"])
        for _, r in train_df[["question", "answer"]].iterrows()
    ]

    trainer = RAGTrainer(model_name=model_name, pretrained_model_name=checkpoint)
    trainer.prepare_training_data(
        raw_data=train_pairs,
        data_out_path=output_data_path,
        all_documents=data.content.to_list(),
        num_new_negatives=10,
        mine_hard_negatives=True,
    )


def get_reader(collection, config, triples, queries):
    if collection is not None:
        if config.reranker:
            reader = RerankBatcher(
                config,
                triples,
                queries,
                collection,
                (0 if config.rank == -1 else config.rank),
                config.nranks,
            )
        else:
            reader = LazyBatcher(
                config,
                triples,
                queries,
                collection,
                (0 if config.rank == -1 else config.rank),
                config.nranks,
            )
        return reader
    else:
        raise NotImplementedError()


def init_model(
    lr=1e-05,
    model_checkpoint="colbert-ir/colbertv2.0",
):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    config = ColBERTConfig(
        bsize=1,
        lr=lr,
        warmup=3000,
        doc_maxlen=180,
        dim=128,
        attend_to_mask_tokens=False,
        nway=2,
        accumsteps=1,
        similarity="cosine",
        use_ib_negatives=True,
        checkpoint=model_checkpoint,
    )

    assert config.bsize % config.nranks == 0, (config.bsize, config.nranks)
    config.bsize = config.bsize // config.nranks

    # model
    if not config.reranker:
        colbert = ColBERT(name=config.checkpoint, colbert_config=config)
    else:
        colbert = ElectraReranker.from_pretrained(config.checkpoint)

    colbert = colbert.to(DEVICE)
    colbert.train()

    ## For training on a multiple GPU

    # # Initialize the process group
    # # Set environment variables
    # os.environ["MASTER_ADDR"] = (
    #     "localhost"  # Or the IP address of the master node if multi-node
    # )
    # os.environ["MASTER_PORT"] = (
    #     "12345"  # Any free port that can be used for communication
    # )
    # dist.init_process_group(
    #     backend="nccl", init_method="env://", rank=config.rank, world_size=config.nranks
    # )
    # # Wrap the model with DistributedDataParallel
    # colbert = torch.nn.parallel.DistributedDataParallel(
    #     colbert,
    #     device_ids=[config.rank],
    #     output_device=config.rank,
    #     find_unused_parameters=True,
    # )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, colbert.parameters()), lr=config.lr, eps=1e-8
    )
    optimizer.zero_grad()

    scheduler = None
    if config.warmup is not None:
        print(
            f"#> LR will use {config.warmup} warmup steps and linear decay over {config.maxsteps} steps."
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup,
            num_training_steps=config.maxsteps,
        )

    # warmup_bert = config.warmup_bert
    # if warmup_bert is not None:
    #     set_bert_grad(colbert, False)
    return (config, colbert, optimizer, scheduler)


def train(
    num_epochs=40,
    triples="data/triples.train.colbert.jsonl",
    queries_path="data/queries.train.colbert.tsv",
    collection_path="data/corpus.train.colbert.tsv",
):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    config, colbert, optimizer, scheduler = init_model()

    amp = MixedPrecisionManager(config.amp)
    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)
    # start_time = time.time()
    train_loss = None
    train_loss_mu = 0.999
    start_batch_idx = 0

    loss_df = pd.DataFrame(columns=["Epoch", "Step", "Loss"])
    loss_history = []

    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch + 1}/{num_epochs}")

        # Reinitialize or reset the reader here if necessary
        reader = get_reader(collection_path, config, triples, queries_path)

        for batch_idx, BatchSteps in zip(range(start_batch_idx, 256), reader):
            # if (warmup_bert is not None) and warmup_bert <= batch_idx:
            #     set_bert_grad(colbert, True)
            #     warmup_bert = None
            this_batch_loss = 0.0

            for batch in BatchSteps:
                with amp.context():
                    try:
                        queries, passages, target_scores = batch
                        encoding = [queries, passages]
                    except:
                        encoding, target_scores = batch
                        encoding = [encoding.to(DEVICE)]

                    scores = colbert(*encoding)

                    if config.use_ib_negatives:
                        scores, ib_loss = scores

                    scores = scores.view(-1, config.nway)

                    if len(target_scores) and not config.ignore_scores:
                        target_scores = (
                            torch.tensor(target_scores).view(-1, config.nway).to(DEVICE)
                        )
                        target_scores = target_scores * config.distillation_alpha
                        target_scores = torch.nn.functional.log_softmax(
                            target_scores, dim=-1
                        )

                        log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
                        loss = torch.nn.KLDivLoss(
                            reduction="batchmean", log_target=True
                        )(log_scores, target_scores)
                    else:
                        loss = nn.CrossEntropyLoss()(scores, labels[: scores.size(0)])

                    if config.use_ib_negatives:
                        if config.rank < 1:
                            print(
                                "EPOCH ",
                                epoch,
                                " \t\t\t\t",
                                loss.item(),
                                ib_loss.item(),
                            )

                        loss += ib_loss

                    loss = loss / config.accumsteps

                if config.rank < 1:
                    print_progress(scores)

                amp.backward(loss)
                this_batch_loss += loss.item()

                if batch_idx % 500 == 0:
                    formatted_loss = "{:.6e}".format(
                        this_batch_loss
                    )  # Adjust the precision (e.g., 6) as needed
                    loss_history.append((epoch + 1, batch_idx + 1, formatted_loss))
                    loss_df = pd.DataFrame(
                        loss_history, columns=["Epoch", "Step", "Loss"]
                    )
                    loss_df.to_csv("loss_history.csv", index=False)

            train_loss = this_batch_loss if train_loss is None else train_loss
            train_loss = (
                train_loss_mu * train_loss + (1 - train_loss_mu) * this_batch_loss
            )

            amp.step(colbert, optimizer, scheduler)

        if config.rank < 1:
            print_message(batch_idx, train_loss)
            epoch_save_path = f"./model_checkpoint/epoch_{epoch}/"
            os.makedirs(epoch_save_path, exist_ok=True)
            checkpoint_filename = f"checkpoint_batch_{batch_idx+1}.pt"
            full_checkpoint_path = os.path.join(epoch_save_path, checkpoint_filename)
            manage_checkpoints(
                config,
                colbert,
                optimizer,
                batch_idx + 1,
                savepath=full_checkpoint_path,
                consumed_all_triples=True,
            )
            config.checkpoint = full_checkpoint_path + "/colbert/"

    if config.rank < 1:
        print_message("#> Done with all triples!")
        ckpt_path = manage_checkpoints(
            config,
            colbert,
            optimizer,
            batch_idx + 1,
            savepath=None,
            consumed_all_triples=True,
        )


if __name__ == "__main__":
    # importing data
    data = pd.read_csv("./data/content.csv")
    train_dataset = pd.read_csv("./data/q_n_a.csv")

    prepare_data(data, train_dataset)
    train()
