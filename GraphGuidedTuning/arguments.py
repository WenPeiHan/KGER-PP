from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="D:\Code\PrML\Bert-base-Chinese",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    T5: bool = field(
        default=False
    )
    graph_guide: str = field(
        default=False,
        metadata={
            "help": "Whether to use a graph structure to guide the generation"
        }
    )
    embedding_enhance: str = field(
        default=False,
        metadata={
            "help": "Whether to use knowledge graph embedding information to enrich the model's input"
        }
    )

    lora: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Use LoRA to train",
        }
    )
    task_type: Optional[str] = field(
        default="CAUSAL_LM",
        metadata={
            "help": "Defines the task type targeted by the currently used LoRA fine-tuned model."
        }
    )
    r: Optional[int] = field(
        default=8,
        metadata={
            "help": "r represents the rank size after low-rank matrix decomposition in LoRA (Low-Rank Adaptation). "
                    "It determines the scale of the number of parameters and the expressive power of the low-rank adapter added when fine-tuning the pre-trained model. "
                    "A smaller r value means fewer learnable parameters and lower computational cost during model fine-tuning, "
                    "but it may have relatively limited fitting ability for complex tasks; a larger r value will increase the number of learnable parameters and have stronger "
                    "fitting ability, but it may also lead to overfitting and higher computational overhead.",
        }
    )
    lora_alpha: Optional[int] = field(
        default=16,
        metadata={
            "help": "在计算更新时，低秩矩阵的更新量会乘以 lora_alpha 除以 r 的值，以此来控制更新的强度，影响模型微调时对原有预训练权重的改变程度。",
        }
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "表示在 LoRA 低秩适配器结构中应用的 Dropout 概率。Dropout 是一种常见的正则化技术，在训练过程中以一定概率随机将神经元的输出置为 0，防止模型过拟合。"
                    "在这里，lora_dropout 就是控制在低秩适配器里进行这种随机置零操作的概率大小，合适的 Dropout 概率有助于提高模型的泛化能力，使得模型在未见过的数据上也能有较好的表现。",
        }
    )
    bias: Optional[str] = field(
        default="none",
        metadata={
            "help": "参数用于指定对模型中的偏置项（bias）如何处理。当设置为 none 时，表示在应用 LoRA 进行微调时，不对模型的偏置项进行调整，只针对选定的 target_modules 里的主要权重参数进行低秩适配更新"
        }
    )
    target_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": "It is a list of strings used to specify which modules in the pre-trained model will apply the LoRA low-rank adapter for fine-tuning.",
        }
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )
    quantization_bit: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of bits for model quantization."
        }
    )
    # pre_seq_len: Optional[int] = field(
    #     # default=None
    #     default=128
    # )
    # prefix_projection: bool = field(
    #     default=False
    # )

    def __post_init__(self):
        if self.model_name_or_path is None :
            raise ValueError("The path or name of the pre-trained model is required.")
        if not self.T5:
            raise ValueError("It is necessary to clarify whether the model is T5 or GLM.")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_file_path: str = field(
        # default=None,
        default="../DataSet/data/data_20250313.json",
        metadata={"help": "The all data set"}
    )
    graph_feature_path: str = field(
        # default=None,
        default="../GraphFeature/output/graph_feature_768.npy",
        metadata={"help": "The KoPL Graph feature path"}
    )
    entities_path: str = field(
        default="../MPKG_Embedding/data/entities.dict"
    )
    entity_embedding_path: str = field(
        default="../MPKG_Embedding/TransE/entity_embedding.npy",
        metadata={
            "help": "The entities embedding file path"
        }
    )
    train_test_split: float = field(
        default=0.1,
        metadata={"help": "The ratio of the division between the training set and the test set. The parameter represents the proportion of the test set in all the data."}
    )

    specify_type: float = field(
        default=None
    )

    type_balance: bool = field(
        default=True,
    )

    balance_max: int = field(
        default=30,
    )

    balance_min: int = field(
        default=10,
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )

    def __post_init__(self):
        if self.data_file_path is None :
            raise ValueError("Need either a dataset file.")
        elif self.graph_feature_path is None:
            raise ValueError("The graph features that need to be obtained in advance.")

