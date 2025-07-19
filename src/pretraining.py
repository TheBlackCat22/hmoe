from datasets import load_dataset
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser
)


accelerator = Accelerator()
def print_on_main(text):
    if accelerator.is_main_process:
        print(text)


def fineweben_pretraining():
    dataset = load_dataset('HuggingFaceFW/fineweb', data_dir="sample/10BT", split="train", streaming=True)
    columns = dataset.column_names
    columns.remove('text')
    dataset = dataset.map(lambda example: {'text' : ' '.join(example['text'].split()[:512])}, remove_columns=columns)
    return dataset

    
def main(training_args, model_args):

    print_on_main('\n\n***************************')
    print_on_main('Loading Model and Tokenizer')
    print_on_main('***************************')
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=model_args.torch_dtype,
    )
    print_on_main(model)

    print_on_main('\n\n***************')
    print_on_main('Loading Dataset')
    print_on_main('***************')
    dataset = fineweben_pretraining()
    print_on_main(dataset)

    print_on_main('\n\n********')
    print_on_main('Training')
    print_on_main('********')
    training_args.logging_dir = training_args.output_dir
    training_args.accelerator_config.split_batches = True
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    parser = TrlParser((SFTConfig, ModelConfig))
    training_args, model_args = parser.parse_args_and_config()
    main(training_args, model_args)