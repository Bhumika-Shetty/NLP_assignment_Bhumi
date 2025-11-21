import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import wandb

from t5_utils import (initialize_model, initialize_optimizer_and_scheduler, 
                      save_model, load_model_from_checkpoint, setup_wandb)
from transformers import T5TokenizerFast
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

# Global constants
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def parse_arguments():
    '''
    Parse command-line arguments for training configuration.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop for text-to-SQL')

    # Model configuration
    parser.add_argument('--train_from_scratch', action='store_true',
                        help="Train from scratch instead of fine-tuning (default: fine-tune)")
    
    # Optimizer and training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", 
                        choices=["AdamW", "SGD"],
                        help="Optimizer to use. Default: AdamW")
    parser.add_argument('--learning_rate', type=float, default=7e-4,
                        help="Learning rate. Default: 7e-4")
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help="Weight decay for regularization. Default: 0.01")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="Momentum for SGD optimizer. Default: 0.9")
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help="Gradient clipping max norm. Default: 1.0")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of gradient accumulation steps. Default: 1")

    # Learning rate scheduler configuration
    parser.add_argument('--scheduler_type', type=str, default="cosine", 
                        choices=["none", "cosine", "linear"],
                        help="Learning rate scheduler type. Default: cosine")
    parser.add_argument('--num_warmup_epochs', type=int, default=1,
                        help="Number of warmup epochs for LR scheduler. Default: 1")
    parser.add_argument('--max_n_epochs', type=int, default=25,
                        help="Maximum number of training epochs. Default: 25")
    parser.add_argument('--patience_epochs', type=int, default=3,
                        help="Early stopping: wait this many epochs without improvement. Default: 3")

    # Experiment tracking
    parser.add_argument('--use_wandb', action='store_true',
                        help="Use Weights & Biases for experiment tracking")
    parser.add_argument('--experiment_name', type=str, default='my_first_run',
                        help="Name for this experiment. Default: my_first_run")

    # Data loading configuration
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Training batch size. Default: 16")
    parser.add_argument('--test_batch_size', type=int, default=16,
                        help="Evaluation batch size. Default: 16")
    
    # Generation configuration
    parser.add_argument('--max_gen_length', type=int, default=512,
                        help="Maximum length for generated SQL queries. Default: 512")
    parser.add_argument('--num_beams', type=int, default=20,
                        help="Number of beams for beam search (1 = greedy). Default: 20")

    args = parser.parse_args()
    args.finetune = not args.train_from_scratch
    return args

def setup_training_paths(args):
    '''
    Setup paths for checkpoints, results, and records.
    '''
    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    
    # Create output directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('records', exist_ok=True)
    
    # Define file paths
    paths = {
        'gt_sql': 'data/dev.sql',
        'gt_records': 'records/ground_truth_dev.pkl',
        'model_sql': f'results/t5_{model_type}_{args.experiment_name}_dev.sql',
        'model_records': f'records/t5_{model_type}_{args.experiment_name}_dev.pkl'
    }
    
    return paths

def perform_training_step(model, batch, criterion):
    '''
    Perform a single training step: forward pass, loss computation, backward pass.
    '''
    encoder_input, encoder_mask, decoder_input, decoder_targets, _ = batch
    
    # Move to device
    encoder_input = encoder_input.to(DEVICE)
    encoder_mask = encoder_mask.to(DEVICE)
    decoder_input = decoder_input.to(DEVICE)
    decoder_targets = decoder_targets.to(DEVICE)

    # Forward pass
    outputs = model(
        input_ids=encoder_input,
        attention_mask=encoder_mask,
        decoder_input_ids=decoder_input,
    )
    logits = outputs.logits

    # Compute loss
    loss = criterion(
        logits.reshape(-1, logits.size(-1)),
        decoder_targets.reshape(-1)
    )
    
    return loss, decoder_targets

def train_epoch(args, model, train_loader, optimizer, scheduler):
    '''
    Train for one epoch.
    
    Returns:
        Average loss per token
    '''
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        
        # Training step
        loss, decoder_targets = perform_training_step(model, batch, criterion)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        # Track loss
        with torch.no_grad():
            num_tokens = (decoder_targets != PAD_IDX).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens if total_tokens > 0 else 0

def generate_sql_queries(args, model, dataloader, is_test=False):
    '''
    Generate SQL queries from the model.
    
    Args:
        args: Training arguments
        model: The T5 model
        dataloader: DataLoader for evaluation
        is_test: Whether this is test set (no ground truth)
    
    Returns:
        generated_queries: List of generated SQL queries
        eval_loss: Average loss (None for test set)
    '''
    model.eval()
    generated_queries = []
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX) if not is_test else None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating" if not is_test else "Test inference"):
            if is_test:
                encoder_input, encoder_mask, _ = batch
                encoder_input = encoder_input.to(DEVICE)
                encoder_mask = encoder_mask.to(DEVICE)
            else:
                encoder_input, encoder_mask, decoder_input, decoder_targets, _ = batch
                encoder_input = encoder_input.to(DEVICE)
                encoder_mask = encoder_mask.to(DEVICE)
                decoder_input = decoder_input.to(DEVICE)
                decoder_targets = decoder_targets.to(DEVICE)

                # Compute loss for monitoring
                outputs = model(
                    input_ids=encoder_input,
                    attention_mask=encoder_mask,
                    decoder_input_ids=decoder_input,
                )
                logits = outputs.logits

                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    decoder_targets.reshape(-1)
                )
                
                num_tokens = (decoder_targets != PAD_IDX).sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

            # Generate SQL queries
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=args.max_gen_length,
                num_beams=args.num_beams,
                early_stopping=True,
            )
            
            # Decode generated queries
            for gen_id in generated_ids:
                query = tokenizer.decode(gen_id, skip_special_tokens=True)
                generated_queries.append(query)

    eval_loss = total_loss / total_tokens if total_tokens > 0 and not is_test else None
    return generated_queries, eval_loss

def eval_epoch(args, model, dev_loader, gt_sql_path, model_sql_path, 
               gt_record_path, model_record_path):
    '''
    Evaluate model on dev set.
    
    Returns:
        eval_loss, record_f1, record_em, sql_em, error_rate
    '''
    # Generate SQL queries
    generated_queries, eval_loss = generate_sql_queries(args, model, dev_loader, is_test=False)
    
    # Save generated queries and compute metrics
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)
    
    # Compute metrics
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )
    
    # Compute error rate
    num_errors = sum(1 for msg in error_msgs if msg != "")
    error_rate = num_errors / len(error_msgs) if error_msgs else 0
    
    return eval_loss, record_f1, record_em, sql_em, error_rate

def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    Run inference on test set and save predictions.
    '''
    generated_queries, _ = generate_sql_queries(args, model, test_loader, is_test=True)
    
    # Save predictions
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)
    print(f"Test predictions saved to:")
    print(f"  SQL: {model_sql_path}")
    print(f"  Records: {model_record_path}")

def log_to_wandb(args, epoch, tr_loss, eval_results=None):
    '''
    Log metrics to Weights & Biases if enabled.
    '''
    if not args.use_wandb:
        return
    
    if eval_results is not None:
        eval_loss, record_f1, record_em, sql_em, error_rate = eval_results
        result_dict = {
            'epoch': epoch,
            'train/loss': tr_loss,
            'dev/loss': eval_loss,
            'dev/record_f1': record_f1,
            'dev/record_em': record_em,
            'dev/sql_em': sql_em,
            'dev/error_rate': error_rate,
        }
    else:
        result_dict = {'epoch': epoch, 'train/loss': tr_loss}
    
    wandb.log(result_dict, step=epoch)

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    '''
    Main training loop with evaluation every 5 epochs.
    '''
    best_f1 = -1
    paths = setup_training_paths(args)
    
    for epoch in range(args.max_n_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.max_n_epochs}")
        print(f"{'='*60}")
        
        # Training
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Train loss: {tr_loss:.4f}")

        # Evaluation - only every 5 epochs or on the last epoch
        should_evaluate = (epoch % 5 == 0) or (epoch == args.max_n_epochs - 1)
        
        if should_evaluate:
            # Evaluation
            eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
                args, model, dev_loader, paths['gt_sql'], paths['model_sql'],
                paths['gt_records'], paths['model_records']
            )
            
            print(f"Dev loss: {eval_loss:.4f}")
            print(f"Dev Record F1: {record_f1:.4f}")
            print(f"Dev Record EM: {record_em:.4f}")
            print(f"Dev SQL EM: {sql_em:.4f}")
            print(f"SQL Error Rate: {error_rate*100:.2f}%")

            # Log to wandb
            log_to_wandb(args, epoch, tr_loss, (eval_loss, record_f1, record_em, sql_em, error_rate))

            # Track best model
            if record_f1 > best_f1:
                best_f1 = record_f1
                print(f"New best F1: {best_f1:.4f} - Saving model")
                save_model(args.checkpoint_dir, model, best=True)
            
            # Always save last model when evaluating
            save_model(args.checkpoint_dir, model, best=False)
        else:
            # Log only training loss
            log_to_wandb(args, epoch, tr_loss)
            print(f"Skipping evaluation (evaluating every 5 epochs)")
    
    print(f"\nTraining complete! Best dev F1: {best_f1:.4f}")

def main():
    '''
    Main entry point for training script.
    '''
    args = parse_arguments()
    print(f"\nExperiment: {args.experiment_name}")
    print(f"Mode: {'Fine-tuning' if args.finetune else 'Training from scratch'}")
    print(f"Device: {DEVICE}")
    
    if args.use_wandb:
        setup_wandb(args)

    # Load data
    print("\nLoading data...")
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Initialize model
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Load best model for final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Final dev set evaluation
    model_type = 'ft' if args.finetune else 'scr'
    paths = setup_training_paths(args)
    
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
        args, model, dev_loader, paths['gt_sql'], paths['model_sql'],
        paths['gt_records'], paths['model_records']
    )
    
    print(f"\nFinal Dev Results:")
    print(f"  Loss: {dev_loss:.4f}")
    print(f"  Record F1: {dev_record_f1:.4f}")
    print(f"  Record EM: {dev_record_em:.4f}")
    print(f"  SQL EM: {dev_sql_em:.4f}")
    print(f"  Error Rate: {dev_error_rate*100:.2f}%")

    # Test set inference
    model_sql_path = f'results/t5_{model_type}_{args.experiment_name}_test.sql'
    model_record_path = f'records/t5_{model_type}_{args.experiment_name}_test.pkl'
    test_inference(args, model, test_loader, model_sql_path, model_record_path)
    
    print("\nDone! Submit these files to Gradescope:")
    print(f"  {model_sql_path}")
    print(f"  {model_record_path}")

if __name__ == "__main__":
    main()
