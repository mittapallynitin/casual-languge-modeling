import math
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, get_scheduler

from data_loader import get_data_loaders
from model_config import create_model

# Configuration
NUM_EPOCHS = 3
BATCH_SIZE = 32
MAX_LEN = 512
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5
SAVE_PATH = "./code_generator_model"
LOG_DIR = "./tensorboard_logs"

# Check for GPU/Device
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def calculate_perplexity(loss):
    """
    Calculate perplexity from loss.
    """
    return math.exp(loss) if loss < 300 else float("inf")

def train_one_epoch(epoch, model, train_loader, optimizer, scheduler, gradient_accumulation_steps, writer):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0
    step_loss = 0
    step_count = 0

    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} Training")):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["input_ids"].to(DEVICE)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / gradient_accumulation_steps
        total_loss += loss.item()
        step_loss += loss.item()

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

            # Log step loss to TensorBoard
            writer.add_scalar("Loss/Train Step", step_loss / step_count, epoch * len(train_loader) + step)
            step_loss = 0

    # Adjust learning rate
    scheduler.step()

    # Calculate epoch metrics
    avg_loss = total_loss / len(train_loader)
    perplexity = calculate_perplexity(avg_loss)

    # Log epoch metrics
    writer.add_scalar("Loss/Train Epoch", avg_loss, epoch + 1)
    writer.add_scalar("Perplexity/Train Epoch", perplexity, epoch + 1)

    return avg_loss, perplexity

def validate_one_epoch(epoch, model, val_loader, writer):
    """
    Validate the model for one epoch.
    """
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch + 1} Validation")):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()

    # Calculate validation metrics
    avg_val_loss = total_val_loss / len(val_loader)
    perplexity = calculate_perplexity(avg_val_loss)

    # Log validation metrics
    writer.add_scalar("Loss/Validation", avg_val_loss, epoch + 1)
    writer.add_scalar("Perplexity/Validation", perplexity, epoch + 1)

    return avg_val_loss, perplexity

def save_model(epoch, model, tokenizer, save_path):
    """
    Save the model and tokenizer after each epoch.
    """
    epoch_path = os.path.join(save_path, f"epoch_{epoch + 1}")
    os.makedirs(epoch_path, exist_ok=True)
    model.save_pretrained(epoch_path)
    tokenizer.save_pretrained(epoch_path)

def main():
    # Create model, tokenizer, optimizer, and scheduler
    model, tokenizer = create_model()  # Your custom model creation logic
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    train_loader, val_loader, _ = get_data_loaders(tokenizer, MAX_LEN)
    num_training_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=LOG_DIR)

    for epoch in range(NUM_EPOCHS):
        print(f"\n===== Epoch {epoch + 1}/{NUM_EPOCHS} =====")

        # Training
        train_loss, train_perplexity = train_one_epoch(
            epoch, model, train_loader, optimizer, scheduler, GRADIENT_ACCUMULATION_STEPS, writer
        )
        print(f"Training Loss: {train_loss:.4f}, Training Perplexity: {train_perplexity:.4f}")

        # Validation
        val_loss, val_perplexity = validate_one_epoch(epoch, model, val_loader, writer)
        print(f"Validation Loss: {val_loss:.4f}, Validation Perplexity: {val_perplexity:.4f}")

        # Save model
        save_model(epoch, model, tokenizer, SAVE_PATH)

    writer.close()
    print("Training Complete!")

if __name__ == "__main__":
    main()