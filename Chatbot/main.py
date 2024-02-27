import torch
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel,TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Define the folder path where you want to save the files
folder_path = "//Desktop-72n0tt8/d/gptxl"
folder_path1 ="E:/python/Chatbot/gptdata/gptxl"
# Create the folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

# Load pre-trained model and tokenizer and specify the folder path
model_name = "gpt2"  # or any other GPT variant
tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=folder_path1)
model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=folder_path1)
# Define the block size
block_size = 128  # Adjust as needed based on your training data and model configuration




model.eval()
# Open a file to log generated text
log_file = open("generated_text_log.txt", "w")

# Function to generate text based on user input
def generate_text_from_input(max_length=500):
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Exiting...")
            break
        input_ids = tokenizer.encode(user_input, return_tensors="pt")
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)  # Set attention mask to ones
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, 
                        pad_token_id=tokenizer.eos_token_id, attention_mask=attention_mask,do_sample=True,
                        temperature=0.9, top_p=0.95)  # Adjust temperature as needed

        generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
        generated_text = post_process(generated_text)

        print("AI:", generated_text)

        # Post-processing function to reduce repetition
def post_process(text):
    # Split the text into sentences
    sentences = text.split('. ')
    
    # Remove duplicates
    unique_sentences = list(set(sentences))
    
    # Join the unique sentences back together
    return '. '.join(unique_sentences)

        

        
        
# Load and tokenize the fine-tuning dataset
train_dataset = TextDataset(tokenizer=tokenizer, file_path="train_dataset.txt",block_size=block_size)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2
)

# Define the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine_tuned_model")

# Call the function
generate_text_from_input()

# Close the log file when done

