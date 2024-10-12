import json
import matplotlib.pyplot as plt


trainer_file_path = "/content/drive/MyDrive/BrAIcht/log_output/checkpoint-500/trainer_state.json"
save_fig_path = "/content/drive/MyDrive/BrAIcht/train_validation_loss_plot/train_validation_loss.png"


with open(trainer_file_path, 'r') as f:
    trainer_state = json.load(f)

log_history = trainer_state['log_history']
training_losses = [entry['loss'] for entry in log_history if 'loss' in entry]
validation_entries = [entry for entry in log_history if 'eval_loss' in entry]
validation_losses = [entry['eval_loss'] for entry in validation_entries]
epoch_values = [entry['epoch'] for entry in validation_entries]

plt.figure(figsize=(10, 6))
plt.plot(epoch_values, training_losses, label='Training Loss')
plt.plot(epoch_values, validation_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.grid(True)

plt.savefig(save_fig_path)

plt.show()