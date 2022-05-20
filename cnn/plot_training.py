import matplotlib.pyplot as plt
import pickle
import glob
import os

def plot(error_dict, dir_path):
    f = open(f"{dir_path}/loss_accuracies.txt", "w")
    
    num_epochs = len(error_dict['train_acc_1'])
    iters = list(range(num_epochs))
    
    # Loss
    plt.figure()
    plt.title("Training Curve - Loss")
    plt.plot(iters, error_dict['train_loss'], label = "Training")
    plt.plot(iters, error_dict['valid_loss'], label = "Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.savefig(f"{dir_path}/loss.png")
    f.write("Final Training Loss: {}\n".format(error_dict['train_loss'][-1]))
    f.write("Final Validation Loss: {}\n".format(error_dict['valid_loss'][-1]))
    f.write("\n")

    # Accuracy
    plt.figure()
    plt.title("Training Curve - Accuracy")
    plt.plot(iters, error_dict['train_acc_1'], label = "Top 1 - Training")
    plt.plot(iters, error_dict['valid_acc_1'], label = "Top 1 - Validation")
    plt.plot(iters, error_dict['train_acc_5'], label = "Top 5 - Training")
    plt.plot(iters, error_dict['valid_acc_5'], label = "Top 5 - Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.savefig(f"{dir_path}/accuracy.png")
    f.write("Final Training Accuracy (top 1): {}\n".format(error_dict['train_acc_1'][-1]))
    f.write("Final Validation Accuracy (top 1): {}\n".format(error_dict['valid_acc_1'][-1]))
    f.write("Final Training Accuracy (top 5): {}\n".format(error_dict['train_acc_5'][-1]))
    f.write("Final Validation Accuracy (top 5): {}\n".format(error_dict['valid_acc_5'][-1]))
    
    f.close()
    

if __name__ == '__main__':
    save_paths = [f for f in glob.glob("github/ConvSearch/save_data*")]
    
    for path in save_paths:
        error_dict_path = path + "/errors_dict.pkl"

        if os.path.exists(error_dict_path):
            print("Writing to:", path)
            
            error_dict = {}
            with open(error_dict_path, 'rb') as f:
                error_dict = pickle.load(f)    
            plot(error_dict, path)
        
    
    
    
    