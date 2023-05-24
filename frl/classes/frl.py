import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow import keras
import json
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

EXPLOITS_TREE_PATH = '/home/sess/Desktop/KLTN/A3C/frl/exploits_tree.json'
def loadExploitsTree(detailed=True):
    exploits_tree = json.loads(open(EXPLOITS_TREE_PATH, "r").read())
    if detailed:
        return exploits_tree
    return [_['exploit'] for _ in exploits_tree]

state_size = 1
exploits_array = loadExploitsTree(detailed=False)
action_size = len(exploits_array)


def build_model(state_size, action_space):
    input_layer = layers.Input(batch_shape=(None, state_size))
    dense_layer1 = layers.Dense(50, activation='relu')(input_layer)
    dense_layer2 = layers.Dense(100, activation='relu')(dense_layer1)
    dense_layer3 = layers.Dense(200, activation='relu')(dense_layer2)
    dense_layer4 = layers.Dense(400, activation='relu')(dense_layer3)
    out_actions = layers.Dense(
        action_space, activation='softmax')(dense_layer4)
    out_value = layers.Dense(1, activation='linear')(dense_layer4)
    model = keras.Model(inputs=[input_layer], outputs=[out_actions, out_value])
    model.make_predict_function()
    return model

class FederatedLearning:
    def __init__(self, models):
        self.models = models  #list of client model
        self.name ='main'
        save_dir = "./"
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.opt = tf.keras.optimizers.RMSprop(0.005, .99)
        self.global_model = build_model(                        
            state_size, action_size)  # global network
        self.global_model(tf.convert_to_tensor(
            np.random.random((1, state_size)), dtype=tf.float32))
        try:
            model_path = os.path.join(self.save_dir, f'pwn_model_{self.name}.h5')
            print('Loading model from: {}'.format(model_path))
            self.global_model.load_weights(model_path)
        except:
            # Doesn't exist
            pass

    def update_main_agent(self):
        print("=============Start collect weights==============")
        local_weights = [model.get_weights() for model in self.models]
        
        new_weights = []
        for main_param, local_params in zip(self.global_model.trainable_weights, zip(*local_weights)):
            weighted_avg = tf.zeros_like(local_params[0])
            for local_param in local_params:    
                weighted_avg += local_param
            weighted_avg /= len(self.models) # Chia tổng trọng số cho số lượng mạng mô hình local
            
            new_weights.append(weighted_avg.numpy())
        print("=============Apply weights to main model==============")
        self.global_model.set_weights(new_weights)
        print("=============Finished update==============")

def collect_models(model_paths):
    models = []  # Danh sách chứa các mô hình

    for model_path in model_paths:
        try:
            model = build_model(state_size, action_size)
            model.load_weights(model_path)
            models.append(model)
            print(f"Finished collect model from file: {model_path}")
        except Exception as e:
            print(f"Fail to load model from file {model_path}: {str(e)}")

    return models

def main():
    ROOT = '/home/sess/Desktop/KLTN/A3C'
    # file_paths = ['path_to_client1.h5', 'path_to_client2.h5', '...']
    file_paths = [f'{ROOT}/pwn_model_client_1.h5', f'{ROOT}/pwn_model_client_2.h5', f'{ROOT}/pwn_model_client_0.h5']
    models = collect_models(file_paths)
    fl = FederatedLearning(models)
    fl.update_main_agent()





if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        sys.stdout.flush()
        PPrint().info("SIGINT received.")
        PPrint().info("Exiting...")
        os._exit(1)
