import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# DATASET FROM https://www.kaggle.com/datasets/scolianni/mnistasjpg

def load_images_and_labels(base_path):
    images = []
    labels = []

    folderNames = []

    for i in range(10):
        folderNames.append(str(i))

    # Iterate through each label folder (0-9)
    for label in folderNames:
        label_folder = os.path.join(base_path, label)
        
        # Iterate through each image in the folder
        for image_name in os.listdir(label_folder):

            # Get the image path
            image_path = os.path.join(label_folder, image_name)
            
            # Load the image
            image = Image.open(image_path)
            # Convert to grayscale (pixels between 0 and 255)
            image = image.convert('L')
            # Resize the images to 20 x 20
            image = image.resize((20, 20))

            # Convert values to np array
            image_array = np.array(image)
            
            # Append the image and label to the lists
            images.append(image_array)
            labels.append(folderNames.index(label))
    
    return np.array(images), np.array(labels)



print("Loading images....")

path = 'C:/Users/agham/Desktop/Docs/Projects/MLprojects/0_9DigitNN'
x, y = load_images_and_labels(path)

print("Images Loaded.")

print(f'X shape = {x.shape}')
print(f'Y shape = {y.shape}')

# Identify the number of samples in each class
unique, counts = np.unique(y, return_counts=True)
class_counts = dict(zip(unique, counts))
print(f'Class counts before balancing: {class_counts}')

# Find the minimum number of samples in any class
min_samples = min(class_counts.values())

# Create lists to store the balanced dataset
x_balanced = []
y_balanced = []

# Iterate through each class and undersample to the minimum number of samples
for label in range(10):
    # Extract all samples of the current class
    x_class = x[y == label]
    y_class = y[y == label]
    
    # Undersample if there are more than min_samples samples
    x_class_undersampled = x_class[:min_samples]
    y_class_undersampled = y_class[:min_samples]
    
    # Append the undersampled data to the balanced dataset lists
    x_balanced.append(x_class_undersampled)
    y_balanced.append(y_class_undersampled)

# Convert lists back to numpy arrays
x_balanced = np.vstack(x_balanced)
y_balanced = np.hstack(y_balanced)

# Shuffle the balanced dataset
x_balanced, y_balanced = shuffle(x_balanced, y_balanced, random_state=42)

print(f'Balanced X shape = {x_balanced.shape}')
print(f'Balanced Y shape = {y_balanced.shape}')


# First split: training + validation and test sets
x_temp, x_test, y_temp, y_test = train_test_split(x_balanced, y_balanced, test_size=0.2, random_state=42)

# Second split: training and validation sets from the temp set
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, random_state=42)

datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
)

x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0


datagen.fit(x_train.reshape(-1, 20, 20, 1))

model = Sequential([
    tf.keras.Input(shape=(20, 20, 1)),
    Flatten(),
    Dense(units=400, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(units=200, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(units=100, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(units=50, activation='relu'),
    Dense(units=10, activation='softmax')
])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.0001),
    metrics=['accuracy']
)


early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(
    datagen.flow(x_train.reshape(-1, 20, 20, 1), y_train, batch_size=32),
    validation_data=(x_val.reshape(-1, 20, 20, 1), y_val),
    epochs=50,
    callbacks=[early_stopping],
)

unique, counts = np.unique(y_balanced, return_counts=True)
majority_class_count = max(counts)
total_samples = sum(counts)
baseline_accuracy = majority_class_count / total_samples

train_loss, train_accuracy = model.evaluate(x_train.reshape(-1, 20, 20, 1), y_train, verbose=0)
val_loss, val_accuracy = model.evaluate(x_val.reshape(-1, 20, 20, 1), y_val, verbose=0)
test_loss, test_accuracy = model.evaluate(x_test.reshape(-1, 20, 20, 1), y_test, verbose=0)

print(f"Baseline accuracy: {baseline_accuracy:.4f}")
print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Validation accuracy: {val_accuracy:.4f}")


fig, axes = plt.subplots(8,8, figsize=(8,8))
fig.tight_layout(pad=0.1,rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top]

for i,ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(len(x_test))
    
    # Display the image
    ax.imshow(x_test[random_index].reshape(20, 20), cmap='gray')

    # Predict using the Neural Network
    prediction = model.predict(x_test[random_index].reshape(1, 20, 20, 1), verbose=0)
    yhat = np.argmax(prediction)  # Get the class with the highest probability
    
    # Display the label above the image
    ax.set_title(f"{str(y_test[random_index])},{yhat}")
    ax.set_axis_off()
fig.suptitle("Label, yhat", fontsize=16)

plt.show()
