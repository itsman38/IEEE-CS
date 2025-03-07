import numpy.random
import pandas as pd
import matplotlib.pyplot as plt
import idx2numpy
import keras
from keras.src.layers import Dense
from keras.src.losses import CategoricalCrossentropy
from keras import Sequential
import shap

df = pd.read_csv("./dataset_aiml_task/data.csv")
df_unique = df.groupby(df.columns[0]).first().reset_index()  # In the dataset, first column is the label


def plot_figure():
    # Plot the first 10 unique images
    fig, axes = plt.subplots(1, 10, figsize=(12, 3))

    for i in range(10):
        label = df_unique.iloc[i, 0]  # First column is the label
        image = df_unique.iloc[i, 1:].values.reshape(28, 28)

        axes[i].imshow(image, cmap="gray")
        axes[i].set_title(f"Label {label}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def stats():
    image_file = "./dataset_aiml_task/images-idx3-ubyte"
    images = idx2numpy.convert_from_file(image_file)
    print(f"Loaded {len(images)} images with shape {images.shape}")
    print(f"the df shape is {df.shape}")


def visualize():
    # Describe pixel values (excluding label column)
    pixel_stats = df.iloc[:, 1:].describe().T  # Transpose for readability
    print(pixel_stats)

    plt.hist(df.iloc[:, 1:].values.flatten(), bins=50, color="purple", alpha=0.7)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.title("Pixel Intensity Distribution")
    plt.show()


def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label="Train Accuracy")
    plt.plot(history.history['val_accuracy'], label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} - Accuracy")
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def visualize_predictions(x_test, y_pred, y_true, model_name):
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle(f"Predictions - {model_name}", fontsize=16)

    for i, ax in enumerate(axes.flat):
        image = x_test[i].reshape(28, 28)
        predicted_label = y_pred[i].argmax()
        true_label = y_true[i].argmax()

        ax.imshow(image, cmap="gray")
        ax.set_title(f"Prediction: {predicted_label}, True: {true_label}")
        ax.axis("off")

    plt.show()


def analyze(model, x_test, x_train):
    explainer = shap.DeepExplainer(model, x_train[:500])
    shap_values = explainer.shap_values(x_test[:200])

    feature_names = [f"Pixel {i}" for i in range(x_test.shape[1])]
    shap.summary_plot(shap_values, x_test[:200], feature_names=feature_names)


def create_model():
    x = df.iloc[:, 1:].values / 255
    y = df.iloc[:, 0].values

    y_one_hot = keras.utils.to_categorical(y)
    shuffle = numpy.random.permutation(len(x))
    x, y_one_hot = x[shuffle], y_one_hot[shuffle]

    idx = int(len(x) * 0.8)
    x_train, y_train = x[:idx], y_one_hot[:idx]
    x_test, y_test = x[idx:], y_one_hot[idx:]

    logistic_model = Sequential([
        Dense(10, "softmax")
    ])
    logistic_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                   loss=CategoricalCrossentropy(from_logits=False),
                   metrics=["accuracy"])

    logistic_history = logistic_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))  # Track training history
    plot_training_history(logistic_history, "Logistic Model")  # Plot training history
    logistic_predict = logistic_model.predict(x_test)
    visualize_predictions(x_test, logistic_predict, y_test, "Logistic Model")


    neural_network_model = Sequential([
        Dense(128, "sigmoid"),
        Dense(64, "sigmoid"),
        Dense(10, "softmax")
    ])

    neural_network_model.compile(keras.optimizers.Adam(learning_rate=1e-3),
                  loss=CategoricalCrossentropy(from_logits=False),
                  metrics=["accuracy"])

    neural_network_history = neural_network_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    plot_training_history(neural_network_history, "Neural Network")  # Plot training history
    y_predict = neural_network_model.predict(x_test)
    visualize_predictions(x_test, y_predict, y_test, "Neural Network")

    y_predict = neural_network_model.predict(x_test)
    print(y_predict)

    analyze(model, x_test, x_train)


def main():
    plot_figure()
    stats()
    visualize()
    create_model()


if __name__ == "__main__":
    main()
