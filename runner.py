import matplotlib.pyplot as matplot
import numpy as numpy
import pandas as pandas
import seaborn as seaborn
import tensorflow as tf
from keras import regularizers
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input, Dense
from keras.models import Model, load_model
from pylab import rcParams
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 314  # used to help randomly select the data points
TEST_DATA_SIZE = 0.2  # 20% of the data
LABELS = ["Normal", "Fraud"]


def analyze_data(fraud_file):
    count_classes = pandas.value_counts(fraud_file['Class'], sort=True)

    print("\nShape is:")
    print(count_classes)

    normal_df = fraud_file[fraud_file.Class == 0]  # save normal_df observations into a separate fraud_file
    fraud_df = fraud_file[fraud_file.Class == 1]  # do the same for frauds

    print("=================================")
    print("Normal Transaction Amount Summary")
    print("=================================")
    print(normal_df.Amount.describe())
    print("\n")
    print("=================================")
    print("Fraud Transaction Amount Summary")
    print("=================================")
    print(fraud_df.Amount.describe())
    print("=================================")

    count_classes.plot(kind='bar', rot=0)
    matplot.xticks(range(2), LABELS)
    matplot.title("Frequency by observation number")
    matplot.xlabel("Class")
    matplot.ylabel("Number of Observations");
    matplot.show()

    bins = numpy.linspace(200, 2500, 100)
    matplot.hist(normal_df.Amount, bins, alpha=1, normed=True, label='Normal')
    matplot.hist(fraud_df.Amount, bins, alpha=0.6, normed=True, label='Fraud')
    matplot.legend(loc='upper right')
    matplot.title("Amount by percentage of transactions (transactions \$200+)")
    matplot.xlabel("Transaction amount (USD)")
    matplot.ylabel("Percentage of transactions (%)");
    matplot.show()

    bins = numpy.linspace(0, 48, 48)  # 48 hours
    matplot.hist((normal_df.Time / (60 * 60)), bins, alpha=1, normed=True, label='Normal')
    matplot.hist((fraud_df.Time / (60 * 60)), bins, alpha=0.6, normed=True, label='Fraud')
    matplot.legend(loc='upper right')
    matplot.title("Percentage of transactions by hour")
    matplot.xlabel("Transaction time as measured from first transaction in the dataset (hours)")
    matplot.ylabel("Percentage of transactions (%)");
    matplot.show()

    # Visual Exploration of Transaction Amount vs. Hour
    matplot.scatter((normal_df.Time / (60 * 60)), normal_df.Amount, alpha=0.6, label='Normal')
    matplot.scatter((fraud_df.Time / (60 * 60)), fraud_df.Amount, alpha=0.9, label='Fraud')
    matplot.title("Amount of transaction by hour")
    matplot.xlabel("Transaction time as measured from first transaction in the dataset (hours)")
    matplot.ylabel('Amount (USD)')
    matplot.legend(loc='upper right')
    matplot.show()


def plot_confusion_matrix(conf_matrix):
    matplot.figure(figsize=(12, 12))
    seaborn.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    matplot.title("Confusion matrix")
    matplot.ylabel('True class')
    matplot.xlabel('Predicted class')
    matplot.show()


def plot_metrics(error_df):
    # ROC Curve Check
    false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df.True_class, error_df.Reconstruction_error)
    roc_auc = auc(false_pos_rate, true_pos_rate, )

    matplot.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f' % roc_auc)
    matplot.plot([0, 1], [0, 1], linewidth=5)
    matplot.xlim([-0.01, 1])
    matplot.ylim([0, 1.01])
    matplot.legend(loc='lower right')
    matplot.title('Receiver operating characteristic curve (ROC)')
    matplot.ylabel('True Positive Rate')
    matplot.xlabel('False Positive Rate')
    matplot.show()

    # Recall vs. Precision Thresholding
    precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
    matplot.plot(recall_rt, precision_rt, linewidth=5, label='Precision-Recall curve')
    matplot.title('Recall vs Precision')
    matplot.xlabel('Recall')
    matplot.ylabel('Precision')
    matplot.show()

    # Precision and recall
    matplot.plot(threshold_rt, precision_rt[1:], label="Precision", linewidth=5)
    matplot.plot(threshold_rt, recall_rt[1:], label="Recall", linewidth=5)
    matplot.title('Precision and recall for different threshold values')
    matplot.xlabel('Threshold')
    matplot.ylabel('Precision/Recall')
    matplot.legend()
    matplot.show()


def plot_loss(history):
    matplot.plot(history['loss'], linewidth=2, label='Train')
    matplot.plot(history['val_loss'], linewidth=2, label='Test')
    matplot.legend(loc='upper right')
    matplot.title('Model Loss')
    matplot.ylabel('Loss')
    matplot.xlabel('Epoch')
    matplot.show()


def start_trainer():
    print('Using tensforflow ' + tf.__version__)

    #--------------------------- SETTING PLOTTERS ----------------------------
    # set up graphic style in this case I am using the color scheme from xkcd.com
    seaborn.set
    rcParams['figure.figsize'] = 14, 8.7  # Golden Mean

    col_list = ["cerulean", "scarlet"]  # https://xkcd.com/color/rgb/
    # Set color_codes to False there is a bug in Seaborn 0.9.0 -- https://github.com/mwaskom/seaborn/issues/1546
    seaborn.set(style='white', font_scale=1.75, palette=seaborn.xkcd_palette(col_list), color_codes=False)

    tensorboard = TensorBoard(log_dir='./logs',
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)

    fraud_file = pandas.read_csv("creditcard.csv")  # unzip and read in data downloaded to the local directory

    #analyze_data(fraud_file)

    data = fraud_file.drop(['Time'], axis=1)
    df_norm = fraud_file
    df_norm['Time'] = StandardScaler().fit_transform(df_norm['Time'].values.reshape(-1, 1))
    df_norm['Amount'] = StandardScaler().fit_transform(df_norm['Amount'].values.reshape(-1, 1))

    # Dividing Training and Test Set
    train_x, test_x = train_test_split(df_norm, test_size=TEST_DATA_SIZE, random_state=RANDOM_SEED)
    train_x = train_x[train_x.Class == 0]  # where normal transactions
    train_x = train_x.drop(['Class'], axis=1)  # drop the class column

    test_y = test_x['Class']  # save the class column for the test set
    test_x = test_x.drop(['Class'], axis=1)  # drop the class column

    train_x = train_x.values  # transform to ndarray
    test_x = test_x.values

    print("train_x.shape = " + str(train_x.shape))

    # Setup layers
    nb_epoch = 10
    batch_size = 128
    input_dim = train_x.shape[1]  # num of columns, 30
    encoding_dim = 14
    hidden_dim = int(encoding_dim / 2)  # i.e. 7
    learning_rate = 1e-7

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(learning_rate))(input_layer)
    encoder = Dense(hidden_dim, activation="relu")(encoder)
    decoder = Dense(hidden_dim, activation='tanh')(encoder)
    decoder = Dense(input_dim, activation='relu')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)

    # ----------------- STARTING TRAINING -------------------------------
    autoencoder.compile(metrics=['accuracy'],
                        loss='mean_squared_error',
                        optimizer='adam')

    cp = ModelCheckpoint(filepath="autoencoder_fraud.h5",
                         save_best_only=True,
                         verbose=0)

    history = autoencoder.fit(train_x, train_x,
                              epochs=nb_epoch,
                              batch_size=batch_size,
                              shuffle=True,
                              validation_data=(test_x, test_x),
                              verbose=1,
                              callbacks=[cp, tensorboard]).history

    autoencoder = load_model('autoencoder_fraud.h5')
    #--------------------- FINISHED TRAINING ------------------------------

    plot_loss(history)

    # Reconstruction Error Check
    test_x_predictions = autoencoder.predict(test_x)
    mse = numpy.mean(numpy.power(test_x - test_x_predictions, 2), axis=1)
    error_df = pandas.DataFrame({'Reconstruction_error': mse,
                                 'True_class': test_y})

    print("Reconstruction Error Check")
    print(error_df.describe())

    plot_metrics(error_df)

    # Reconstruction Error vs Threshold Check
    threshold_fixed = 5
    groups = error_df.groupby('True_class')
    fig, ax = matplot.subplots()

    for name, group in groups:
        ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
                label="Fraud" if name == 1 else "Normal")
    ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
    ax.legend()
    matplot.title("Reconstruction error for different classes")
    matplot.ylabel("Reconstruction error")
    matplot.xlabel("Data point index")
    matplot.show();

    # Confusion Matrix
    pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]
    conf_matrix = confusion_matrix(error_df.True_class, pred_y)
    plot_confusion_matrix(conf_matrix)


if __name__ == "__main__":
    start_trainer()


