from Model_and_DataLoader import *
import matplotlib.pyplot as plt
import datetime


"""
This is training part, to train model to recognize hand landmarks.
"""


def main():
    model = CSV_Dense_Model()
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()])
    # load data
    datas, labels = Data_Loader.load_csv()
    labels = tf.one_hot(labels, 12)
    print("label shape: ", labels.shape)
    datas = datas[:, :, np.newaxis]
    # reshape datas
    datas = tf.transpose(datas, perm=[0, 2, 1])
    # calculate quantity of data and use 80% of them to train, 20% to test
    train_data, train_labels = datas[:int(len(datas) * 0.8)], labels[:int(len(labels) * 0.8)]
    test_data, test_labels = datas[int(len(datas) * 0.8):], labels[int(len(labels) * 0.8):]

    train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(200)
    test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(200)

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.CategoricalCrossentropy()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    # train
    @tf.function
    def train_step(images, temp_labels):
        with tf.GradientTape() as tape:
            print("data shape: ", images.shape)
            predictions = model(images)
            print("prediction shape: ", predictions.shape)
            print("label shape: ", temp_labels.shape)
            loss = loss_object(temp_labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(temp_labels, predictions)

    # test
    @tf.function
    def test_step(images, temp_labels):
        predictions = model(images)
        t_loss = loss_object(temp_labels, predictions)

        test_loss(t_loss)
        test_accuracy(temp_labels, predictions)


    # start training
    print("Start training...")
    epochs = 1000
    losses = []
    accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        # reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for input_data, input_label in train_ds:
            train_step(input_data, input_label)
        for input_t_data, input_t_label in test_ds:
            test_step(input_t_data, input_t_label)

        losses.append(train_loss.result())
        accuracies.append(train_accuracy.result())
        test_losses.append(test_loss.result())
        test_accuracies.append(test_accuracy.result())

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))


    # draw graph
    plt.subplot(2, 1, 1)
    plt.ylim(0, np.max(losses))
    plt.plot(range(1, epochs + 1), losses)
    plt.title("Loss")
    plt.subplot(2, 1, 2)
    plt.ylim(0, 1)
    plt.plot(range(1, epochs + 1), accuracies)
    plt.title("Accuracy")
    plt.show()

    plt.subplot(2, 1, 1)
    plt.ylim(0, np.max(test_losses))
    plt.plot(range(1, epochs + 1), test_losses)
    plt.title("Test Loss")
    plt.subplot(2, 1, 2)
    plt.ylim(0, 1)
    plt.plot(range(1, epochs + 1), test_accuracies)
    plt.title("Test Accuracy")
    plt.show()

    # save model
    model.save('Training_Model1')



if __name__ == '__main__':
    main()