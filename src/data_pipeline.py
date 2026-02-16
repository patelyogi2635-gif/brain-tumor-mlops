import tensorflow as tf


IMAGE_SIZE = 300
BATCH_SIZE = 32

def load_data(train_dir,test_dir):

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    return train_ds,test_ds