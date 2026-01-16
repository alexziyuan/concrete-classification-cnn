import os, json, random, argparse, pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# -------------------------
# Reproducibility utilities
# -------------------------
def set_seed(seed=13):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# -------------------------
# Data listing & splitting
# -------------------------
def list_images(data_root):
    data_root = pathlib.Path(data_root)
    items = []
    for label_name in sorted(["Negative", "Positive"]):
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            for p in (data_root / label_name).glob(ext):
                items.append((str(p), 0 if label_name=="Negative" else 1))
    return items


def split_paths(items, seed=13, train=0.7, val=0.15, test=0.15):
    assert abs(train + val + test - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    items = np.array(items, dtype=object)
    y = np.array([lbl for _, lbl in items])
    idx = np.arange(len(items))
    # stratified split
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(y):
        cls_idx = idx[y==cls]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        ntr = int(n * train)
        nval = int(n * val)
        train_idx.extend(cls_idx[:ntr])
        val_idx.extend(cls_idx[ntr:ntr + nval])
        test_idx.extend(cls_idx[ntr + nval:])
    return items[train_idx], items[val_idx], items[test_idx]

def make_ds(paths, img_size=(224, 224), batch=64, shuffle=False, seed=13):
    paths = np.array(paths, dtype=object)
    x_paths = paths[:,0]
    y = paths[:, 1].astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((x_paths, y))

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size)
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(8192, seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)



# -------------------------
# Model
# -------------------------
def build_model(input_shape=(224,224,3), num_classes=2, dropout=0.25, l2=0.0, label_smoothing=0.0):
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ], name="aug")

    base = keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = False

    reg = keras.regularizers.l2(l2) if l2 > 0 else None

    inputs = keras.Input(shape=input_shape)
    x = aug(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", kernel_regularizer=reg)(x)
    model = keras.Model(inputs, outputs, name='mbv2_concrete')
    return model, base # no loss here



# -------------------------
# Plotting
# -------------------------
def plot_loss_curves(history, outpath):
    plt.figure()
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss (train vs val)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_names, outpath, title):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha='center',
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
    return cm


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Folder containing Positive/ and Negative/")
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs_stage1", type=int, default=6)
    ap.add_argument("--epochs_stage2", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--l2", type=float, default=0.0)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--make_splits_only", action="store_true", help="Only compute and save splits then exit")
    ap.add_argument("--splits_dir", type=str, default="splits")
    ap.add_argument("--smoke", action="store_true", help="1-epoch tiny test for pipeline sanity check")
    args = ap.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.splits_dir, exist_ok=True)

    # Create or reuse splits
    split_paths_json = [f"{args.splits_dir}/{name}.json" for name in ["train", "val", "test"]]
    if not all(os.path.exists(p) for p in split_paths_json):
        items = list_images(args.data_root)
        tr, va, te = split_paths(items, seed=args.seed)
        for name, data in zip(["train", "val", "test"], [tr, va, te]):
            with open(f"{args.splits_dir}/{name}.json", "w") as f:
                json.dump([[p, int(lbl)] for p, lbl in data], f, indent=2)

    if args.make_splits_only:
        print("Splits created. Exiting as requested.")
        return

    # Build datasets from JSON lists
    def ds_from_json(jpath, img_size=(224, 224), batch=64, shuffle=False, seed=13):
        with open(jpath) as f:
            pairs = json.load(f)
        paths = np.array([p for p, _ in pairs], dtype=object)
        labels = np.array([int(l) for _, l in pairs], dtype=np.int32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

        def _load(p, y):
            x = tf.io.read_file(p)
            x = tf.image.decode_image(x, channels=3, expand_animations=False)
            x = tf.image.resize(x, img_size)
            return x, y

        ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(8192, seed=seed, reshuffle_each_iteration=True)
        return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

    image_size = (args.img_size, args.img_size)
    train_ds = ds_from_json(f"{args.splits_dir}/train.json", img_size=image_size, batch=args.batch, shuffle=True, seed=args.seed)
    val_ds   = ds_from_json(f"{args.splits_dir}/val.json",   img_size=image_size, batch=args.batch, shuffle=False, seed=args.seed)
    test_ds  = ds_from_json(f"{args.splits_dir}/test.json",  img_size=image_size, batch=args.batch, shuffle=False, seed=args.seed)
    class_names = ["Negative", "Positive"]

    if args.smoke:
        # shrink dataset for quick local tests
        train_ds = train_ds.take(2)
        val_ds = val_ds.take(1)
        test_ds = test_ds.take(1)
    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        model, base = build_model(
            input_shape=image_size + (3,),
            num_classes=2,
            dropout=args.dropout,
            l2=args.l2,
            label_smoothing=args.label_smoothing,
        )
        loss_fn = get_loss(num_classes=2, label_smoothing=args.label_smoothing)

        model.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss=loss_fn,
                      metrics=["accuracy"])

    ckpt_path = os.path.join(args.out_dir, "stage1.keras")
    ckpt = keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True,
                                           monitor="val_accuracy", mode="max")
    early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                          restore_best_weights=True)
    hist1 = model.fit(train_ds, validation_data=val_ds,
                      epochs=1 if args.smoke else args.epochs_stage1, callbacks=[ckpt, early])


    # Fine-tune top layers
    base.trainable = True
    for layer in base.layers[:-40]:  # keep most layers frozen
        layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss=loss_fn,
                  metrics=["accuracy"])
    ckpt_path2 = os.path.join(args.out_dir, "stage2.keras")
    ckpt2 = keras.callbacks.ModelCheckpoint(ckpt_path2, save_best_only=True,
                                            monitor="val_accuracy", mode="max")
    hist2 = model.fit(train_ds, validation_data=val_ds,
                      epochs=1 if args.smoke else args.epochs_stage2, callbacks=[ckpt2, early])


    # Curves (train vs val loss on one plot)
    from collections import defaultdict
    H = defaultdict(list)
    for k in ["loss", "val_loss"]:
        H[k] = hist1.history.get(k, []) + hist2.history.get(k, [])
    class Hist: pass
    history = Hist()
    history.history = H
    plot_loss_curves(history, os.path.join(args.out_dir, "loss_curves.png"))

    # Training confusion matrix
    y_train_true = np.concatenate([y.numpy() for _, y in train_ds], axis=0)
    y_train_prob = model.predict(train_ds, verbose=0)
    y_train_pred = y_train_prob.argmax(1)
    cm_train = save_confusion_matrix(
        y_train_true, y_train_pred, class_names,
        os.path.join(args.out_dir, "confusion_matrix_train.png"),
        "Training Confusion Matrix"
    )

    # Test metrics + confusion matrix
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    y_test_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    y_test_prob = model.predict(test_ds, verbose=0)
    y_test_pred = y_test_prob.argmax(1)
    cm_test = save_confusion_matrix(
        y_test_true, y_test_pred, class_names,
        os.path.join(args.out_dir, "confusion_matrix_test.png"),
        "Test Confusion Matrix"
    )

    # Write metrics.txt (include accuracy + confusion matrix)
    with open(os.path.join(args.out_dir, "metrics.txt"), "w") as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy {test_acc:.4f}\n")
        f.write("Test Confusion Matrix (rows=true, cols=pred):\n")
        for row in cm_test:
            f.write(" ".join(map(str,row)) + "\n")

    # Save model
    model.save(os.path.join(args.out_dir, "model.keras"))
    print("Done. Artifacts saved to:", args.out_dir)

if __name__== "__main__":
    main()













