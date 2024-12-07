import numpy as np
import pandas as pd
import sys
import logging
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Input, Embedding, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_shadow_datasets(aux_data, target_record, num_shadow_datasets):
    shadow_datasets = []
    for _ in range(num_shadow_datasets):
        shadow_dataset = aux_data.sample(n=len(aux_data)-1, replace=False)
        if np.random.rand() > 0.5:
            shadow_dataset = shadow_dataset._append(target_record)
            label = 'IN'
        else:
            random_record = aux_data.sample(n=1)
            shadow_dataset = shadow_dataset._append(random_record)
            label = 'OUT'
        shadow_datasets.append((shadow_dataset, label))
    return shadow_datasets

def build_generator(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(output_dim, activation='tanh'))
    return model

def build_discriminator(input_dim):
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

def prepare_data(data):
    # Convert the data to a DataFrame if it's not already
    data_df = pd.DataFrame(data)
    
    # Convert categorical columns to numeric using LabelEncoder
    label_encoders = {}
    for column in data_df.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        data_df[column] = label_encoders[column].fit_transform(data_df[column])
    
    return data_df

def train_gan(generator, discriminator, combined, data, epochs=1000, batch_size=32, sample_interval=1000):
    half_batch = int(batch_size / 2)
    data_df = prepare_data(data)  # Prepare the data
    numeric_data = data_df.select_dtypes(include=[np.number])  # Select only numeric columns
    input_dim = numeric_data.shape[1]  # Get the number of numeric columns

    if input_dim == 0:
        raise ValueError("No numeric columns found in the dataset. Ensure the dataset contains numeric data.")

    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, numeric_data.shape[0], half_batch)
        real_data = numeric_data.iloc[idx].astype(np.float32).values  # Convert real_data to float and NumPy array
        noise = np.random.normal(0, 1, (half_batch, generator.input_shape[1]))
        gen_data = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_data, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_data, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, generator.input_shape[1]))
        valid_y = np.array([1] * batch_size)
        g_loss = combined.train_on_batch(noise, valid_y)

        if epoch % sample_interval == 0:
            logging.info(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100*d_loss[1]}] [G loss: {g_loss}]")

def train_shadow_models(shadow_datasets):
    shadow_synthetic_datasets = []
    for shadow_dataset, label in tqdm(shadow_datasets, desc="Training shadow models"):
        # Prepare data
        data = shadow_dataset.values
        input_dim = data.shape[1]

        # Build and compile the discriminator
        discriminator = build_discriminator(input_dim)
        discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

        # Build the generator
        generator = build_generator(100, input_dim)

        # The generator takes noise as input and generates data
        z = Input(shape=(100,))
        gen_data = generator(z)

        # For the combined model, only train the generator
        discriminator.trainable = False

        # The discriminator takes generated data as input and determines validity
        valid = discriminator(gen_data)

        # The combined model (stacked generator and discriminator)
        combined = Model(z, valid)
        combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

        # Train the GAN
        train_gan(generator, discriminator, combined, data)

        # Generate synthetic data
        noise = np.random.normal(0, 1, (len(data), 100))
        synthetic_data = generator.predict(noise)
        synthetic_df = pd.DataFrame(synthetic_data, columns=shadow_dataset.columns)

        shadow_synthetic_datasets.append((synthetic_df, label))
    return shadow_synthetic_datasets

def predict_with_indices(model, data):
    predictions = model.predict(data)
    in_indices = [i for i, pred in enumerate(predictions) if pred > 0.5]
    return in_indices

def main(synthetic_data_path, auxiliary_data_path, synthetic_data_save_path=None):
    logging.info("Loading datasets...")
    synthetic_data = pd.read_csv(synthetic_data_path)
    auxiliary_data = pd.read_csv(auxiliary_data_path)

    if synthetic_data_save_path and os.path.exists(synthetic_data_save_path): 
        logging.info("Loading synthetic data from file...") 
        synthetic_data = pd.read_csv(synthetic_data_save_path) 
    else: 
        logging.info("Creating shadow datasets...") 
        target_record = synthetic_data.iloc[0] # Example target record 
        num_shadow_datasets = 100
        shadow_datasets = create_shadow_datasets(auxiliary_data, target_record, num_shadow_datasets) 
        logging.info("Training shadow models and generating synthetic datasets...") 
        shadow_synthetic_datasets = train_shadow_models(shadow_datasets) 

        # Concatenate all DataFrames in the list into a single DataFrame 
        synthetic_data_combined = pd.concat([df for df, label in shadow_synthetic_datasets])
        
        if synthetic_data_save_path: 
            logging.info("Saving synthetic data to file...") 
            synthetic_data_combined.to_csv(synthetic_data_save_path, index=False)
        else:
            logging.info("Saving synthetic data to default file...") 
            synthetic_data_combined.to_csv("synthetic_data_save_path.csv", index=False)

    logging.info("Preparing data for meta-classifier...")
    X = []
    y = []
    for shadow_synthetic_dataset, label in shadow_synthetic_datasets:
        # Extract numeric features from the synthetic dataset (e.g., mean, std, etc.)
        numeric_features = shadow_synthetic_dataset.select_dtypes(include=[np.number]).mean().values
        X.append(numeric_features)
        y.append(1 if label == 'IN' else 0)

    X = np.array(X)
    y = np.array(y)

    logging.info("Training meta-classifier...")
    meta_classifier = RandomForestClassifier(n_estimators=100)
    meta_classifier.fit(X, y)

    logging.info("Running MIA on the synthetic dataset...")
    synthetic_data_prepared = prepare_data(synthetic_data) # Prepare the synthetic data
    synthetic_numeric_features = synthetic_data_prepared.select_dtypes(include=[np.number]).mean().values.reshape(1, -1)
    if synthetic_numeric_features.shape[1] != X.shape[1]: 
        raise ValueError(f"Feature mismatch: synthetic_numeric_features has {synthetic_numeric_features.shape[1]} features, but the classifier expects {X.shape[1]} features.")
    
    prediction = meta_classifier.predict(synthetic_numeric_features)
    logging.info('Prediction: %s', 'IN' if prediction[0] == 1 else 'OUT')

    # Identify the specific data points resembling the original dataset
    in_indices = predict_with_indices(meta_classifier, X)
    logging.info("Indices of data points resembling original dataset: %s", in_indices)

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python mia_attack.py <relative/path/to/synthetic_data> <relative/path/to/auxiliary_data> <relative/path/to/synthetic_dataset_(optional)>")
        sys.exit(1)

    synthetic_data_path = sys.argv[1]
    auxiliary_data_path = sys.argv[2]
    if len(sys.argv) == 4:
        synthetic_dataset_save_path = sys.argv[3]
        main(synthetic_data_path, auxiliary_data_path, synthetic_dataset_save_path)
    else:
        main(synthetic_data_path, auxiliary_data_path)