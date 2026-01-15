import os
import numpy as np
from PIL import Image
import pickle

class SimpleNeuralNetwork:
    """A simple feedforward neural network implemented from scratch"""
    
    def __init__(self, input_size=81, hidden_size=32, output_size=1):
        # Initialize weights with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        return x * (1 - x)
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def forward(self, X):
        """Forward propagation"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output, learning_rate=0.01):
        """Backpropagation"""
        m = X.shape[0]
        
        # Output layer gradient
        dz2 = output - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer gradient
        dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs=500, learning_rate=0.1, verbose=True):
        """Train the network"""
        history = {'loss': [], 'accuracy': []}
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Calculate loss (binary cross-entropy)
            loss = -np.mean(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8))
            
            # Calculate accuracy
            predictions = (output > 0.5).astype(int)
            accuracy = np.mean(predictions == y)
            
            # Backward pass
            self.backward(X, y, output, learning_rate)
            
            history['loss'].append(loss)
            history['accuracy'].append(accuracy)
            
            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy*100:.2f}%")
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        output = self.forward(X)
        return output
    
    def save(self, filepath):
        """Save model to file"""
        model_data = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.W1 = model_data['W1']
        self.b1 = model_data['b1']
        self.W2 = model_data['W2']
        self.b2 = model_data['b2']

def load_image(image_path):
    """Load a 9x9 PNG image and convert to binary array"""
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img_array = np.array(img)
    # Convert to binary: white (255) -> 0, black (0) -> 1
    binary_array = (img_array < 128).astype(np.float32)
    # Flatten to 81 features
    return binary_array.flatten()

def load_dataset(image_dir):
    """Load all images from subdirectories and create dataset with labels"""
    images = []
    labels = []
    
    # Load conifers from conifers/ subdirectory
    conifers_dir = os.path.join(image_dir, 'conifers')
    if os.path.exists(conifers_dir):
        for filename in sorted(os.listdir(conifers_dir)):
            if filename.endswith('.png'):
                image_path = os.path.join(conifers_dir, filename)
                img_data = load_image(image_path)
                images.append(img_data)
                labels.append([1])  # Conifer label
    
    # Load non-conifers from non-conifers/ subdirectory
    non_conifers_dir = os.path.join(image_dir, 'non-conifers')
    if os.path.exists(non_conifers_dir):
        for filename in sorted(os.listdir(non_conifers_dir)):
            if filename.endswith('.png'):
                image_path = os.path.join(non_conifers_dir, filename)
                img_data = load_image(image_path)
                images.append(img_data)
                labels.append([0])  # Non-conifer label
    
    if len(images) == 0:
        raise ValueError(f"No images found in {image_dir}/conifers/ or {image_dir}/non-conifers/")
    
    return np.array(images), np.array(labels)

def train_test_split(X, y, test_size=0.2, random_state=42):
    """Simple train-test split"""
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def train_model(image_dir, model_save_path='conifer_model.pkl'):
    """Train the neural network on the dataset"""
    
    print("Loading dataset...")
    X, y = load_dataset(image_dir)
    
    print(f"Loaded {len(X)} images")
    print(f"Conifers: {np.sum(y)}, Non-conifers: {len(y) - np.sum(y)}")
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print(f"\nTraining set: {len(X_train)} images")
    print(f"Test set: {len(X_test)} images")
    
    # Create and train model
    print("\nCreating neural network...")
    print("Architecture: 81 -> 32 -> 1")
    model = SimpleNeuralNetwork(input_size=81, hidden_size=32, output_size=1)
    
    print("\nTraining model...")
    history = model.train(X_train, y_train, epochs=500, learning_rate=0.1, verbose=True)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_predictions = model.predict(X_test)
    test_predictions_binary = (test_predictions > 0.5).astype(int)
    test_accuracy = np.mean(test_predictions_binary == y_test)
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
    
    # Save the model
    model.save(model_save_path)
    print(f"\nModel saved to '{model_save_path}'")
    
    return model, history

def predict_image(model_path, image_path):
    """Load a trained model and predict if an image is a conifer"""
    model = SimpleNeuralNetwork()
    model.load(model_path)
    
    img_data = load_image(image_path)
    img_data = np.expand_dims(img_data, axis=0)  # Add batch dimension
    
    prediction = model.predict(img_data)[0][0]
    
    print(f"\nPrediction for '{image_path}':")
    print(f"Conifer probability: {prediction*100:.2f}%")
    
    if prediction > 0.5:
        print("Classification: CONIFER")
    else:
        print("Classification: NON-CONIFER")
    
    return prediction

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Training: python script.py train <image_directory>")
        print("  Prediction: python script.py predict <model_path> <image_path>")
        print("\nExample:")
        print("  python script.py train png_images/")
        print("  python script.py predict conifer_model.pkl test_image.png")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == 'train':
        if len(sys.argv) < 3:
            print("Please provide image directory")
            print("Usage: python script.py train <image_directory>")
            sys.exit(1)
        
        image_directory = sys.argv[2]
        train_model(image_directory)
    
    elif mode == 'predict':
        if len(sys.argv) < 4:
            print("Please provide model path and image path")
            print("Usage: python script.py predict <model_path> <image_path>")
            sys.exit(1)
        
        model_path = sys.argv[2]
        image_path = sys.argv[3]
        predict_image(model_path, image_path)
    
    else:
        print(f"Unknown mode: {mode}")
        print("Use 'train' or 'predict'")
