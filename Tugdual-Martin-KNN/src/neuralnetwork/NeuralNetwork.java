package neuralnetwork;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.util.Random;
import java.util.Scanner;

import javax.swing.JFrame;
import javax.swing.WindowConstants;

import cs107KNN.Helpers;
import cs107KNN.KNN;

public class NeuralNetwork {
	//LOAD FILES
	static double[][] trainImages = parseIDXimages(Helpers.readBinaryFile("datasets/5000-per-digit_images_train")) ;
	static int[][] trainLabels = parseIDXlabels(Helpers.readBinaryFile("datasets/5000-per-digit_labels_train")) ;
	static double[][] testImages = parseIDXimages(Helpers.readBinaryFile("datasets/10k_images_test")) ;
	static int[][] testLabels = parseIDXlabels(Helpers.readBinaryFile("datasets/10k_labels_test")) ;

	//[LAYERS][NEURONS IN THAT LAYER][WEIGHTS CONNECTING THAT NEURON TO PREVIOUS LAYER]
	static double[][][] network;
	static double[][] output;
	static double[][] bias;
	static double[][] neuronError;
	static double[][] outputDerivative;

	static double learningRate;
	static int LAYER_SIZE;

	//Random thing that doesn't work
	static Random random; 

	//Panel on which to draw
	static neuronPanel panel;

	//Speed of the iterations in ms
	static int waitTime = 500;

	public static void main(String[] args) {
		//Create random object
		random = new Random();
		//Create neural network
		initializeNetwork();
	}

	public static void initializeNetwork() {

		//Setup the neural network
		Scanner scan = new Scanner(System.in);
		//Learning rate
		System.out.println("Learning rate of the network (For gradient descent) (Choose around 0.5 - 0.1)");
		learningRate = scan.nextDouble();
		//Amount of layers and neurons
		System.out.println("Amount of hidden layers wanted in the neural network");
		int hiddenLayers = scan.nextInt();
		if(hiddenLayers < 1) {
			System.out.println("Amount of hidden layers too low, system set hidden layers to 1");
			hiddenLayers = 1;
		}

		LAYER_SIZE = hiddenLayers + 2;

		//Includes the output and input layer
		network = new double[LAYER_SIZE][][];
		output = new double[LAYER_SIZE][];
		neuronError = new double[LAYER_SIZE][];
		outputDerivative = new double[LAYER_SIZE][];
		bias = new double[LAYER_SIZE][];

		//Special since there needs to be 28 * 28 weights for the first layer (input layer is before it) and in neuronValues
		network[0] = new double[28 * 28][1];
		output[0] = new double[28 * 28];
		neuronError[0] = new double[28 * 28];
		outputDerivative[0] = new double[28 * 28];
		bias[0] = new double[28 * 28];

		//Asks for amount of neurons for every hidden layer, and in neuronValues
		for(int i = 1; i <= hiddenLayers; i++) {
			System.out.println("Amount of neurons wanted in hidden layer " + i);
			int neurons = scan.nextInt();
			network[i] = new double[neurons][network[i - 1].length];
			output[i] = new double[neurons];
			neuronError[i] = new double[neurons];
			outputDerivative[i] = new double[neurons];
			bias[i] = new double[neurons];
		}

		//Sets amount of output neurons to ten in network and neuronValues
		network[LAYER_SIZE-1] = new double[10][network[hiddenLayers].length];
		output[LAYER_SIZE-1] = new double[10];
		neuronError[LAYER_SIZE-1] = new double[10];
		outputDerivative[LAYER_SIZE-1] = new double[10];
		bias[LAYER_SIZE-1] = new double[10];

		//Closes scanner
		scan.close();

		//Randomise the weights
		randomizeWeightsAndBiases();

		//Display the network
		show(network);

		//Train the network on some data
		train(10, 10000);

		int[] predictions = new int[700];

		int[] trueLabels = new int[testLabels.length];
		for(int i = 0; i < testLabels.length; i++) {
			for(int number = 0; number < testLabels[i].length; number++) {
				if(testLabels[i][number] == 1.0) trueLabels[i] = number;
			}
		}

		for (int i = 0 ; i < 700 ; i++) {
			forwardPropagation(testImages[i], testLabels[i]);
			predictions[i] = indexOfMax();
		}

		accuracy(predictions, trueLabels);

		Helpers.show("Neural network", testImages, predictions, trueLabels, 20, 35);
	}
	
	

	/**
	 * Train the network with batches (Don't know how to do yet)
	 *
	 * @param image vector, inputed into the neural network
	 */
	private static void train(int batchSize, int amountOfBatches) {
		double[][][] batches = new double[amountOfBatches][batchSize][];
		int[][][] labels = new int[amountOfBatches][batchSize][];
		for(int batch = 0; batch < amountOfBatches; batch++) {
			for(int image = 0; image < batchSize; image++) {
				int randomIndex = new Random().nextInt(trainImages.length);
				batches[batch][image] = trainImages[randomIndex];
				labels[batch][image] = trainLabels[randomIndex];
			}
		}

		for(int batch = 0; batch < amountOfBatches; batch++) {
			//Sum the errors of the batch to then apply gradient descent
			double sumOfErrors = 0;
			for(int image = 0; image < batchSize; image++) {
				//Propagate forward and return the MSE
				sumOfErrors += forwardPropagation(batches[batch][image], labels[batch][image]);
				backPropagation(labels[batch][image]);
				updateWeights();
			}
			//Modify the weights after the batch
			System.out.println("MSE of the batch: " + sumOfErrors / batchSize);
			neuronPanel.MSE = Math.round((sumOfErrors / batchSize) * 10000.0) / 10000.0;
		}
	}

	/**
	 * Takes an image as input and returns the output (10 big vector)
	 *
	 * @param image vector, inputed into the neural network
	 */
	private static void updateWeights() {
		for(int layer = 1; layer < LAYER_SIZE; layer++) {
			for(int neuron = 0; neuron < network[layer].length; neuron++) {

				double delta = - learningRate * neuronError[layer][neuron];

				for(int weight = 0; weight < network[layer][neuron].length; weight++) {

					network[layer][neuron][weight] += delta * output[layer - 1][weight];

				}

			}
		}
	}

	/**
	 * Takes an image as input and returns the output (10 big vector)
	 *
	 * @param image vector, inputed into the neural network
	 */
	private static void backPropagation(int[] target) {	
		//Calculate the output layer first, which is calculated differently
		for(int neuron = 0; neuron < network[LAYER_SIZE - 1].length; neuron++) {
			neuronError[LAYER_SIZE - 1][neuron] = (output[LAYER_SIZE - 1][neuron] - target[neuron]) * outputDerivative[LAYER_SIZE - 1][neuron];
		}

		//Proceed from forward to backwards, already having affected the last layer, thus layer - 2
		for(int layer = LAYER_SIZE - 2; layer >= 0; layer--) {
			//This works as it should: Sum the weights of a neuron to the nextNeurons times the respective error of the next neuron, and time it by outDeriv to get neuronError
			for(int neuron = 0; neuron < network[layer].length; neuron++) {
				double sum = 0;
				for(int nextNeuron = 0; nextNeuron < network[layer + 1].length; nextNeuron++) {
					sum += network[layer + 1][nextNeuron][neuron] * neuronError[layer + 1][nextNeuron];
				}
				neuronError[layer][neuron] = sum * outputDerivative[layer][neuron];
			}
		}
	}

	/**
	 * Takes an image as input and returns the output (10 big vector)
	 *
	 * @param image vector, inputed into the neural network
	 */
	private static double forwardPropagation(double[] image, int[] target) {
		assert image.length == 28*28;
		output[0] = image;

		//Set the values of the first layer to the values of the image vector
		for(int neuron = 0; neuron < network[0].length; neuron++) {
			outputDerivative[0][neuron] = output[0][neuron] * (1 - output[0][neuron]);
		}

		//Calculate the values of the neurons for that image until the last layer
		for(int layer = 1; layer < LAYER_SIZE; layer++) {
			for(int neuron = 0; neuron < network[layer].length; neuron++) {

				double totalOfThatNeuron = bias[layer][neuron];
				for(int weight = 0; weight < network[layer][neuron].length; weight++) {
					//Sum up all the outputs of the previous layer with their respective weights
					totalOfThatNeuron += output[layer - 1][weight] * network[layer][neuron][weight];
				}
				output[layer][neuron] = sigmoid(totalOfThatNeuron);
				outputDerivative[layer][neuron] = output[layer][neuron] * (1 - output[layer][neuron]);
			}
		}

		neuronPanel.neuronValues = output;
		neuronPanel.network = network;
		neuronPanel.input = image;
		neuronPanel.target = target;
		
		int expected = 0;
		for(int number = 0; number < target.length; number++) {
			if(target[number] == 1.0) expected = number;
		}
		if(indexOfMax() == expected) {
			neuronPanel.correct = true;
		} else {
			neuronPanel.correct = false;
		}
		
		panel.repaint();

		try {
			Thread.sleep(waitTime);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		//Return the values of the last layer of neurons, the output layer
		return MeanSquaredError(output[LAYER_SIZE - 1], target);
	}

	private static double MeanSquaredError(double[] output, int[] target) {
		double total = 0;
		for(int guess = 0; guess < output.length; guess++) {
			total += Math.pow(output[guess] - target[guess], 2);
		}
		total /= output.length;
		return total;
	}

	/**
	 * Applies the sigmoid function to the input
	 *
	 * @param sum, 
	 */
	private static double sigmoid(double sum) {
		return 1d / (1 + Math.exp(-sum));
	}

	/**
	 * Gives random values between -1.0 and 1.0 to the weights
	 *
	 * @param tensor, the tensor to which we randomise the weights
	 */
	private static void randomizeWeightsAndBiases() {
		for(int layer = 1; layer < LAYER_SIZE; layer++) {
			for(int neuron = 0; neuron < network[layer].length; neuron++) {
				for(int weight = 0; weight < network[layer][neuron].length; weight++) {
					network[layer][neuron][weight] = new Random().nextDouble() * 2 - 1d;
					bias[layer][neuron] = new Random().nextDouble() * 2 - 1d;
				}
			}
		}
	}

	/**
	 * Returns the array of images as double[][]
	 * 
	 * @param data the binary content of the file 
	 * 
	 * @return the array of images as doubles[width][height]
	 */
	public static double[][] parseIDXimages(byte[] data) {
		int amountOfImages = KNN.extractInt(data[4], data[5], data[6], data[7]);
		int heightOfImages = KNN.extractInt(data[8], data[9], data[10], data[11]);
		int widthOfImages = KNN.extractInt(data[12], data[13], data[14], data[15]);

		System.out.println("Amount: " + amountOfImages + ", height: " + heightOfImages + ", width: " + widthOfImages);

		double[][] tensor = new double[amountOfImages][widthOfImages * heightOfImages];
		for(int image = 0; image < amountOfImages; image++) {
			for(int y = 0; y < heightOfImages; y++) {
				for(int x = 0; x < widthOfImages; x++) {
					//For one image, there are 28 x 28 pixels, plus for one row, there are 28 pixels
					int displacement = image * (widthOfImages * heightOfImages) + y * heightOfImages + x;
					byte smallToBigEndian = (byte) 128;
					//data is between 0 and 256 with 0 being black, thus add 128 to make into -128 to 128 and then divide by 255 to get inbetween 0.0 and 1.0
					tensor[image][y * widthOfImages + x] = ((double) ((byte) (data[16 + displacement] + smallToBigEndian)) + 128.0) / 255.0;
				}
			}
		}
		return tensor;
	}

	/**
	 * Returns a list of one-hot encoded vectors of the labels of the images
	 *
	 * @param data the binary content of the file
	 *
	 * @return the parsed labels
	 */
	public static int[][] parseIDXlabels(byte[] data) {
		int amountOfLabels = KNN.extractInt(data[4], data[5], data[6], data[7]);
		int[][] vector = new int[amountOfLabels][10];
		for(int image = 0; image < amountOfLabels; image++) {
			for(int digits = 0; digits < vector[image].length; digits++) {
				vector[image][digits] = 0;
			}
			vector[image][data[8 + image]] = 1;
		}
		return vector;
	}

	/**
	 * @brief Returns the index of the largest element in the array
	 * 
	 * @param array an array of integers
	 * 
	 * @return the index of the largest integer
	 */
	public static byte indexOfMax() {
		int max = 0;
		for(int i = 0; i < output[LAYER_SIZE - 1].length; i++) {
			if(output[LAYER_SIZE - 1][i] > output[LAYER_SIZE - 1][max]) max = i;
		}
		return (byte) max;
	}

	/**
	 * Creates the window for the viewing of the neural network
	 *
	 * @param network, The network to display
	 */
	public static void show(double[][][] network) {
		JFrame frame = new JFrame("Neural network");
		frame.addKeyListener(new KeyListener() {
			@Override
			public void keyPressed(KeyEvent e) {
				if(e.getKeyCode() == KeyEvent.VK_UP) {
					NeuralNetwork.waitTime += 100;
				} else if (e.getKeyCode() == KeyEvent.VK_DOWN) {
					if(NeuralNetwork.waitTime >= 100) NeuralNetwork.waitTime -= 100;
				}
			}

			@Override
			public void keyReleased(KeyEvent e) {

			}

			@Override
			public void keyTyped(KeyEvent e) {

			}

		});
		frame.setSize(1920, 1080);
		frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		frame.setResizable(false);
		panel = new neuronPanel(network);
		frame.add(panel);
		frame.pack();
		frame.setVisible(true);
		frame.setAlwaysOnTop(true);
		frame.setAlwaysOnTop(false);
		frame.requestFocus();
	}
	
	/**
	 * Computes accuracy between two arrays of predictions
	 * 
	 * @param predictedLabels the array of labels predicted by the algorithm
	 * @param trueLabels      the array of true labels
	 * 
	 * @return the accuracy of the predictions. Its value is in [0, 1]
	 */
	public static double accuracy(int[] predictedLabels, int[] trueLabels) {
		double accuracy = 0;
		for(int prediction = 0; prediction < predictedLabels.length; prediction++) {
			if(predictedLabels[prediction] == trueLabels[prediction]) {
				accuracy += 1d;
			}
		} 
		accuracy /= predictedLabels.length;
		System.out.println("Neural Network acc: " + accuracy * 100);
		return accuracy * 100;
	}
}
