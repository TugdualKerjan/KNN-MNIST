package cs107KNN;

import java.util.Arrays;

public class KNN {

	static byte[][][] trainImages = parseIDXimages(Helpers.readBinaryFile("datasets/1000-per-digit_images_train")) ;
	static byte[] trainLabels = parseIDXlabels(Helpers.readBinaryFile("datasets/1000-per-digit_labels_train")) ;
	static byte[][][] testImages = parseIDXimages(Helpers.readBinaryFile("datasets/reduced10Kto1K_images")) ;
	static byte[] testLabels = parseIDXlabels(Helpers.readBinaryFile("datasets/reduced10Kto1K_labels")) ;

	public static void main(String[] args) {
		//Amount of tested images and k closest neighbours
		int TESTS = 700;
		int K = 5;
		
		//Starting time
		long startTime = System.currentTimeMillis();

		//Calculate the predictions 
		byte[] predictions = new byte[TESTS];
		for (int i = 0 ; i < TESTS ; i++) {
			System.out.println("Progress: " + (int) (((double) i/TESTS) * 100) + "%");
			predictions[i] = knnClassify(testImages[i], trainImages , trainLabels , K) ;
		}

		//Obtain the time taken in seconds
		double timeTaken = (System.currentTimeMillis() - startTime) / 1000d;

		//Check the accuracy of the KNN
		double acc = accuracy(predictions, Arrays.copyOfRange(testLabels , 0, TESTS));

		//Print results
		System.out.println("\n\n\n\n");
		System.out.println("Accuracy: " + acc * 100 + "% correct, \nTime taken: " + timeTaken + " seconds, \nTime per image: " + timeTaken/TESTS);


		//Show the images and their labels
		Helpers.show("Test", testImages , predictions , testLabels , 20, 30);
	}

	/**
	 * Composes four bytes into an integer using big endian convention.
	 *
	 * @param bXToBY The byte containing the bits to store between positions X and Y
	 * 
	 * @return the integer having form [ b31ToB24 | b23ToB16 | b15ToB8 | b7ToB0 ]
	 */
	public static int extractInt(byte b31ToB24, byte b23ToB16, byte b15ToB8, byte b7ToB0) {
		//Shift the bits to their corresponding place, these bits are signed
		int result = (((b31ToB24 & 0xFF) << 24) | ((b23ToB16 & 0xFF) << 16) | ((b15ToB8 & 0xFF) << 8) | ((b7ToB0 & 0xFF) << 0));
		return result;
	}

	/**
	 * Parses an IDX file containing images
	 *
	 * @param data the binary content of the file
	 *
	 * @return A tensor of images
	 */
	public static byte[][][] parseIDXimages(byte[] data) {
		if(data.length < 16) return null;
		
		if(extractInt(data[0], data[1], data[2], data[3]) != 2051) return null;
		
		//Fetch the initial data
		int amountOfImages = extractInt(data[4], data[5], data[6], data[7]);
		int heightOfImages = extractInt(data[8], data[9], data[10], data[11]);
		int widthOfImages = extractInt(data[12], data[13], data[14], data[15]);

		//Output the amount of images, width and height
		System.out.println("Amount: " + amountOfImages + ", height: " + heightOfImages + ", width: " + widthOfImages);

		//Iterate through the data and return the tensor of images
		byte[][][] tensor = new byte[amountOfImages][widthOfImages][heightOfImages];
		for(int image = 0; image < amountOfImages; image++) {
			for(int y = 0; y < heightOfImages; y++) {
				for(int x = 0; x < widthOfImages; x++) {
					//For one image, there are 28 x 28 pixels, plus for one row, there are 28 pixels
					int displacement = (image * (widthOfImages * heightOfImages)) + (y * heightOfImages) + x;
					//Shift the interval down 128, such that the values align with [-128; 127]
					byte smallToBigEndian = (byte) 128;
					tensor[image][y][x] = (byte) (data[16 + displacement] - smallToBigEndian);
				}
			}
		}
		return tensor;
	}

	/**
	 * Parses an idx images containing labels
	 *
	 * @param data the binary content of the file
	 *
	 * @return the parsed labels
	 */
	public static byte[] parseIDXlabels(byte[] data) {
		if(data.length < 8) return null;
		
		//Verify that this is a 'fichier ettiquettes'
		if(extractInt(data[0], data[1], data[2], data[3]) != 2049) return null;
		
		//Get the amount of labels to set size of returned labels vector
		int amountOfLabels = extractInt(data[4], data[5], data[6], data[7]);

		byte[] labels = new byte[amountOfLabels];
		for(int i = 0; i < amountOfLabels; i++) {
			labels[i] = data[8 + i];
		}
		return labels;
	}

	/**
	 * @brief Computes the squared L2 distance of two images
	 * 
	 * @param a, b two images of same dimensions
	 * 
	 * @return the squared euclidean distance between the two images
	 */
	public static float squaredEuclideanDistance(byte[][] a, byte[][] b) {
		assert a.length == b.length;

		float distance = 0f;

		for(int y = 0; y < a.length; y++) {
			assert a[y].length == b[y].length;

			for(int x = 0; x < a[y].length; x++) {
				distance += (float) Math.pow(a[y][x] - b[y][x], 2);
			}
		}
		return distance;
	}

	/**
	 * @brief Computes the average values of pixels in an image
	 * 
	 * @param a, image represented by matrix 
	 * 
	 * @return the average of the values of the pixels in the matrix a
	 */
	public static float averageOfPixelValues(byte[][] a) {
		int average = 0;
		for(byte[] y : a) {
			for(byte x : y) {
				average += x;
			}
		}
		return average /= (double) (a.length * a[0].length);
	}

	/**
	 * @brief Computes the inverted similarity between 2 images.
	 * 
	 * @param a, b two images of same dimensions
	 * 
	 * @return the inverted similarity between the two images
	 */
	public static float invertedSimilarity(byte[][] a, byte[][] b) {
		assert a.length == b.length;

		float averageA = averageOfPixelValues(a);
		float averageB = averageOfPixelValues(b);
		float sumA = 0f;
		float sumB = 0f;

		float numerator = 0f;

		//Iterate through the matrixes to obtain numerator and denominator
		for(int y = 0; y < a.length; y++) {
			for(int x = 0; x < a[0].length; x++) {
				//Denominator, no byte overflow since calculation is int to int
				sumA += (float) Math.pow(a[y][x] - averageA, 2);
				sumB += (float) Math.pow(b[y][x] - averageB, 2);

				//Numerator
				numerator += (a[y][x] - averageA) * (b[y][x] - averageB);
			}
		}

		float denominator = (float) Math.sqrt(sumA * sumB);

		//Return 2 if denominator is equal to 0, otherwise return the inverted similarity
		return (denominator == 0f) ? 2f : 1f - (numerator / denominator);
	}

	/**
	 * @brief Quicksorts and returns the new indices of each value.
	 * 
	 * @param values the values whose indices have to be sorted in non decreasing order
	 * 
	 * @return the array of sorted indices
	 * 
	 *         Example: values = quicksortIndices([3, 7, 0, 9]) gives [2, 0, 1, 3]
	 */
	public static int[] quicksortIndices(float[] values) {
		int[] indices = new int[values.length];

		//Set indices equal to their index value
		for(int i = 0; i < indices.length; i++) {
			indices[i] = i;
		}

		//Quicksort that quality index
		quicksortIndices(values, indices, 0, indices.length - 1);
		return indices;
	}

	/**
	 * @brief Sorts the provided values between two indices while applying the same
	 *        transformations to the array of indices
	 * 
	 * @param values  the values to sort
	 * @param indices the indices to sort according to the corresponding values
	 * @param         low, high are the **inclusive** bounds of the portion of array
	 *                to sort
	 */
	public static void quicksortIndices(float[] values, int[] indices, int low, int high) {
		//Based off the algorithm seen in the paper
		int h = high;
		int l = low;
		while(l <= h) {
			if (values[l] < values[low]) l++;
			else if (values[h] > values[low]) h--;
			else {
				swap(l, h, values, indices);
				l++;
				h--;
			}
		}
		//Call itself
		if (low < h) quicksortIndices(values, indices, low, h);
		if (high > l) quicksortIndices(values, indices, l, high);
	}

	/**
	 * @brief Swaps the elements of the given arrays at the provided positions
	 * 
	 * @param         i, j the indices of the elements to swap
	 * @param values  the array floats whose values are to be swapped
	 * @param indices the array of ints whose values are to be swapped
	 */
	public static void swap(int i, int j, float[] values, int[] indices) {
		//swap the float array
		float tempFloat = values[i];
		values[i] = values[j];
		values[j] = tempFloat;

		//swap the int array
		int tempInt = indices[i];
		indices[i] = indices[j];
		indices[j] = tempInt;
	}

	/**
	 * @brief Returns the index of the largest element in the array
	 * 
	 * @param array an array of integers
	 * 
	 * @return the index of the largest integer
	 */
	public static int indexOfMax(int[] array) {
		//Find the index of the highest element by going through all elements of the set
		int max = 0;
		for(int i = 0; i < array.length; i++) {
			if(array[i] > array[max]) max = i;
		}
		return max;
	}

	/**
	 * The k first elements of the provided array vote for a label
	 *
	 * @param sortedIndices the indices sorted by non-decreasing distance
	 * @param labels        the labels corresponding to the indices
	 * @param k             the number of labels asked to vote
	 *
	 * @return the winner of the election
	 */
	public static byte electLabel(int[] sortedIndices, byte[] labels, int k) {
		//Store the amount of votes in this array
		int[] votes = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		for(int i = 0; i < k; i++) {
			votes[labels[sortedIndices[i]]] += 1;
		}
		//Get the most voted number and return it as a byte
		return (byte) indexOfMax(votes);
	}

	/**
	 * Classifies the symbol drawn on the provided image
	 *
	 * @param image       the image to classify
	 * @param trainImages the tensor of training images
	 * @param trainLabels the list of labels corresponding to the training images
	 * @param k           the number of voters in the election process
	 *
	 * @return the label of the image
	 */
	public static byte knnClassify(byte[][] image, byte[][][] trainImages, byte[] trainLabels, int k) {
		//Get the distances of this image with all the other images in the trainImages array
		float[] similarity = new float[trainImages.length];
		for(int i = 0; i < trainImages.length; i++) {
			similarity[i] = squaredEuclideanDistance(image, trainImages[i]);
		}

		//Get the array of the most similar indexes 
		int[] rearangedIndexOfDistances = quicksortIndices(similarity);

		//Elect the most often image in the k first most similar elements
		return electLabel(rearangedIndexOfDistances, trainLabels, k);
	}

	/**
	 * Computes accuracy between two arrays of predictions
	 * 
	 * @param predictedLabels the array of labels predicted by the algorithm
	 * @param trueLabels      the array of true labels
	 * 
	 * @return the accuracy of the predictions. Its value is in [0, 1]
	 */
	public static double accuracy(byte[] predictedLabels, byte[] trueLabels) {
		//Count the amount of correct answers in the tested images
		double accuracy = 0;
		for(int prediction = 0; prediction < predictedLabels.length; prediction++) {
			if(predictedLabels[prediction] == trueLabels[prediction]) {
				accuracy += 1d;
			}
		}

		//Transform into a ratio
		accuracy = (accuracy / predictedLabels.length);
		return accuracy;
	}
}
