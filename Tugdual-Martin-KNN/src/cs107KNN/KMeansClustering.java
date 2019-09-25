package cs107KNN;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

public class KMeansClustering {
	public static void main(String[] args) {
		//We are going to have 1000 centroids, and iterate through the averaging 20 times
		int K = 1000;
		int maxIters = 20;

		//Load the 10000 initial images
		byte[][][] images = KNN.parseIDXimages(Helpers.readBinaryFile("datasets/1000-per-digit_images_train"));
		byte[] labels = KNN.parseIDXlabels(Helpers.readBinaryFile("datasets/1000-per-digit_labels_train"));

		//Reduce the images to 1000
		byte[][][] reducedImages = KMeansReduce(images, K, maxIters);

		//Assign them to their respective labels
		byte[] reducedLabels = new byte[reducedImages.length];
		
		for (int i = 0; i < reducedLabels.length; i++) {
			reducedLabels[i] = KNN.knnClassify(reducedImages[i], images, labels, 5);
			System.out.println("Classified " + (i + 1) + " / " + reducedImages.length);
		}

		//Write the images and labels to their respective files
		Helpers.writeBinaryFile("datasets/reduced10Kto1K_images", encodeIDXimages(reducedImages));
		Helpers.writeBinaryFile("datasets/reduced10Kto1K_labels", encodeIDXlabels(reducedLabels));
	}

	/**
	 * @brief Encodes a tensor of images into an array of data ready to be written on a file
	 * 
	 * @param images the tensor of image to encode
	 * 
	 * @return the array of byte ready to be written to an IDX file
	 */
	public static byte[] encodeIDXimages(byte[][][] images) {
		//Set the headers
		int magicNumber = 2051;
		int amountOfImages = images.length;
		int heightOfImages = images[0].length;
		int widthOfImages = images[0][0].length;

		//Output the amount of images, width and height
		System.out.println("Amount: " + amountOfImages + ", height: " + heightOfImages + ", width: " + widthOfImages);

		//Size of data = 16 first bytes, plus the amount of images each containing width * height bytes
		byte[] data = new byte[16 + amountOfImages * (widthOfImages * heightOfImages)];

		//Encode the magic number, amount of images, height and width
		encodeInt(magicNumber, data, 0);
		encodeInt(amountOfImages, data, 4);
		encodeInt(heightOfImages, data, 8);
		encodeInt(widthOfImages, data, 12);

		//Transform tensor into vector
		for(int image = 0; image < amountOfImages; image++) {
			for(int y = 0; y < heightOfImages; y++) {
				for(int x = 0; x < widthOfImages; x++) {
					
					//For one image, there are 28 x 28 pixels, plus for one row, there are 28 pixels
					int displacement = (image * (widthOfImages * heightOfImages)) + (y * heightOfImages) + x;
					byte shift = (byte) 128;
					System.out.println(displacement + " " + (byte) (images[image][y][x] + shift));
					data[16 + displacement] = (byte) (images[image][y][x] + shift);
				}
			}
		}
		return data;
	}

	/**
	 * @brief Prepares the array of labels to be written on a binary file
	 * 
	 * @param labels the array of labels to encode
	 * 
	 * @return the array of bytes ready to be written to an IDX file
	 */
	public static byte[] encodeIDXlabels(byte[] labels) {
		//Set the headers
		int magicNumber = 2049;
		System.out.println(labels.length);
		int amountOfLabels = labels.length;

		//Output the amount of labels
		System.out.println("Amount of labels: " + amountOfLabels);

		//Size of data = headers + amountOfLabels
		byte[] data = new byte[16 + amountOfLabels];

		//Encode the magic number, amount of labels
		encodeInt(magicNumber, data, 0);
		encodeInt(amountOfLabels, data, 4);

		for(int label = 0; label < amountOfLabels; label++) {
			data[8 + label] = labels[label];
		}
		return data;
	}

	/**
	 * @brief Decomposes an integer into 4 bytes stored consecutively in the destination
	 * array starting at position offset
	 * 
	 * @param n the integer number to encode
	 * @param destination the array where to write the encoded int
	 * @param offset the position where to store the most significant byte of the integer,
	 * the others will follow at offset + 1, offset + 2, offset + 3
	 */
	public static void encodeInt(int n, byte[] destination, int offset) {
		//Bytes in java are signed, we need to place them in the array
		destination[offset + 0] = (byte) ((n >> 24) & 0xFF);
		destination[offset + 1] = (byte) ((n >> 16) & 0xFF);
		destination[offset + 2] = (byte) ((n >> 8) & 0xFF);
		destination[offset + 3] = (byte) ((n >> 0) & 0xFF);
	}

	/**
	 * @brief Runs the KMeans algorithm on the provided tensor to return size elements.
	 * 
	 * @param images the tensor of images to reduce
	 * @param size the number of images in the reduced dataset
	 * @param maxIters the number of iterations of the KMeans algorithm to perform
	 * 
	 * @return the tensor containing the reduced dataset
	 */
	public static byte[][][] KMeansReduce(byte[][][] images, int size, int maxIters) {
		//Returns an array with a random number going from 0 to K for each image, assigning it to a random representative
		int[] assignments = new Random().ints(images.length, 0, size).toArray();

		//We have K centroids
		byte[][][] centroids = new byte[size][][];

		//We initialize lmao
		initialize(images, assignments, centroids);

		
		
		//Recompute the centroids and image averages 20 times
		int nIter = 0;
		while (nIter < maxIters) {
			
			double start = System.currentTimeMillis();
			// Step 1: Assign image to closest centroid
			recomputeAssignments(images, centroids, assignments);
			
			// Step 2: Recompute centroids as average of points
			recomputeCentroids(images, centroids, assignments);

			Helpers.show("Memes", centroids, 20, 20);
			
			System.out.println("Recomputed assignments and centroids, took " + (System.currentTimeMillis() - start) / 1000d + " seconds");
			System.out.println("KMeans completed iteration " + (nIter + 1) + " / " + maxIters);

			nIter++;
		}

		return centroids;
	}

	/**
	 * @brief Computes the L2 distance of two images
	 * 
	 * @param a, b two images of same dimensions
	 * 
	 * @return the squared euclidean distance between the two images
	 */
	public static double euclideanDistance(byte[][] a, byte[][] b) {
		float distance = 0f;

		for(int y = 0; y < a.length; y++) {

			for(int x = 0; x < a[y].length; x++) {
				distance += (float) Math.pow(a[y][x] - b[y][x], 2);
			}
		}
		return Math.sqrt(distance);
	}

	/**
	 * @brief Assigns each image to the cluster whose centroid is the closest.
	 * It modifies.
	 * 
	 * @param images the tensor of images to cluster
	 * @param centroids the tensor of centroids that represent the cluster of images
	 * @param assignments the vector indicating to what cluster each image belongs to.
	 *  if j is at position i, then image i belongs to cluster j
	 */
	public static void recomputeAssignments(byte[][][] images, byte[][][] centroids, int[] assignments) {
		//Find the smallest euclidianDistance cluster and assign each image to it
		for(int image = 0; image < images.length; image++) {
			
			byte[][] currentImage = images[image];
			int closestCentroidIndex = 0;
			double smallesteuclideanDistance = euclideanDistance(currentImage, centroids[0]);
			
			for(int centroid = 0; centroid < centroids.length; centroid++) {
				//Calculate the distance between the image and the current centroid
				double currentTestedDistance = euclideanDistance(currentImage, centroids[centroid]);
				
				//Check if it's less than the smallest current one, if it is set the index of closest to the new one as well as the smallestDistance
				if(currentTestedDistance <= smallesteuclideanDistance) {
					smallesteuclideanDistance = currentTestedDistance;
					closestCentroidIndex = centroid;
				}
			}
			
			//Assign that image to the closest resembling centroid
			assignments[image] = closestCentroidIndex;
		}
	}

	/**
	 * @brief Computes the centroid of each cluster by averaging the images in the cluster
	 * 
	 * @param tensor the tensor of images to cluster
	 * @param centroids the tensor of centroids that represent the cluster of images
	 * @param assignments the vector indicating to what cluster each image belongs to.
	 *  if j is at position i, then image i belongs to cluster j
	 */
	public static void recomputeCentroids(byte[][][] tensor, byte[][][] centroids, int[] assignments) {
		//For each centroid
		for(int centroid = 0; centroid < centroids.length; centroid++) {
			//Create an image that will be the new centroid
			int[][] averageImage = new int[tensor[0].length][tensor[0][0].length];
			for(int y = 0; y < averageImage.length; y++) {
				for(int x = 0; x < averageImage[y].length; x++) {
					averageImage[y][x] = 0;
				}
			}
			
			//Add the images associated with the current centroid
			int totalImagesAssignedToThatCentroid = 0;
			for(int image = 0; image < assignments.length; image++) {
				if(assignments[image] == centroid) {
					totalImagesAssignedToThatCentroid++;
					for(int y = 0; y < averageImage.length; y++) {
						for(int x = 0; x < averageImage[y].length; x++) {
							averageImage[y][x] += tensor[image][y][x];
						}
					}
				}
			}
			
			//Average each pixel of the new image, and set that to the new centroid
			for(int y = 0; y < averageImage.length; y++) {
				for(int x = 0; x < averageImage[y].length; x++) {
					centroids[centroid][y][x] = (byte) ((double)(averageImage[y][x] / (double) totalImagesAssignedToThatCentroid));
				}
			}
		}
	}

	/**
	 * Initializes the centroids and assignments for the algorithm.
	 * The assignments are initialized randomly and the centroids
	 * are initialized by randomly choosing images in the tensor.
	 * 
	 * @param tensor the tensor of images to cluster
	 * @param assignments the vector indicating to what cluster each image belongs to.
	 * @param centroids the tensor of centroids that represent the cluster of images
	 *  if j is at position i, then image i belongs to cluster j
	 */
	public static void initialize(byte[][][] tensor, int[] assignments, byte[][][] centroids) {
		Set<Integer> centroidIds = new HashSet<>();

		Random r = new Random("cs107-2018".hashCode());

		while (centroidIds.size() != centroids.length)
			centroidIds.add(r.nextInt(tensor.length));

		Integer[] cids = centroidIds.toArray(new Integer[] {});

		for (int i = 0; i < centroids.length; i++)
			centroids[i] = tensor[cids[i]];

		for (int i = 0; i < assignments.length; i++)
			assignments[i] = cids[r.nextInt(cids.length)];

	}
}
